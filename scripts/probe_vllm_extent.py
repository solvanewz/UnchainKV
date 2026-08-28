#!/usr/bin/env python3
"""Probe vLLM BlockPool fragmentation for off/normalize/prefer modes."""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time


def count_runs(block_ids: list[int]) -> int:
    if not block_ids:
        return 0
    runs = 1
    for i in range(1, len(block_ids)):
        if block_ids[i] != block_ids[i - 1] + 1:
            runs += 1
    return runs


def self_check() -> bool:
    cases = [([], 0), ([1, 2, 3], 1), ([3, 2, 1], 3), ([1, 2, 8, 9], 2)]
    for ids, expected in cases:
        got = count_runs(ids)
        if got != expected:
            print(f"FAIL: count_runs({ids}) = {got}, expected {expected}", file=sys.stderr)
            return False
    print("self-check passed")
    return True


def normalize_released_blocks(ordered_blocks):
    blocks = list(ordered_blocks)
    movable_idx, movable = [], []
    for i, b in enumerate(blocks):
        if not b.is_null and b.block_hash is None and b.ref_cnt == 1:
            movable_idx.append(i)
            movable.append(b)
    movable.sort(key=lambda b: b.block_id)
    for idx, b in zip(movable_idx, movable):
        blocks[idx] = b
    return blocks


def make_block_pool(num_blocks, mode):
    os.environ["UNCHAIN_KV_EXTENT_ALLOC"] = mode
    from vllm.v1.core.block_pool import BlockPool

    block_pool = BlockPool(
        num_gpu_blocks=num_blocks,
        enable_caching=False,
        hash_block_size=16,
    )
    if not hasattr(block_pool, "unchain_kv_extent_alloc"):
        raise RuntimeError("formal allocator probe requires patched vLLM BlockPool")
    return block_pool


def allocate(bp, count, mode, existing=None, latencies_ns=None):
    preferred_after = None
    if mode == "prefer" and existing:
        preferred_after = existing[-1].block_id
    started = time.perf_counter_ns()
    blocks = bp.get_new_blocks(count, preferred_after=preferred_after)
    if latencies_ns is not None:
        latencies_ns.append(time.perf_counter_ns() - started)
    return blocks


def simulate_fixed(bp, request_blocks, concurrency, batches, mode, completion_order, seed):
    rng = random.Random(seed)
    allocated = []
    batch_runs = []
    fragmented = 0
    measured_runs = []

    for _ in range(batches):
        if allocated:
            free_order = list(range(concurrency))
            if completion_order == "reverse":
                free_order.reverse()
            elif completion_order == "random":
                rng.shuffle(free_order)
            for idx in free_order:
                ordered = list(reversed(allocated[idx]))
                if mode != "off":
                    ordered = normalize_released_blocks(ordered)
                bp.free_blocks(ordered)
        allocated = [allocate(bp, request_blocks, mode) for _ in range(concurrency)]
        runs = [count_runs([block.block_id for block in req]) for req in allocated]
        batch_runs.append(runs)
        measured_runs.extend(runs)
        fragmented += sum(run > 1 for run in runs)

    return {
        "fragmented_requests": fragmented,
        "mean_runs": sum(measured_runs) / len(measured_runs) if measured_runs else 0,
        "runs_by_batch": batch_runs,
    }


def simulate_mixed(bp, mixed_blocks, mixed_requests, concurrency, mode, seed):
    """Concurrency-bounded mixed-context turnover: alloc C requests at a time,
    free them in random order, re-allocate, repeat for mixed_requests turnovers."""
    rng = random.Random(seed)
    allocated = []
    all_runs = []
    fragmented = 0
    turnover = 0
    latencies_ns = []

    # Initial fill
    while len(allocated) < concurrency and turnover < mixed_requests:
        n = rng.choice(mixed_blocks)
        blocks = allocate(bp, n, mode, latencies_ns=latencies_ns)
        allocated.append(blocks)
        runs = count_runs([block.block_id for block in blocks])
        all_runs.append(runs)
        fragmented += runs > 1
        turnover += 1

    while turnover < mixed_requests:
        # Pick one to free and replace
        idx = rng.randrange(len(allocated))
        blocks = allocated[idx]
        ordered = list(reversed(blocks))
        if mode != "off":
            ordered = normalize_released_blocks(ordered)
        bp.free_blocks(ordered)
        n = rng.choice(mixed_blocks)
        allocated[idx] = allocate(bp, n, mode, latencies_ns=latencies_ns)
        runs = count_runs([block.block_id for block in allocated[idx]])
        all_runs.append(runs)
        fragmented += runs > 1
        turnover += 1

    ordered = sorted(latencies_ns)
    return {
        "fragmented_requests": fragmented,
        "mean_runs": sum(all_runs) / len(all_runs) if all_runs else 0,
        "runs": all_runs,
        "allocation_ns": latencies_ns,
        "allocation_ns_p50": statistics.median(ordered) if ordered else 0,
        "allocation_ns_p95": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)] if ordered else 0,
        "allocation_ns_max": max(ordered, default=0),
        "failed_allocations": 0,
        "active_kv_migrations": 0,
    }


def simulate_chunked(
    bp, request_blocks, chunk_blocks, concurrency, mode, reserve_blocks
):
    from unchain_kv.patch_vllm import (
        activate_reserved_blocks,
        select_extent_lease,
    )

    allocated = [[] for _ in range(concurrency)]
    reservations = [[] for _ in range(concurrency)]
    while any(len(blocks) < request_blocks for blocks in allocated):
        for index in range(concurrency):
            needed = min(chunk_blocks, request_blocks - len(allocated[index]))
            if needed <= 0:
                continue
            if reserve_blocks and not reservations[index]:
                reservations[index] = select_extent_lease(
                    bp,
                    needed,
                    reserve_blocks,
                    allocated[index][-1].block_id if allocated[index] else None,
                ) or []
            consumed = reservations[index][:needed]
            reservations[index] = reservations[index][needed:]
            remaining = needed - len(consumed)
            new_blocks = consumed + (
                allocate(bp, remaining, mode, allocated[index] + consumed)
                if remaining
                else []
            )
            activate_reserved_blocks(bp, consumed)
            allocated[index].extend(new_blocks)
    runs = [count_runs([block.block_id for block in blocks]) for blocks in allocated]
    return {
        "fragmented_requests": sum(run > 1 for run in runs),
        "mean_runs": sum(runs) / len(runs),
        "max_runs": max(runs),
        "reserved_blocks": sum(map(len, reservations)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-blocks", type=int, default=19680)
    parser.add_argument("--request-blocks", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batches", type=int, default=9)
    parser.add_argument("--mixed-blocks", type=int, nargs="+", default=[512, 1024, 2000, 512])
    parser.add_argument("--mixed-requests", type=int, default=500)
    parser.add_argument("--chunk-request-blocks", type=int, default=1488)
    parser.add_argument("--chunk-blocks", type=int, default=32)
    parser.add_argument("--reserve-blocks", type=int, default=0)
    parser.add_argument("--completion-order", default="fifo",
                        choices=["fifo", "reverse", "random"])
    parser.add_argument("--mode", default="off", choices=["off", "normalize", "prefer"])
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        sys.exit(0 if self_check() else 1)

    bp = make_block_pool(args.num_blocks, args.mode)
    fixed = simulate_fixed(
        bp, args.request_blocks, args.concurrency,
        args.batches, args.mode, args.completion_order, args.seed,
    )
    bp2 = make_block_pool(args.num_blocks, args.mode)
    mixed = simulate_mixed(
        bp2, args.mixed_blocks, args.mixed_requests,
        args.concurrency, args.mode, args.seed,
    )
    bp3 = make_block_pool(args.num_blocks, args.mode)
    chunked = simulate_chunked(
        bp3,
        args.chunk_request_blocks,
        args.chunk_blocks,
        args.concurrency,
        args.mode,
        args.reserve_blocks,
    )

    result = {
        "mode": args.mode,
        "reserve_blocks": args.reserve_blocks,
        "fixed": fixed,
        "mixed": mixed,
        "chunked": chunked,
    }
    if args.json:
        json.dump(result, sys.stdout)
        return
    print(f"mode={args.mode} order={args.completion_order}")
    print(f"  fixed: fragmented={fixed['fragmented_requests']}/{args.batches * args.concurrency} "
          f"mean_runs={fixed['mean_runs']:.2f}")
    print(f"  mixed: fragmented={mixed['fragmented_requests']} "
          f"mean_runs={mixed['mean_runs']:.2f}")
    print(f"  chunked: fragmented={chunked['fragmented_requests']}/{args.concurrency} "
          f"mean_runs={chunked['mean_runs']:.2f} max_runs={chunked['max_runs']}")


if __name__ == "__main__":
    main()
