#!/usr/bin/env python3
"""Run one C1/M01 codec correctness and stage-timing point."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from zlib import crc32


def block_ids_for_runs(blocks: int, requested_runs: int) -> list[int]:
    if blocks <= 0 or requested_runs <= 0:
        raise ValueError("blocks and runs must be positive")
    runs = min(blocks, requested_runs)
    lengths = [blocks // runs + (index < blocks % runs) for index in range(runs)]
    result = []
    start = 0
    for length in lengths:
        result.extend(range(start, start + length))
        start += length + 1
    return result


def count_runs(values: list[int]) -> int:
    return int(bool(values)) + sum(
        current != previous + 1 for previous, current in zip(values, values[1:])
    )


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    mean = statistics.fmean(ordered)
    half = 1.96 * statistics.stdev(ordered) / math.sqrt(len(ordered)) if len(ordered) > 1 else 0.0
    return {
        "mean": mean,
        "mean_ci95_low": mean - half,
        "mean_ci95_high": mean + half,
        "p50": statistics.median(ordered),
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
    }


def load_geometry(config_path: Path) -> dict[str, int]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    heads = int(config["num_attention_heads"])
    return {
        "layers": int(config["num_hidden_layers"]),
        "kv_heads": int(config.get("num_key_value_heads", heads)),
        "head_dim": int(config.get("head_dim", int(config["hidden_size"]) // heads)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=int)
    parser.add_argument("--mode", required=True, choices=("top16", "fixed6", "raw"))
    parser.add_argument("--runs", required=True, type=int)
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.tokens <= 0 or args.tokens % args.block_tokens:
        raise ValueError("tokens must be a positive multiple of block-tokens")
    if args.library:
        import os

        os.environ["UNCHAIN_KV_SPLITZIP_LIB"] = args.library

    import torch

    from unchain_kv import splitzip_cuda

    geometry = load_geometry(args.model_config)
    if args.mode == "top16" and len(splitzip_cuda._TOP16_CODEBOOKS) // 32 < geometry["layers"]:
        raise RuntimeError("Top-16 codebooks do not cover every model layer")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    blocks = args.tokens // args.block_tokens
    block_words = args.block_tokens * geometry["kv_heads"] * geometry["head_dim"]
    block_ids = block_ids_for_runs(blocks, args.runs)
    actual_runs = count_runs(block_ids)
    ids = torch.tensor(block_ids, dtype=torch.int64, device=device)
    source = torch.empty((2, blocks, block_words), dtype=torch.bfloat16, device=device)
    cache = torch.empty(
        (2, max(block_ids) + 1, block_words), dtype=torch.bfloat16, device=device
    )
    target = torch.zeros_like(cache)
    pack = torch.empty_like(source)
    raw_bytes = source.numel() * source.element_size()
    encoded = torch.empty((raw_bytes + 1,), dtype=torch.uint8, device=device)
    received = torch.empty_like(encoded)
    raw_received = torch.empty_like(source)
    host = torch.empty((raw_bytes + 1,), dtype=torch.uint8, pin_memory=True)
    stream = torch.cuda.current_stream(device)

    def prepare(layer: int) -> None:
        words = source.view(torch.uint16)
        if args.mode == "top16":
            books = splitzip_cuda._TOP16_CODEBOOKS[layer * 32 : (layer + 1) * 32]
            words[0].fill_(int(books[0]) << 7 | 0x15)
            words[1].fill_(int(books[16]) << 7 | 0x2A)
            escape = next(value for value in range(1, 255) if value not in books)
            words.reshape(-1)[::200].fill_(escape << 7 | 0x33)
        elif args.mode == "fixed6":
            words[0].fill_(0x7F15)
            words[1].fill_(0x802A)
        else:
            words[0].fill_(0x3F95)
            words[1].fill_(0xC02A)
        cache.index_copy_(1, ids, source)

    def packed_source():
        if actual_runs == 1:
            return cache[:, block_ids[0] : block_ids[0] + blocks]
        torch.index_select(cache, 1, ids, out=pack)
        return pack

    def encode(current, layer: int) -> int:
        if args.mode == "top16":
            return int(splitzip_cuda.encode_top16(current, encoded, layer) or 0)
        if args.mode == "fixed6":
            return int(splitzip_cuda.encode_bf16(current, encoded, bits=6) or 0)
        encoded[0].fill_(6)
        encoded[1 : raw_bytes + 1].copy_(current.view(torch.uint8).reshape(-1))
        return raw_bytes + 1

    def restore(encoded_bytes: int, layer: int) -> int:
        if args.mode == "raw":
            raw_received.view(torch.uint8).reshape(-1).copy_(
                host[1:encoded_bytes], non_blocking=True
            )
            target.index_copy_(1, ids, raw_received)
            return raw_bytes
        received[:encoded_bytes].copy_(host[:encoded_bytes], non_blocking=True)
        if args.mode == "top16":
            return int(
                splitzip_cuda.decode_top16(
                    received[:encoded_bytes], target, block_ids, raw_bytes, layer
                )
                or 0
            )
        if args.mode == "fixed6":
            return int(
                splitzip_cuda.decode_fixed6(
                    received[:encoded_bytes], target, block_ids, raw_bytes
                )
                or 0
            )
        raise AssertionError("unreachable codec mode")

    def run_once(layer: int) -> tuple[dict[str, float], int, object]:
        events = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
        events[0].record(stream)
        current = packed_source()
        events[1].record(stream)
        events[2].record(stream)
        encoded_bytes = encode(current, layer)
        if encoded_bytes <= 0:
            raise RuntimeError(f"{args.mode} encoder failed")
        events[3].record(stream)
        events[4].record(stream)
        host[:encoded_bytes].copy_(encoded[:encoded_bytes], non_blocking=True)
        events[5].record(stream)
        events[6].record(stream)
        copied = restore(encoded_bytes, layer)
        events[7].record(stream)
        events[7].synchronize()
        if copied != raw_bytes:
            raise RuntimeError(f"restore copied {copied}/{raw_bytes} bytes")
        return (
            {
                "pack_ms": events[0].elapsed_time(events[1]),
                "encode_ms": events[2].elapsed_time(events[3]),
                "d2h_ms": events[4].elapsed_time(events[5]),
                "restore_ms": events[6].elapsed_time(events[7]),
                "total_ms": events[0].elapsed_time(events[7]),
            },
            encoded_bytes,
            current,
        )

    layer_checks = []
    layers = range(geometry["layers"]) if args.mode == "top16" else range(1)
    encoded_bytes = 0
    current = None
    for layer in layers:
        prepare(layer)
        _timing, encoded_bytes, current = run_once(layer)
        restored = target.index_select(1, ids)
        differences = int(
            torch.count_nonzero(
                restored.view(torch.uint16) != current.view(torch.uint16)
            ).item()
        )
        payload_mode = int(host[0].item())
        expected_mode = {"top16": 5, "fixed6": 3, "raw": 6}[args.mode]
        if differences or payload_mode != expected_mode:
            raise RuntimeError(
                f"layer {layer} mismatch: differences={differences} mode={payload_mode}"
            )
        layer_checks.append(
            {"layer": layer, "bitwise_differences": differences, "payload_mode": payload_mode}
        )

    assert current is not None
    final_layer = geometry["layers"] - 1 if args.mode == "top16" else 0
    for _ in range(args.warmup):
        run_once(final_layer)
    samples = {name: [] for name in ("pack_ms", "encode_ms", "d2h_ms", "restore_ms", "total_ms")}
    for _ in range(args.iterations):
        row, encoded_bytes, current = run_once(final_layer)
        for name, value in row.items():
            samples[name].append(value)
    restored = target.index_select(1, ids)
    final_differences = int(
        torch.count_nonzero(restored.view(torch.uint16) != current.view(torch.uint16)).item()
    )
    if final_differences:
        raise RuntimeError(f"timed restore has {final_differences} bitwise differences")
    encoded_checksum = crc32(memoryview(host[:encoded_bytes].numpy())) & 0xFFFFFFFF
    with args.model_config.open("rb") as handle:
        config_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    result = {
        "passed": True,
        "model_config": str(args.model_config),
        "model_config_sha256": config_sha256,
        "gpu": torch.cuda.get_device_name(device),
        "geometry": geometry,
        "tokens": args.tokens,
        "block_tokens": args.block_tokens,
        "blocks": blocks,
        "requested_runs": args.runs,
        "actual_runs": actual_runs,
        "mode": args.mode,
        "raw_bytes": raw_bytes,
        "encoded_bytes": encoded_bytes,
        "encoded_crc32": encoded_checksum,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "layer_checks": layer_checks,
        "final_bitwise_differences": final_differences,
        "timing_ms": {name: summary(values) for name, values in samples.items()},
        "samples_ms": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
