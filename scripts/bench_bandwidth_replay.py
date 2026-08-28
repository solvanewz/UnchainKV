#!/usr/bin/env python3
"""Replay production-sized TCP and PCIe KV payloads without model compute."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import socket
import statistics
import struct
import threading
import time
from zlib import crc32


RAW_LAYER_BYTES_32K = 65_536_000
TOP16_LAYER_BYTES_32K = 49_971_205
LAYERS = 28
BLOCK_TOKENS = 16
KV_HEADS = 4
HEAD_DIM = 128
_FRAME_HEADER = struct.Struct("!II")


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def stats(values: list[float]) -> dict[str, float]:
    return {
        "p50": statistics.median(values) if values else 0.0,
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0.0),
    }


def payload_sizes(tokens: int) -> tuple[int, int]:
    if tokens <= 0 or tokens % BLOCK_TOKENS:
        raise ValueError(f"tokens must be a positive multiple of {BLOCK_TOKENS}")
    raw = 2 * tokens * KV_HEADS * HEAD_DIM * 2
    # Top-16 production payload is a fixed 0.762499... fraction at 32k.
    top16 = round(raw * TOP16_LAYER_BYTES_32K / RAW_LAYER_BYTES_32K)
    return raw, top16


def parse_address(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator or not host or not port.isdigit():
        raise ValueError("address must be HOST:PORT")
    return host, int(port)


def emit(payload: dict[str, object], output: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def _recv_checksum(conn: socket.socket, size: int) -> int | None:
    remaining = size
    buffer = bytearray(min(size, 1024 * 1024))
    view = memoryview(buffer)
    checksum = 0
    while remaining:
        read = conn.recv_into(view[: min(remaining, len(view))])
        if not read:
            return None
        checksum = crc32(view[:read], checksum)
        remaining -= read
    return checksum & 0xFFFFFFFF


def serve_tcp(
    bind: tuple[str, int], expected_frames: int, timeout_s: float
) -> dict[str, object]:
    state: dict[str, object] = {
        "frames": 0,
        "frame_bytes": 0,
        "connections": 0,
        "errors": [],
        "first_s": None,
        "last_s": None,
    }
    lock = threading.Lock()
    done = threading.Event()
    connections: list[socket.socket] = []
    threads: list[threading.Thread] = []

    def handle(conn: socket.socket) -> None:
        try:
            while not done.is_set():
                prefix = bytearray(_FRAME_HEADER.size)
                view = memoryview(prefix)
                offset = 0
                while offset < len(prefix):
                    read = conn.recv_into(view[offset:])
                    if not read:
                        return
                    offset += read
                frame_bytes, expected_checksum = _FRAME_HEADER.unpack(prefix)
                with lock:
                    if state["first_s"] is None:
                        state["first_s"] = time.perf_counter()
                checksum = _recv_checksum(conn, frame_bytes)
                if checksum is None:
                    raise ConnectionError("connection closed inside frame")
                if checksum != expected_checksum:
                    raise ValueError("tcp frame checksum mismatch")
                now = time.perf_counter()
                with lock:
                    state["last_s"] = now
                    state["frames"] = int(state["frames"]) + 1
                    state["frame_bytes"] = int(state["frame_bytes"]) + frame_bytes
                    if int(state["frames"]) >= expected_frames:
                        done.set()
        except BaseException as exc:
            with lock:
                errors = state["errors"]
                assert isinstance(errors, list)
                errors.append(str(exc))

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(bind)
    listener.listen(128)
    listener.settimeout(0.1)
    deadline = time.monotonic() + timeout_s
    try:
        while not done.is_set() and time.monotonic() < deadline:
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            connections.append(conn)
            with lock:
                state["connections"] = int(state["connections"]) + 1
            thread = threading.Thread(target=handle, args=(conn,), daemon=True)
            threads.append(thread)
            thread.start()
    finally:
        listener.close()
        if not done.is_set():
            with lock:
                errors = state["errors"]
                assert isinstance(errors, list)
                errors.append("server timeout")
        for conn in connections:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()
        for thread in threads:
            thread.join(1)

    first = state.pop("first_s")
    last = state.pop("last_s")
    elapsed = max(0.0, float(last) - float(first)) if first and last else 0.0
    state.update(
        {
            "expected_frames": expected_frames,
            "elapsed_s": elapsed,
            "frame_gbps": (
                int(state["frame_bytes"]) * 8 / elapsed / 1e9 if elapsed else 0.0
            ),
            "ok": int(state["frames"]) == expected_frames and not state["errors"],
        }
    )
    return state


def _send_tcp_request(
    peer: tuple[str, int],
    path: str,
    payload: bytes,
    raw_layer_bytes: int,
    block_count: int,
    layers: int,
    request_index: int,
    scheduled_s: float,
) -> dict[str, object]:
    from unchain_kv.tcp_data import (
        send_compressed_native_layer_blocks,
        send_native_layer_blocks,
    )

    started = time.perf_counter()
    transfer_id = f"replay-{request_index}"
    request_id = f"request-{request_index}"
    block_size = raw_layer_bytes // block_count
    for layer in range(layers):
        if path == "raw":
            send_native_layer_blocks(
                peer,
                transfer_id,
                request_id,
                layer,
                payload,
                block_size,
                block_count,
            )
        else:
            send_compressed_native_layer_blocks(
                peer,
                transfer_id,
                request_id,
                layer,
                payload,
                raw_block_size=block_size,
                block_count=block_count,
                raw_bytes=raw_layer_bytes,
                codec="splitzip_bf16",
            )
    ended = time.perf_counter()
    return {
        "request": request_index,
        "scheduled_s": scheduled_s,
        "started_s": started,
        "ended_s": ended,
        "queue_wait_s": max(0.0, started - scheduled_s),
        "service_s": ended - started,
    }


def replay_tcp(
    peer: tuple[str, int],
    path: str,
    raw_layer_bytes: int,
    top16_layer_bytes: int,
    block_count: int,
    layers: int,
    requests: int,
    warmup_requests: int,
    concurrency: int,
    offered_rps: float,
) -> dict[str, object]:
    if raw_layer_bytes % block_count:
        raise ValueError("raw layer bytes must be divisible by block count")
    payload_bytes = raw_layer_bytes if path == "raw" else top16_layer_bytes
    payload = bytes(payload_bytes)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        warmups = [
            pool.submit(
                _send_tcp_request,
                peer,
                path,
                payload,
                raw_layer_bytes,
                block_count,
                layers,
                -index - 1,
                time.perf_counter(),
            )
            for index in range(warmup_requests)
        ]
        for future in warmups:
            future.result()

        formal_start = time.perf_counter()
        futures = []
        for index in range(requests):
            scheduled = (
                formal_start + index / offered_rps if offered_rps > 0 else formal_start
            )
            delay = scheduled - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            futures.append(
                pool.submit(
                    _send_tcp_request,
                    peer,
                    path,
                    payload,
                    raw_layer_bytes,
                    block_count,
                    layers,
                    index,
                    scheduled,
                )
            )
        rows = [future.result() for future in futures]

    formal_end = max((float(row["ended_s"]) for row in rows), default=formal_start)
    elapsed = formal_end - formal_start
    application_bytes = requests * layers * payload_bytes
    queue_waits = [float(row["queue_wait_s"]) for row in rows]
    service = [float(row["service_s"]) for row in rows]
    return {
        "ok": True,
        "path": path,
        "raw_layer_bytes": raw_layer_bytes,
        "payload_layer_bytes": payload_bytes,
        "layers": layers,
        "requests": requests,
        "warmup_requests": warmup_requests,
        "concurrency": concurrency,
        "offered_rps": offered_rps,
        "elapsed_s": elapsed,
        "completed_rps": requests / elapsed if elapsed else 0.0,
        "application_bytes": application_bytes,
        "application_gbps": application_bytes * 8 / elapsed / 1e9 if elapsed else 0.0,
        "queue_wait_s": stats(queue_waits),
        "service_s": stats(service),
        "rows": rows,
    }


def _pcie_setup(args, torch):
    from unchain_kv import splitzip_cuda

    raw_bytes, estimated_top16 = payload_sizes(args.tokens)
    blocks = args.tokens // BLOCK_TOKENS
    shape = (2, blocks, BLOCK_TOKENS, KV_HEADS, HEAD_DIM)
    device = torch.device("cuda", args.device)
    source = torch.ones(shape, device=device, dtype=torch.bfloat16)
    block_ids = list(range(blocks))
    ids = None
    cache = None
    if args.layout == "fragmented":
        cache = torch.ones(
            (2, blocks * 2, BLOCK_TOKENS, KV_HEADS, HEAD_DIM),
            device=device,
            dtype=torch.bfloat16,
        )
        ids = torch.arange(0, blocks * 2, 2, device=device)
        block_ids = list(range(0, blocks * 2, 2))

    encoded_bytes = estimated_top16
    encoded_host = None
    if args.path == "writeback":
        probe = torch.empty(raw_bytes, device=device, dtype=torch.uint8)
        encoded_bytes = splitzip_cuda.encode_top16(source, probe, 0) or 0
        torch.cuda.synchronize(args.device)
        if not encoded_bytes or int(probe[0].item()) == 255:
            raise RuntimeError("Top-16 replay payload overflowed or encoder unavailable")
        encoded_host = torch.empty(encoded_bytes, dtype=torch.uint8, pin_memory=True)
        encoded_host.copy_(probe[:encoded_bytes], non_blocking=True)
        torch.cuda.synchronize(args.device)

    slots = []
    for _ in range(args.concurrency):
        slot: dict[str, object] = {
            "stream": torch.cuda.Stream(device=device),
            "done": None,
        }
        if args.layout == "fragmented" and args.direction == "d2h":
            slot["pack"] = torch.empty_like(source)
        if args.direction == "d2h":
            if args.path == "writeback":
                slot["encoded"] = torch.empty(
                    raw_bytes, device=device, dtype=torch.uint8
                )
                slot["host"] = torch.empty(
                    encoded_bytes, dtype=torch.uint8, pin_memory=True
                )
            else:
                slot["host"] = torch.empty(raw_bytes, dtype=torch.uint8, pin_memory=True)
        else:
            if args.path == "writeback":
                slot["encoded"] = torch.empty(
                    encoded_bytes, device=device, dtype=torch.uint8
                )
                target_blocks = blocks * 2 if args.layout == "fragmented" else blocks
                slot["target"] = torch.empty(
                    (2, target_blocks, BLOCK_TOKENS, KV_HEADS, HEAD_DIM),
                    device=device,
                    dtype=torch.bfloat16,
                )
            else:
                slot["gpu_raw"] = torch.empty(
                    raw_bytes, device=device, dtype=torch.uint8
                )
                slot["host"] = torch.empty(raw_bytes, dtype=torch.uint8, pin_memory=True)
        slots.append(slot)
    return {
        "raw_bytes": raw_bytes,
        "encoded_bytes": encoded_bytes,
        "source": source,
        "cache": cache,
        "ids": ids,
        "block_ids": block_ids,
        "encoded_host": encoded_host,
        "slots": slots,
    }


def _submit_pcie(args, state, slot, torch):
    from unchain_kv import splitzip_cuda

    stream = slot["stream"]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        for layer in range(args.layers):
            if args.direction == "d2h":
                source = state["source"]
                if args.layout == "fragmented":
                    torch.index_select(
                        state["cache"], 1, state["ids"], out=slot["pack"]
                    )
                    source = slot["pack"]
                if args.path == "raw":
                    slot["host"].copy_(
                        source.view(torch.uint8).reshape(-1), non_blocking=True
                    )
                else:
                    actual = splitzip_cuda.encode_top16(
                        source, slot["encoded"], layer % args.layers
                    )
                    if actual != state["encoded_bytes"]:
                        raise RuntimeError("Top-16 encoded size changed during replay")
                    slot["host"].copy_(
                        slot["encoded"][:actual], non_blocking=True
                    )
            elif args.path == "raw":
                slot["gpu_raw"].copy_(slot["host"], non_blocking=True)
            else:
                slot["encoded"].copy_(state["encoded_host"], non_blocking=True)
                copied = splitzip_cuda.decode_top16(
                    slot["encoded"],
                    slot["target"],
                    state["block_ids"],
                    state["raw_bytes"],
                    0,
                )
                if copied != state["raw_bytes"]:
                    raise RuntimeError("Top-16 restore failed during replay")
        end.record(stream)
    slot["done"] = end
    return start, end


def replay_pcie(args) -> dict[str, object]:
    if args.library:
        os.environ["UNCHAIN_KV_SPLITZIP_LIB"] = args.library
    import torch

    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    state = _pcie_setup(args, torch)
    slots = state["slots"]
    for index in range(args.warmup_requests):
        slot = slots[index % len(slots)]
        if slot["done"] is not None:
            slot["done"].synchronize()
        _, end = _submit_pcie(args, state, slot, torch)
        end.synchronize()

    records = []
    formal_start = time.perf_counter()
    for index in range(args.requests):
        scheduled = (
            formal_start + index / args.offered_rps
            if args.offered_rps > 0
            else formal_start
        )
        delay = scheduled - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        slot = slots[index % len(slots)]
        if slot["done"] is not None:
            slot["done"].synchronize()
        acquired = time.perf_counter()
        start, end = _submit_pcie(args, state, slot, torch)
        records.append(
            {
                "request": index,
                "scheduled_s": scheduled,
                "acquired_s": acquired,
                "queue_wait_s": max(0.0, acquired - scheduled),
                "start_event": start,
                "end_event": end,
            }
        )
    for slot in slots:
        if slot["done"] is not None:
            slot["done"].synchronize()
    formal_end = time.perf_counter()

    operation_ms = []
    rows = []
    for record in records:
        elapsed_ms = record["start_event"].elapsed_time(record["end_event"])
        operation_ms.append(elapsed_ms)
        rows.append(
            {
                "request": record["request"],
                "queue_wait_s": record["queue_wait_s"],
                "operation_ms": elapsed_ms,
            }
        )
    elapsed = formal_end - formal_start
    layer_bytes = (
        state["raw_bytes"] if args.path == "raw" else state["encoded_bytes"]
    )
    moved = args.requests * args.layers * layer_bytes
    return {
        "ok": True,
        "gpu": torch.cuda.get_device_name(args.device),
        "path": args.path,
        "direction": args.direction,
        "layout": args.layout,
        "tokens": args.tokens,
        "layers": args.layers,
        "raw_layer_bytes": state["raw_bytes"],
        "payload_layer_bytes": layer_bytes,
        "compression_ratio": state["raw_bytes"] / state["encoded_bytes"],
        "requests": args.requests,
        "warmup_requests": args.warmup_requests,
        "concurrency": args.concurrency,
        "offered_rps": args.offered_rps,
        "elapsed_s": elapsed,
        "completed_rps": args.requests / elapsed if elapsed else 0.0,
        "effective_gbytes_s": moved / elapsed / 1e9 if elapsed else 0.0,
        "queue_wait_s": stats([float(row["queue_wait_s"]) for row in rows]),
        "operation_ms": stats(operation_ms),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
        "rows": rows,
    }


def _range(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _tc_dropped(run_root: Path) -> int | None:
    path = run_root / "tc-after.txt"
    if not path.is_file():
        return None
    match = re.search(r"\(dropped (\d+),", path.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else None


def summarize_replays(paths: list[Path]) -> dict[str, object]:
    network = []
    pcie = []
    for requested in paths:
        path = requested / "client.json" if requested.is_dir() else requested
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "direction" in payload:
            pcie.append(
                {
                    "run": path.stem,
                    "path": payload["path"],
                    "direction": payload["direction"],
                    "layout": payload["layout"],
                    "offered_rps": payload["offered_rps"],
                    "completed_rps": payload["completed_rps"],
                    "effective_gbytes_s": payload.get(
                        "effective_gbytes_s", payload.get("effective_gbps", 0.0)
                    ),
                    "queue_wait_p99_s": payload["queue_wait_s"]["p99"],
                    "operation_p50_ms": payload["operation_ms"]["p50"],
                    "ok": payload["ok"],
                }
            )
        elif "service_s" in payload:
            network.append(
                {
                    "run": path.parent.name,
                    "path": payload["path"],
                    "offered_rps": payload["offered_rps"],
                    "completed_rps": payload["completed_rps"],
                    "application_gbps": payload["application_gbps"],
                    "queue_wait_p99_s": payload["queue_wait_s"]["p99"],
                    "service_p50_s": payload["service_s"]["p50"],
                    "tc_dropped": _tc_dropped(path.parent),
                    "ok": payload["ok"],
                }
            )
        else:
            raise ValueError(f"not a replay result: {path}")

    def aggregate(rows, metrics):
        result = {}
        for path_name in sorted({str(row["path"]) for row in rows}):
            selected = [row for row in rows if row["path"] == path_name]
            result[path_name] = {
                metric: _range([float(row[metric]) for row in selected])
                for metric in metrics
            }
        return result

    return {
        "validation": {
            "network_cells": len(network),
            "pcie_cells": len(pcie),
            "all_ok": all(bool(row["ok"]) for row in network + pcie),
            "network_dropped": sum(
                int(row["tc_dropped"] or 0) for row in network
            ),
        },
        "network": network,
        "network_by_path": aggregate(
            network,
            ("completed_rps", "application_gbps", "queue_wait_p99_s"),
        ),
        "pcie": pcie,
        "pcie_by_path": aggregate(
            pcie,
            ("completed_rps", "effective_gbytes_s", "queue_wait_p99_s"),
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    server = subparsers.add_parser("tcp-server")
    server.add_argument("--bind", default="0.0.0.0:29620")
    server.add_argument("--expected-frames", type=int, required=True)
    server.add_argument("--timeout-s", type=float, default=900)
    server.add_argument("--output")

    client = subparsers.add_parser("tcp-client")
    client.add_argument("--peer", required=True)
    client.add_argument("--path", choices=("raw", "writeback"), required=True)
    client.add_argument("--raw-layer-bytes", type=int, default=RAW_LAYER_BYTES_32K)
    client.add_argument(
        "--top16-layer-bytes", type=int, default=TOP16_LAYER_BYTES_32K
    )
    client.add_argument("--block-count", type=int, default=2000)
    client.add_argument("--layers", type=int, default=LAYERS)
    client.add_argument("--requests", type=int, default=16)
    client.add_argument("--warmup-requests", type=int, default=1)
    client.add_argument("--concurrency", type=int, default=4)
    client.add_argument("--offered-rps", type=float, default=0.75)
    client.add_argument("--output")

    pcie = subparsers.add_parser("pcie")
    pcie.add_argument("--path", choices=("raw", "writeback"), required=True)
    pcie.add_argument("--direction", choices=("d2h", "h2d"), required=True)
    pcie.add_argument(
        "--layout", choices=("contiguous", "fragmented"), default="fragmented"
    )
    pcie.add_argument("--tokens", type=int, default=32000)
    pcie.add_argument("--layers", type=int, default=LAYERS)
    pcie.add_argument("--requests", type=int, default=32)
    pcie.add_argument("--warmup-requests", type=int, default=2)
    pcie.add_argument("--concurrency", type=int, default=2)
    pcie.add_argument("--offered-rps", type=float, default=10.5)
    pcie.add_argument("--device", type=int, default=1)
    pcie.add_argument("--library")
    pcie.add_argument("--output")

    summary = subparsers.add_parser("summarize")
    summary.add_argument("paths", nargs="+", type=Path)
    summary.add_argument("--output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "tcp-server":
        result = serve_tcp(
            parse_address(args.bind), args.expected_frames, args.timeout_s
        )
    elif args.command == "tcp-client":
        result = replay_tcp(
            parse_address(args.peer),
            args.path,
            args.raw_layer_bytes,
            args.top16_layer_bytes,
            args.block_count,
            args.layers,
            args.requests,
            args.warmup_requests,
            args.concurrency,
            args.offered_rps,
        )
    elif args.command == "pcie":
        result = replay_pcie(args)
    else:
        result = summarize_replays(args.paths)
    emit(result, args.output)
    ok = result.get("ok", result.get("validation", {}).get("all_ok"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
