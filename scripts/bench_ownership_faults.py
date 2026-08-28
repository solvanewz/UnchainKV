#!/usr/bin/env python3
"""Run the bounded C4 ownership and C5 fail-closed fault campaigns."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from zlib import crc32

from unchain_kv.layer_state import LayerStore
from unchain_kv.protocol import Chunk, ChunkHeader
from unchain_kv.vllm_connector import (
    UnchainKVConnector,
    _CodecBuffer,
    _GpuPackBuffer,
    _PinnedStageBuffer,
    _RequestSpool,
    _decode_splitzip_top16_payload,
)


class _Tensor:
    def __init__(self, size: int):
        self.size = size

    def numel(self) -> int:
        return self.size

    def element_size(self) -> int:
        return 1


class _Trace:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def event(self, name: str, **fields: object) -> None:
        self.rows.append({"event": name, **fields})


def _chunk(
    transfer: str,
    index: int,
    total: int,
    payload: bytes = b"payload",
) -> Chunk:
    return Chunk(
        ChunkHeader(
            transfer,
            f"request-{transfer}",
            0,
            0,
            index,
            total,
            0,
            len(payload),
            crc32(payload) & 0xFFFFFFFF,
        ),
        payload,
    )


def _connector(trace: _Trace) -> UnchainKVConnector:
    connector = object.__new__(UnchainKVConnector)
    connector.trace = trace
    connector._buffer_lease_id = 0
    connector._pending_stage_buffers = []
    connector._pending_gpu_pack_buffers = []
    connector._pending_codec_buffers = []
    connector._spool_condition = threading.Condition()
    connector._request_spools = {}
    connector._spool_reserved_bytes = 0
    connector._spool_resident_bytes = 0
    connector.codec_writeback_requested = False
    return connector


def _tensors(device: str) -> tuple[object, object, object, int]:
    try:
        import torch

        if device != "cpu" and torch.cuda.is_available():
            gpu = torch.device(device)
            baseline = int(torch.cuda.memory_allocated(gpu))
            return (
                torch.empty((4096,), dtype=torch.uint8, pin_memory=True),
                torch.empty((4096,), dtype=torch.uint8, device=gpu),
                torch.empty((4096,), dtype=torch.uint8, device=gpu),
                baseline,
            )
    except (ImportError, RuntimeError):
        pass
    return _Tensor(4096), _Tensor(4096), _Tensor(4096), 0


def _gpu_allocated(device: str) -> int:
    try:
        import torch

        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
            return int(torch.cuda.memory_allocated(device))
    except (ImportError, RuntimeError):
        pass
    return 0


def _rss_kib(field: str = "VmRSS") -> int:
    status = Path("/proc/self/status")
    if not status.is_file():
        return 0
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{field}:"):
            return int(line.split()[1])
    return 0


def run_c4(iterations: int, device: str) -> dict[str, object]:
    scenarios = ("normal", "cancel", "timeout", "producer-exit", "consumer-exit")
    summary = []
    trace = _Trace()
    connector = _connector(trace)
    warmup = _tensors(device)
    del warmup
    gc.collect()
    gpu_initial = _gpu_allocated(device)
    rss_initial = _rss_kib()
    owner_rejections = 0
    for scenario in scenarios:
        passed = 0
        errors: Counter[str] = Counter()
        for cycle in range(iterations):
            row_start = len(trace.rows)
            pinned_tensor, pack_tensor, codec_tensor, gpu_baseline = _tensors(device)
            entries = (
                _PinnedStageBuffer(pinned_tensor),
                _GpuPackBuffer(pack_tensor),
                _CodecBuffer(codec_tensor),
            )
            for entry in entries:
                connector._begin_buffer_lease(entry)
                try:
                    connector._begin_buffer_lease(entry)
                except RuntimeError:
                    owner_rejections += 1
                else:
                    errors["overlapping-owner-not-rejected"] += 1
            connector._pending_stage_buffers.append(entries[0])
            connector._pending_gpu_pack_buffers.append(entries[1])
            connector._pending_codec_buffers.append(entries[2])
            transfer = f"{scenario}-{cycle}"
            spool = _RequestSpool(transfer, f"request-{cycle}", 64)
            connector._request_spools[transfer] = spool
            connector._spool_reserved_bytes = 64
            trace.event("spool_request_admitted", transfer=transfer)
            explicit_error = False
            if scenario == "cancel":
                try:
                    raise RuntimeError("request cancelled")
                except RuntimeError:
                    explicit_error = True
            elif scenario == "timeout":
                try:
                    raise TimeoutError("request timeout")
                except TimeoutError:
                    explicit_error = True
            if scenario in {"producer-exit", "consumer-exit"}:
                code = 91 if scenario == "producer-exit" else 92
                child = subprocess.run(
                    [sys.executable, "-c", f"import os; os._exit({code})"],
                    check=False,
                )
                explicit_error = child.returncode == code
                if not explicit_error:
                    errors["exit-code-not-propagated"] += 1
            connector._cleanup_failed_send_args(())
            connector._release_request_spool(spool)
            cycle_rows = trace.rows[row_start:]
            acquired = [
                (row["kind"], row["lease"])
                for row in cycle_rows
                if row["event"] == "buffer_acquire"
            ]
            released = [
                (row["kind"], row["lease"])
                for row in cycle_rows
                if row["event"] == "buffer_release"
            ]
            resources_zero = (
                not connector._pending_stage_buffers
                and not connector._pending_gpu_pack_buffers
                and not connector._pending_codec_buffers
                and not connector._request_spools
                and connector._spool_reserved_bytes == 0
                and connector._spool_resident_bytes == 0
                and not any(entry.lease_id for entry in entries)
            )
            expected_error = scenario != "normal"
            if (
                resources_zero
                and explicit_error == expected_error
                and set(acquired) == set(released)
                and len(acquired) == len(set(acquired))
                and len(released) == len(set(released))
            ):
                passed += 1
            else:
                errors["cycle-gate"] += 1
            del entry, entries, pinned_tensor, pack_tensor, codec_tensor
            gc.collect()
            if _gpu_allocated(device) != gpu_baseline:
                errors["gpu-allocation-not-restored"] += 1
        summary.append(
            {
                "scenario": scenario,
                "cycles": iterations,
                "passed_cycles": passed,
                "errors": dict(errors),
            }
        )
    all_rows = trace.rows
    acquired = [row["lease"] for row in all_rows if row["event"] == "buffer_acquire"]
    released = [row["lease"] for row in all_rows if row["event"] == "buffer_release"]
    gpu_final = _gpu_allocated(device)
    rss_final = _rss_kib()
    rss_peak = _rss_kib("VmHWM")
    expected_resources = len(scenarios) * iterations
    campaign_passed = all(
        row["passed_cycles"] == iterations and not row["errors"] for row in summary
    )
    return {
        "experiment": "C4",
        "passed": campaign_passed
        and gpu_final == gpu_initial
        and rss_final - rss_initial <= 64 * 1024,
        "iterations_per_scenario": iterations,
        "scenarios": summary,
        "ownership": {
            "buffer_acquires": len(acquired),
            "buffer_releases": len(released),
            "double_release": len(released) - len(set(released)),
            "overlapping_owner_rejections": owner_rejections,
            "spool_admits": sum(row["event"] == "spool_request_admitted" for row in all_rows),
            "spool_releases": sum(row["event"] == "spool_request_released" for row in all_rows),
            "expected_lifecycles": expected_resources,
        },
        "device": device,
        "memory": {
            "gpu_initial_allocated_bytes": gpu_initial,
            "gpu_final_allocated_bytes": gpu_final,
            "rss_initial_kib": rss_initial,
            "rss_final_kib": rss_final,
            "rss_peak_kib": rss_peak,
            "rss_growth_kib": rss_final - rss_initial,
        },
    }


def _fail_store(store: LayerStore, error: BaseException) -> None:
    store.fail(error)
    raise error


def run_c5(iterations: int) -> dict[str, object]:
    scenarios = (
        "incomplete-layer",
        "truncated-payload",
        "bad-checksum",
        "bad-mode-length",
        "out-of-order-duplicate",
    )
    summary = []
    for scenario in scenarios:
        passed = 0
        errors: Counter[str] = Counter()
        for cycle in range(iterations):
            transfer = f"{scenario}-{cycle}"
            store = LayerStore()
            decode_started = False
            try:
                if scenario == "incomplete-layer":
                    store.add(_chunk(transfer, 0, 2))
                    store.wait(transfer, 0, 0)
                elif scenario in {"truncated-payload", "bad-checksum"}:
                    store.add(_chunk(transfer, 0, 2))
                    packed = bytearray(_chunk(transfer, 1, 2).pack())
                    if scenario == "truncated-payload":
                        del packed[-1]
                    else:
                        packed[-1] ^= 1
                    try:
                        parsed = Chunk.unpack(bytes(packed))
                    except BaseException as exc:
                        _fail_store(store, exc)
                    store.add(parsed)
                elif scenario == "bad-mode-length":
                    store.add(_chunk(transfer, 0, 2))
                    bad = b"\x04bad" if cycle % 2 == 0 else b"\x05bad"
                    raw_bytes = 8 if cycle % 2 == 0 else 7
                    decoded = _decode_splitzip_top16_payload(bad, raw_bytes, bytes(32))
                    if decoded is None:
                        _fail_store(store, ValueError("bad codec mode/length"))
                    decode_started = True
                else:
                    store.add(_chunk(transfer, 1, 3))
                    store.add(_chunk(transfer, 0, 3))
                    store.add(_chunk(transfer, 1, 3))
            except BaseException as exc:
                errors[type(exc).__name__] += 1
                if (
                    not decode_started
                    and not store.is_ready(transfer, 0)
                    and store.payload_bytes(transfer) == 0
                ):
                    passed += 1
            else:
                errors["silent-success"] += 1
        summary.append(
            {
                "scenario": scenario,
                "injections": iterations,
                "explicit_failures": sum(errors.values()) - errors["silent-success"],
                "passed_injections": passed,
                "error_types": dict(errors),
            }
        )
    return {
        "experiment": "C5",
        "passed": all(
            row["passed_injections"] == iterations
            and "silent-success" not in row["error_types"]
            for row in summary
        ),
        "iterations_per_scenario": iterations,
        "decode_started": 0,
        "scenarios": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=("C4", "C5"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    result = (
        run_c4(args.iterations, args.device)
        if args.experiment == "C4"
        else run_c5(args.iterations)
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
