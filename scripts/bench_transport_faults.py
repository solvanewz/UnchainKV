#!/usr/bin/env python3
"""Bounded transport, restart, request-failure, and recovery campaigns."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import multiprocessing
from pathlib import Path
import queue
import socket
import threading
import time
from zlib import crc32

from unchain_kv.layer_state import LayerStore
from unchain_kv.protocol import Chunk, ChunkHeader
from unchain_kv.tcp_data import TcpReceiver, _FRAME_HEADER, send_chunks
from unchain_kv.vllm_connector import Session, _RequestSpool
from scripts.bench_ownership_faults import _connector, _Trace


def chunk(transfer: str, layer: int, index: int = 0, total: int = 1) -> Chunk:
    payload = f"{transfer}:{layer}:{index}".encode()
    return Chunk(
        ChunkHeader(
            transfer,
            f"request-{transfer}",
            layer,
            0,
            index,
            total,
            0,
            len(payload),
            crc32(payload) & 0xFFFFFFFF,
        ),
        payload,
    )


def frame(payload: bytes) -> bytes:
    return _FRAME_HEADER.pack(len(payload), crc32(payload) & 0xFFFFFFFF) + payload


def wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def receiver(store: LayerStore, trace: _Trace | None = None) -> tuple[TcpReceiver, threading.Thread]:
    service = TcpReceiver(("127.0.0.1", 0), store, trace=trace)
    thread = threading.Thread(target=service.serve, daemon=True)
    thread.start()
    return service, thread


def receiver_process(control) -> None:
    store = LayerStore()
    service, thread = receiver(store)
    control.send(service.port)
    try:
        while True:
            command = control.recv()
            if command[0] == "state":
                transfer = command[1]
                control.send(
                    {
                        "payload_bytes": store.payload_bytes(transfer),
                        "ready": store.is_ready(transfer, 0),
                        "session": store.observed_session(),
                    }
                )
            else:
                return
    finally:
        service.close()
        thread.join(timeout=1)


def spool_process(control, transfer: str | None) -> None:
    spools = {}
    reserved = 0
    if transfer is not None:
        spools[transfer] = _RequestSpool(transfer, f"request-{transfer}", 64)
        reserved = 64
    control.send({"transfers": sorted(spools), "reserved": reserved, "resident": 0})
    control.recv()


def run_s2(iterations: int) -> dict[str, object]:
    scenarios = ("header-before", "payload-mid", "layer-between", "spool-admission-after")
    rows = []
    for scenario in scenarios:
        passed = 0
        errors = Counter()
        for cycle in range(iterations):
            transfer = f"{scenario}-{cycle}"
            store, trace = LayerStore(), _Trace()
            service, thread = receiver(store, trace)
            connector = None
            spool = None
            try:
                payload = chunk(transfer, 0, 0, 2).pack()
                wire = frame(payload)
                if scenario == "layer-between":
                    send_chunks(("127.0.0.1", service.port), [chunk(transfer, 0, 0, 2)])
                    wait_for(lambda: store.payload_bytes(transfer) > 0)
                else:
                    with socket.create_connection(("127.0.0.1", service.port)) as sock:
                        if scenario == "header-before":
                            sock.sendall(wire[:4])
                        else:
                            sock.sendall(wire[: len(_FRAME_HEADER.pack(0, 0)) + len(payload) // 2])
                if scenario == "spool-admission-after":
                    connector = _connector(_Trace())
                    spool = _RequestSpool(transfer, f"request-{cycle}", 64)
                    connector._request_spools[transfer] = spool
                    connector._spool_reserved_bytes = 64
                explicit = wait_for(lambda: any(row["event"] == "tcp_receive_error" for row in trace.rows))
                if scenario == "layer-between":
                    try:
                        store.wait(transfer, 0, 0.05)
                    except TimeoutError:
                        explicit = True
                if connector is not None and spool is not None:
                    connector._release_request_spool(spool)
                resources_zero = store.payload_bytes() == 0 and (
                    connector is None
                    or (not connector._request_spools and connector._spool_reserved_bytes == 0 and connector._spool_resident_bytes == 0)
                )
                if explicit and not store.is_ready(transfer, 0) and resources_zero:
                    passed += 1
                else:
                    errors["cycle-gate"] += 1
            finally:
                service.close()
                thread.join(timeout=1)
        rows.append({"scenario": scenario, "cycles": iterations, "passed_cycles": passed, "errors": dict(errors)})
    return {"experiment": "S2", "passed": all(row["passed_cycles"] == iterations and not row["errors"] for row in rows), "iterations_per_scenario": iterations, "scenarios": rows}


def run_s3(iterations: int) -> dict[str, object]:
    rows = []
    for scenario in ("consumer-receive-restart", "producer-spool-restart"):
        passed = 0
        errors = Counter()
        for cycle in range(iterations):
            old = f"old-{scenario}-{cycle}"
            new = f"new-{scenario}-{cycle}"
            context = multiprocessing.get_context("fork")
            if scenario == "consumer-receive-restart":
                parent1, child1 = context.Pipe()
                process1 = context.Process(target=receiver_process, args=(child1,))
                process1.start()
                port1 = parent1.recv()
                send_chunks(("127.0.0.1", port1), [chunk(old, 0, 0, 2)])
                parent1.send(("state", old))
                old_state = parent1.recv()
                process1.terminate()
                process1.join(timeout=2)

                parent2, child2 = context.Pipe()
                process2 = context.Process(target=receiver_process, args=(child2,))
                process2.start()
                port2 = parent2.recv()
                send_chunks(("127.0.0.1", port2), [chunk(new, 0)])
                time.sleep(0.02)
                parent2.send(("state", new))
                new_state = parent2.recv()
                parent2.send(("stop",))
                process2.join(timeout=2)
                good = (
                    old_state["payload_bytes"] > 0
                    and process1.exitcode not in (None, 0)
                    and new_state["ready"]
                    and new_state["session"] == (new, f"request-{new}")
                    and process2.exitcode == 0
                )
            else:
                parent1, child1 = context.Pipe()
                process1 = context.Process(target=spool_process, args=(child1, old))
                process1.start()
                old_state = parent1.recv()
                process1.terminate()
                process1.join(timeout=2)
                parent2, child2 = context.Pipe()
                process2 = context.Process(target=spool_process, args=(child2, None))
                process2.start()
                new_state = parent2.recv()
                parent2.send(("stop",))
                process2.join(timeout=2)
                good = (
                    old_state == {"transfers": [old], "reserved": 64, "resident": 0}
                    and process1.exitcode not in (None, 0)
                    and new_state == {"transfers": [], "reserved": 0, "resident": 0}
                    and process2.exitcode == 0
                )
            if good:
                passed += 1
            else:
                errors["restart-gate"] += 1
        rows.append({"scenario": scenario, "cycles": iterations, "passed_cycles": passed, "errors": dict(errors)})
    return {"experiment": "S3", "passed": all(row["passed_cycles"] == iterations and not row["errors"] for row in rows), "iterations_per_scenario": iterations, "scenarios": rows}


def run_s4(iterations: int) -> dict[str, object]:
    rows = []
    for scenario in ("client-cancel", "timeout", "batch-single-failure", "spool-rejection"):
        passed = 0
        errors = Counter()
        for cycle in range(iterations):
            store = LayerStore()
            left, right, victim = f"left-{cycle}", f"right-{cycle}", f"victim-{cycle}"
            store.add(chunk(left, 0))
            store.add(chunk(right, 0))
            store.add(chunk(victim, 0, 0, 2))
            explicit = False
            if scenario == "spool-rejection":
                connector = _connector(_Trace())
                connector._spool_layer_block_bytes = [64]
                connector.expected_layers = 1
                connector.max_blocks = 0
                connector.request_spool_bytes = 64
                try:
                    connector._admit_request_spools(
                        [Session("a", "a", [1]), Session("b", "b", [2])]
                    )
                except RuntimeError as error:
                    explicit = "batch exceeds cap" in str(error)
                clean = not connector._request_spools and connector._spool_reserved_bytes == 0
            else:
                error = TimeoutError("request timeout") if scenario == "timeout" else RuntimeError(scenario)
                store.fail(error, victim)
                explicit = True
                clean = store.payload_bytes(victim) == 0
            siblings = store.is_ready(left, 0) and store.is_ready(right, 0)
            store.discard_transfer(left)
            store.discard_transfer(right)
            store.discard_transfer(victim)
            if explicit and clean and siblings and store.payload_bytes() == 0:
                passed += 1
            else:
                errors["isolation-gate"] += 1
        rows.append({"scenario": scenario, "cycles": iterations, "passed_cycles": passed, "errors": dict(errors)})
    return {"experiment": "S4", "passed": all(row["passed_cycles"] == iterations and not row["errors"] for row in rows), "iterations_per_scenario": iterations, "scenarios": rows}


def run_s5(iterations: int) -> dict[str, object]:
    rows = []
    for scenario in ("sender-pause-resume", "10G-2G-10G"):
        passed = 0
        errors = Counter()
        recoveries = []
        for cycle in range(iterations):
            transfer = f"{scenario}-{cycle}"
            store = LayerStore()
            service, receiver_thread = receiver(store)
            frames: queue.Queue[bytes | None] = queue.Queue(maxsize=4)
            frame_bytes = [frame(chunk(transfer, layer).pack()) for layer in range(12)]
            hard_cap = max(map(len, frame_bytes)) * 4
            max_queued = 0
            started = time.monotonic()

            def produce() -> None:
                nonlocal max_queued
                for wire in frame_bytes:
                    frames.put(wire)
                    max_queued = max(max_queued, frames.qsize() * max(map(len, frame_bytes)))
                frames.put(None)

            def send() -> None:
                with socket.create_connection(("127.0.0.1", service.port)) as sock:
                    index = 0
                    while True:
                        wire = frames.get()
                        if wire is None:
                            return
                        if scenario == "sender-pause-resume" and index == 4:
                            time.sleep(0.05)
                        elif scenario == "10G-2G-10G" and 4 <= index < 8:
                            time.sleep(0.01)
                        sock.sendall(wire)
                        index += 1

            producer = threading.Thread(target=produce)
            sender = threading.Thread(target=send)
            producer.start()
            sender.start()
            producer.join(timeout=5)
            sender.join(timeout=5)
            ready = wait_for(lambda: all(store.is_ready(transfer, layer) for layer in range(12)))
            recovery = time.monotonic() - started
            recoveries.append(recovery)
            fifo = ready and all(bytes(store.layer_payloads(transfer, layer)[0]) == chunk(transfer, layer).payload for layer in range(12))
            released = store.discard_transfer(transfer)
            service.close()
            receiver_thread.join(timeout=1)
            if fifo and max_queued <= hard_cap and released > 0 and store.payload_bytes() == 0 and frames.empty():
                passed += 1
            else:
                errors["recovery-gate"] += 1
        rows.append({"scenario": scenario, "cycles": iterations, "passed_cycles": passed, "errors": dict(errors), "recovery_s": recoveries, "max_recovery_s": max(recoveries, default=0)})
    return {"experiment": "S5", "passed": all(row["passed_cycles"] == iterations and not row["errors"] for row in rows), "iterations_per_scenario": iterations, "scenarios": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=("S2", "S3", "S4", "S5"))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"S2": run_s2, "S3": run_s3, "S4": run_s4, "S5": run_s5}[args.experiment](args.iterations)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(not result["passed"])


if __name__ == "__main__":
    main()
