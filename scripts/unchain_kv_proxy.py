#!/usr/bin/env python3
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import threading
import time
import urllib.request
import uuid


def with_transfer(payload: dict, transfer_id: str, *, producer: bool) -> dict:
    copied = dict(payload)
    params = dict(copied.get("kv_transfer_params") or {})
    params["transfer_id"] = transfer_id
    if producer:
        params["do_remote_decode"] = True
        params.pop("do_remote_prefill", None)
        copied["stream"] = False
        copied["max_tokens"] = 1
        if "max_completion_tokens" in copied:
            copied["max_completion_tokens"] = 1
        copied.pop("stream_options", None)
    else:
        params["do_remote_prefill"] = True
        params.pop("do_remote_decode", None)
    copied["kv_transfer_params"] = params
    return copied


def post_json(url: str, payload: dict, timeout_s: float, out: queue.Queue) -> None:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            response.read()
    except BaseException as exc:
        out.put(("prefill_error", str(exc)))


def stream_decode(url: str, payload: dict, timeout_s: float, out: queue.Queue) -> None:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            out.put(("headers", response.status, list(response.headers.items())))
            while True:
                chunk = response.readline()
                if not chunk:
                    break
                out.put(("data", chunk))
    except BaseException as exc:
        out.put(("decode_error", str(exc)))
    finally:
        out.put(("end", None))


class Handler(BaseHTTPRequestHandler):
    prefill_url = ""
    decode_url = ""
    decode_lead_s = 1.0
    timeout_s = 300.0
    decode_capacity = 0
    decode_slots: threading.BoundedSemaphore | None = None
    decode_inflight = 0
    decode_waiting = 0
    metrics_path: Path | None = None
    admission_lock = threading.Lock()
    metrics_lock = threading.Lock()

    @classmethod
    def configure_admission(cls, capacity: int, metrics_path: Path | None) -> None:
        cls.decode_capacity = max(0, capacity)
        cls.decode_slots = (
            threading.BoundedSemaphore(cls.decode_capacity)
            if cls.decode_capacity
            else None
        )
        cls.decode_inflight = 0
        cls.decode_waiting = 0
        cls.metrics_path = metrics_path
        if metrics_path is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text("", encoding="utf-8")

    @classmethod
    def _snapshot(cls) -> dict[str, int | None]:
        return {
            "prefill_queue": cls.decode_waiting,
            "decode_queue": cls.decode_waiting,
            "decode_inflight": cls.decode_inflight,
            "decode_available": (
                cls.decode_capacity - cls.decode_inflight
                if cls.decode_capacity
                else None
            ),
        }

    @classmethod
    def _metric(cls, event: str, request: str, **fields: object) -> None:
        if cls.metrics_path is None:
            return
        row = {
            "event": event,
            "request": request,
            "t": time.perf_counter(),
            **fields,
        }
        with cls.metrics_lock:
            with cls.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    @classmethod
    def acquire_decode_slot(cls, request: str) -> dict[str, int | None]:
        if cls.decode_slots is None:
            return cls._snapshot()
        start = time.perf_counter()
        with cls.admission_lock:
            cls.decode_waiting += 1
            queued = cls._snapshot()
        cls._metric("admission_queued", request, **queued)
        cls.decode_slots.acquire()
        with cls.admission_lock:
            cls.decode_waiting -= 1
            cls.decode_inflight += 1
            acquired = cls._snapshot()
        cls._metric(
            "admission_acquired",
            request,
            wait_s=time.perf_counter() - start,
            **acquired,
        )
        return acquired

    @classmethod
    def release_decode_slot(cls, request: str) -> None:
        if cls.decode_slots is None:
            return
        with cls.admission_lock:
            cls.decode_inflight -= 1
            released = cls._snapshot()
        cls.decode_slots.release()
        cls._metric("admission_released", request, **released)

    def do_POST(self) -> None:
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        transfer_id = uuid.uuid4().hex
        events: queue.Queue = queue.Queue()
        self.acquire_decode_slot(transfer_id)
        decode_thread = None
        try:
            decode_thread = threading.Thread(
                target=stream_decode,
                args=(
                    self.decode_url,
                    with_transfer(payload, transfer_id, producer=False),
                    self.timeout_s,
                    events,
                ),
                daemon=True,
            )
            decode_thread.start()
            time.sleep(self.decode_lead_s)
            threading.Thread(
                target=post_json,
                args=(
                    self.prefill_url,
                    with_transfer(payload, transfer_id, producer=True),
                    self.timeout_s,
                    events,
                ),
                daemon=True,
            ).start()

            headers_sent = False
            while True:
                item = events.get()
                kind = item[0]
                if kind == "headers":
                    _, status, headers = item
                    self.send_response(status)
                    for key, value in headers:
                        if key.lower() not in {
                            "content-length",
                            "connection",
                            "transfer-encoding",
                        }:
                            self.send_header(key, value)
                    self.end_headers()
                    headers_sent = True
                elif kind == "data" and headers_sent:
                    self.wfile.write(item[1])
                    self.wfile.flush()
                elif kind == "prefill_error":
                    if not headers_sent:
                        self.send_error(502, item[1])
                    return
                elif kind == "decode_error":
                    if not headers_sent:
                        self.send_error(502, item[1])
                    return
                elif kind == "end":
                    return
        finally:
            if decode_thread is not None and decode_thread.ident is not None:
                decode_thread.join()
            self.release_decode_slot(transfer_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="127.0.0.1:18082")
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--decode-lead-s", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--decode-slots", type=int, default=0)
    parser.add_argument("--metrics")
    args = parser.parse_args()

    host, port = args.listen.rsplit(":", 1)
    Handler.prefill_url = args.prefill_url.rstrip("/") + "/v1/completions"
    Handler.decode_url = args.decode_url.rstrip("/") + "/v1/completions"
    Handler.decode_lead_s = args.decode_lead_s
    Handler.timeout_s = args.timeout_s
    Handler.configure_admission(
        args.decode_slots,
        Path(args.metrics) if args.metrics else None,
    )
    ThreadingHTTPServer((host, int(port)), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
