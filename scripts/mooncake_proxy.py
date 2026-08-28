#!/usr/bin/env python3
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import queue
import threading
import urllib.request
import uuid


def query_engine_id(url: str, timeout_s: float) -> str:
    with urllib.request.urlopen(url.rstrip("/") + "/query", timeout=timeout_s) as response:
        data = json.loads(response.read().decode())
    if not data:
        raise RuntimeError("empty Mooncake bootstrap query")
    dp0 = data.get("0") or data.get(0)
    if not dp0:
        raise RuntimeError(f"missing dp rank 0 in bootstrap query: {data}")
    if isinstance(dp0, dict) and dp0.get("engine_id"):
        return str(dp0["engine_id"])
    return next(iter(dp0))


def prefill_payload(payload: dict, transfer_id: str) -> dict:
    copied = dict(payload)
    copied["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": transfer_id,
    }
    copied["stream"] = False
    copied["max_tokens"] = 1
    copied.pop("stream_options", None)
    if "max_completion_tokens" in copied:
        copied["max_completion_tokens"] = 1
    return copied


def decode_payload(payload: dict, transfer_id: str, bootstrap_url: str, engine_id: str) -> dict:
    copied = dict(payload)
    copied["kv_transfer_params"] = {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_bootstrap_addr": bootstrap_url,
        "remote_engine_id": engine_id,
        "transfer_id": transfer_id,
    }
    return copied


def post_json(url: str, payload: dict, timeout_s: float, request_id: str, out: queue.Queue) -> None:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
            "X-data-parallel-rank": "0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            response.read()
    except BaseException as exc:
        out.put(exc)


class Handler(BaseHTTPRequestHandler):
    prefill_url = ""
    decode_url = ""
    prefill_bootstrap_url = ""
    remote_engine_id = ""
    timeout_s = 900.0

    def do_GET(self) -> None:
        if self.path != "/healthcheck":
            self.send_error(404)
            return
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode())
        transfer_id = uuid.uuid4().hex
        request_id = uuid.uuid4().hex
        errors: queue.Queue = queue.Queue()

        prefill_thread = threading.Thread(
            target=post_json,
            args=(
                self.prefill_url,
                prefill_payload(payload, transfer_id),
                self.timeout_s,
                request_id,
                errors,
            ),
            daemon=True,
        )
        prefill_thread.start()

        req = urllib.request.Request(
            self.decode_url,
            data=json.dumps(
                decode_payload(
                    payload,
                    transfer_id,
                    self.prefill_bootstrap_url,
                    self.remote_engine_id,
                )
            ).encode(),
            headers={"Content-Type": "application/json", "X-Request-Id": request_id},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
            self.send_response(response.status)
            for key, value in response.headers.items():
                if key.lower() not in {"content-length", "connection", "transfer-encoding"}:
                    self.send_header(key, value)
            self.end_headers()
            while chunk := response.readline():
                self.wfile.write(chunk)
                self.wfile.flush()
        prefill_thread.join()
        if not errors.empty():
            raise errors.get()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="127.0.0.1:18082")
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--prefill-bootstrap-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    args = parser.parse_args()

    host, port = args.listen.rsplit(":", 1)
    Handler.prefill_url = args.prefill_url.rstrip("/") + "/v1/completions"
    Handler.decode_url = args.decode_url.rstrip("/") + "/v1/completions"
    Handler.prefill_bootstrap_url = args.prefill_bootstrap_url.rstrip("/")
    Handler.remote_engine_id = query_engine_id(Handler.prefill_bootstrap_url, args.timeout_s)
    Handler.timeout_s = args.timeout_s
    ThreadingHTTPServer((host, int(port)), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
