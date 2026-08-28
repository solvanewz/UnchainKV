#!/usr/bin/env python3
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import urllib.request
import uuid


def prefill_payload(payload: dict) -> dict:
    copied = dict(payload)
    copied["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    copied["stream"] = False
    copied["max_tokens"] = 1
    copied.pop("max_completion_tokens", None)
    copied.pop("min_tokens", None)
    copied.pop("min_completion_tokens", None)
    copied.pop("stream_options", None)
    return copied


def post_prefill(url: str, payload: dict, request_id: str, timeout_s: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(prefill_payload(payload)).encode(),
        headers={"Content-Type": "application/json", "X-Request-Id": request_id},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        result = json.loads(response.read().decode())
    params = result.get("kv_transfer_params")
    if not params:
        raise RuntimeError("NIXL prefiller returned no kv_transfer_params")
    return params


class Handler(BaseHTTPRequestHandler):
    prefill_url = ""
    decode_url = ""
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
        request_id = uuid.uuid4().hex
        try:
            payload["kv_transfer_params"] = post_prefill(
                self.prefill_url,
                payload,
                request_id,
                self.timeout_s,
            )
            request = urllib.request.Request(
                self.decode_url,
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "X-Request-Id": request_id,
                },
            )
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() not in {
                        "content-length",
                        "connection",
                        "transfer-encoding",
                    }:
                        self.send_header(key, value)
                self.end_headers()
                while chunk := response.readline():
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except Exception as exc:
            self.send_error(502, str(exc))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="127.0.0.1:18082")
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    args = parser.parse_args()

    host, port = args.listen.rsplit(":", 1)
    Handler.prefill_url = args.prefill_url.rstrip("/") + "/v1/completions"
    Handler.decode_url = args.decode_url.rstrip("/") + "/v1/completions"
    Handler.timeout_s = args.timeout_s
    ThreadingHTTPServer((host, int(port)), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
