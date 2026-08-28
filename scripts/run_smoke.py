#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import urllib.request


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--transfer-id", required=True)
    parser.add_argument("--model", default=os.environ.get("UNCHAIN_KV_MODEL", "default"))
    parser.add_argument("--prompt", default="Explain paged attention in one sentence.")
    parser.add_argument("--decode-lead-s", type=float, default=1.0)
    args = parser.parse_args()

    base = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": 1,
        "temperature": 0,
    }
    prefill = dict(base)
    prefill["kv_transfer_params"] = {
        "do_remote_decode": True,
        "transfer_id": args.transfer_id,
    }
    decode = dict(base)
    decode["kv_transfer_params"] = {
        "do_remote_prefill": True,
        "transfer_id": args.transfer_id,
    }

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as pool:
        decode_future = pool.submit(
            post_json, args.decode_url.rstrip("/") + "/v1/completions", decode
        )
        time.sleep(args.decode_lead_s)
        prefill_response = post_json(
            args.prefill_url.rstrip("/") + "/v1/completions", prefill
        )
        decode_response = decode_future.result(timeout=180)
    print(
        json.dumps(
            {
                "elapsed_s": time.perf_counter() - start,
                "prefill_has_choices": bool(prefill_response.get("choices")),
                "decode_has_choices": bool(decode_response.get("choices")),
            },
            indent=2,
        )
    )
    return 0 if decode_response.get("choices") else 1


if __name__ == "__main__":
    raise SystemExit(main())
