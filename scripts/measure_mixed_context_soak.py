#!/usr/bin/env python3
"""Deterministic mixed-context soak driver for extent experiments.

Drives HTTP streaming completions at 8k/16k/32k prompt lengths, collects
per-request JSONL and summary JSON.  Uses stdlib only.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_prompt(prompt_dir: Path, length: int, index: int) -> str:
    path = prompt_dir / f"prompt-{length}-{index}.txt"
    if not path.is_file():
        path = prompt_dir / f"prompt-{length}-{index:04d}.txt"
    return path.read_text()


def send_request(url: str, model: str, prompt: str, max_tokens: int, timeout: int) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first_token = None
    tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                body = line[5:].strip()
                if body == b"[DONE]":
                    break
                event = json.loads(body.decode())
                choices = event.get("choices") or []
                if choices:
                    c = choices[0]
                    text = c.get("text") or (c.get("delta") or {}).get("content") or ""
                    if text and first_token is None:
                        first_token = time.perf_counter()
                    tokens += 1
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}
    end = time.perf_counter()
    return {
        "ok": True,
        "ttft_s": (first_token - start) if first_token else None,
        "post_ttft_s": (end - first_token) if first_token else None,
        "e2e_s": end - start,
        "tokens": tokens,
    }


def pct(values, q):
    s = sorted(values)
    if not s:
        return 0.0
    return s[min(len(s) - 1, max(0, round((len(s) - 1) * q)))]


def run_phase(pool, schedule, seq_start, warmup, args, prompt_dir):
    results = []
    wave_size = args.concurrency if args.fixed_batches else max(1, len(schedule))
    for wave_start in range(0, len(schedule), wave_size):
        futures = {}
        for offset, (length, idx) in enumerate(
            schedule[wave_start : wave_start + wave_size]
        ):
            prompt = load_prompt(prompt_dir, length, idx)
            future = pool.submit(
                send_request,
                args.url, args.model, prompt, args.max_tokens, args.timeout,
            )
            futures[future] = (seq_start + wave_start + offset, length)
        for future in as_completed(futures):
            seq, length = futures[future]
            result = future.result()
            result.update(seq=seq, warmup=warmup, length=length)
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384, 32000])
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--fixed-batches", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    prompt_dir = Path(args.prompt_dir)
    rng = random.Random(args.seed)
    lengths = args.lengths

    # Build schedule: balance lengths
    schedule = []
    for i in range(args.requests + args.warmup):
        length = lengths[i % len(lengths)]
        idx = rng.randint(0, 9)  # up to 10 prompts per length
        schedule.append((length, idx))

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = run_phase(
            pool, schedule[:args.warmup], 0, True, args, prompt_dir
        )
        formal_started = time.perf_counter()
        results.extend(run_phase(
            pool, schedule[args.warmup:], args.warmup, False, args, prompt_dir
        ))
        formal_elapsed_s = time.perf_counter() - formal_started

    failed = sum(not result["ok"] for result in results)

    # Sort by seq
    results.sort(key=lambda r: r["seq"])

    # Write JSONL output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary (exclude warmup)
    formal = [r for r in results if not r["warmup"] and r["ok"]]
    ttft = [r["ttft_s"] for r in formal if r["ttft_s"] is not None]
    e2e = [r["e2e_s"] for r in formal]
    post_ttft = [r["post_ttft_s"] for r in formal if r["post_ttft_s"] is not None]

    summary = {
        "requests": len(formal),
        "failed": failed,
        "warmup_attempted": args.warmup,
        "formal_attempted": args.requests,
        "formal_elapsed_s": formal_elapsed_s,
        "completed_rps": len(formal) / formal_elapsed_s if formal_elapsed_s else 0.0,
        "submission_mode": "fixed_batches" if args.fixed_batches else "closed_loop",
        "ttft": {
            "p50": pct(ttft, 0.5),
            "p95": pct(ttft, 0.95),
            "p99": pct(ttft, 0.99),
        },
        "e2e": {
            "p50": pct(e2e, 0.5),
            "p95": pct(e2e, 0.95),
            "p99": pct(e2e, 0.99),
        },
        "post_ttft": {
            "p50": pct(post_ttft, 0.5),
            "p95": pct(post_ttft, 0.95),
            "p99": pct(post_ttft, 0.99),
        },
        "by_length": {},
    }
    for length in lengths:
        by_len = [r for r in formal if r.get("length") == length]
        summary["by_length"][str(length)] = {
            "requests": len(by_len),
            "ttft_p50": pct([r["ttft_s"] for r in by_len if r["ttft_s"] is not None], 0.5),
        }

    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    if failed:
        print(f"{failed} requests failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
