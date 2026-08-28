#!/usr/bin/env python3
"""Issue an exact balanced mixed-context schedule and emit benchmark JSON."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import time

from scripts.measure_ttft import bench_result, post_stream


def build_schedule(
    lengths: list[int], requests: int, seed: int
) -> list[tuple[int, int]]:
    if not lengths or requests % len(lengths):
        raise ValueError("requests must be divisible by the number of lengths")
    schedule = [
        (length, index)
        for index in range(requests // len(lengths))
        for length in lengths
    ]
    random.Random(seed).shuffle(schedule)
    return schedule


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-dir", type=Path)
    parser.add_argument("--lengths", type=int, nargs="+")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument("--max-duration-s", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bench-output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.manifest) == bool(args.prompt_dir and args.lengths):
        parser.error("set either --manifest or both --prompt-dir and --lengths")
    if args.manifest:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        schedule = payload.get("samples", [])
        if len(schedule) < args.requests:
            parser.error("manifest has fewer samples than --requests")
        schedule = schedule[: args.requests]
    else:
        try:
            schedule = build_schedule(args.lengths, args.requests, args.seed)
        except ValueError as exc:
            parser.error(str(exc))

    def request(item: object) -> dict[str, object]:
        if isinstance(item, dict):
            length = int(item["input_tokens"])
            index = int(item["index"])
            prompt = (args.manifest.parent / str(item["prompt"])).read_text(
                encoding="utf-8"
            )
            sample_id = str(item["sample_id"])
            max_tokens = int(item.get("output_tokens", args.max_tokens))
        else:
            length, index = item
            prompt = (args.prompt_dir / f"prompt-{length}-{index % 10}.txt").read_text(
                encoding="utf-8"
            )
            sample_id = f"context-{length}-{index % 10}"
            max_tokens = args.max_tokens
        row = post_stream(
            args.url.rstrip("/") + "/v1/completions",
            {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "ignore_eos": True,
            },
            args.timeout_s,
        )
        row.update(input_tokens=length, prompt_index=index, sample_id=sample_id)
        if isinstance(item, dict):
            row.update(
                cache=str(item.get("cache", "")),
                requested_output_tokens=max_tokens,
            )
        return row

    started = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for start in range(0, len(schedule), args.concurrency):
            if args.max_duration_s and time.perf_counter() - started >= args.max_duration_s:
                break
            rows.extend(pool.map(request, schedule[start : start + args.concurrency]))
    duration_s = time.perf_counter() - started
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    args.bench_output.write_text(
        json.dumps(bench_result(rows, 0, duration_s), sort_keys=True),
        encoding="utf-8",
    )
    return 0 if all(row.get("ok") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
