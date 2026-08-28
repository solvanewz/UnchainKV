#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
import urllib.request


def first_text(event: dict) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if choice.get("text"):
        return str(choice["text"])
    delta = choice.get("delta") or {}
    return str(delta.get("content") or "")


def completion_tokens(event: dict) -> int | None:
    usage = event.get("usage") or {}
    value = usage.get("completion_tokens")
    return None if value is None else int(value)


def post_stream(url: str, payload: dict, timeout_s: float) -> dict:
    payload = dict(payload)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = None
    chunks = 0
    output_tokens = None
    finish_reason = None
    text = []
    timeline = []
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.strip()
                if not line.startswith(b"data:"):
                    continue
                body = line[5:].strip()
                if body == b"[DONE]":
                    break
                event = json.loads(body.decode("utf-8"))
                now = time.perf_counter()
                timeline.append({"elapsed_s": now - start, "event": event})
                usage_tokens = completion_tokens(event)
                if usage_tokens is not None:
                    output_tokens = usage_tokens
                choices = event.get("choices") or []
                if choices and choices[0].get("finish_reason") is not None:
                    finish_reason = str(choices[0]["finish_reason"])
                piece = first_text(event)
                if piece and first is None:
                    first = now
                if piece:
                    text.append(piece)
                    chunks += 1
    except Exception as exc:
        end = time.perf_counter()
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "ttft_s": None if first is None else first - start,
            "e2e_s": end - start,
            "chunks": chunks,
            "output_tokens": output_tokens,
            "finish_reason": finish_reason,
            "generated_text": "".join(text),
            "timeline": timeline,
        }
    end = time.perf_counter()
    expected_tokens = payload.get("max_tokens") if payload.get("ignore_eos") else None
    if first is None:
        error = "no output text"
    elif output_tokens is None:
        error = "missing usage completion_tokens"
    elif output_tokens < 0:
        error = "negative completion_tokens"
    elif expected_tokens is not None and output_tokens != int(expected_tokens):
        error = f"token mismatch: expected {expected_tokens}, got {output_tokens}"
    else:
        error = ""
    return {
        "ok": not error,
        "error": error,
        "ttft_s": None if first is None else first - start,
        "e2e_s": end - start,
        "chunks": chunks,
        "output_tokens": output_tokens,
        "finish_reason": finish_reason,
        "generated_text": "".join(text),
        "timeline": timeline,
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[index]


def summarize(rows: list[dict]) -> dict:
    ok = [row for row in rows if row.get("ok") and row.get("ttft_s") is not None]
    ttft = [float(row["ttft_s"]) for row in ok]
    e2e = [float(row["e2e_s"]) for row in ok]
    return {
        "count": len(rows),
        "ok": len(ok),
        "ttft_median_s": statistics.median(ttft) if ttft else 0.0,
        "ttft_p90_s": percentile(ttft, 0.90),
        "ttft_p99_s": percentile(ttft, 0.99),
        "ttft_min_s": min(ttft, default=0.0),
        "ttft_max_s": max(ttft, default=0.0),
        "e2e_median_s": statistics.median(e2e) if e2e else 0.0,
        "e2e_p90_s": percentile(e2e, 0.90),
        "e2e_p99_s": percentile(e2e, 0.99),
        "e2e_min_s": min(e2e, default=0.0),
        "e2e_max_s": max(e2e, default=0.0),
    }


def run_batch(request, concurrency: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda _index: request(), range(concurrency)))


def bench_result(
    rows: list[dict], input_tokens: int, duration_s: float
) -> dict[str, object]:
    ok = [row for row in rows if row.get("ok") and row.get("ttft_s") is not None]
    ttft_ms = [float(row["ttft_s"]) * 1000 for row in ok]
    e2e_ms = [float(row["e2e_s"]) * 1000 for row in ok]
    tpot_ms = [
        (float(row["e2e_s"]) - float(row["ttft_s"]))
        / max(1, int(row["output_tokens"]) - 1)
        * 1000
        for row in ok
    ]
    output_lens = [int(row["output_tokens"]) for row in ok]
    completed = len(ok)
    input_lens = [int(row.get("input_tokens", input_tokens)) for row in rows]
    total_input = sum(
        input_lens[index] for index, row in enumerate(rows) if row.get("ok")
    )
    total_output = sum(output_lens)

    def pct(values: list[float], q: float) -> float:
        return percentile(values, q) if values else 0.0

    return {
        "num_prompts": len(rows),
        "completed": completed,
        "failed": len(rows) - completed,
        "duration": duration_s,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": completed / duration_s,
        "input_throughput": total_input / duration_s,
        "output_throughput": total_output / duration_s,
        "total_token_throughput": (total_input + total_output) / duration_s,
        "input_lens": input_lens,
        "output_lens": [int(row.get("output_tokens") or 0) if row.get("ok") else 0 for row in rows],
        "generated_texts": [str(row.get("generated_text") or "") for row in rows],
        "finish_reasons": [row.get("finish_reason") for row in rows],
        "errors": ["" if row.get("ok") else str(row.get("error") or "request failed") for row in rows],
        "ttfts": [float(row["ttft_s"]) if row.get("ttft_s") is not None else 0.0 for row in rows],
        "e2els": [float(row.get("e2e_s") or 0.0) for row in rows],
        "p50_ttft_ms": pct(ttft_ms, 0.50),
        "p95_ttft_ms": pct(ttft_ms, 0.95),
        "p99_ttft_ms": pct(ttft_ms, 0.99),
        "p50_tpot_ms": pct(tpot_ms, 0.50),
        "p95_tpot_ms": pct(tpot_ms, 0.95),
        "p99_tpot_ms": pct(tpot_ms, 0.99),
        "p50_e2el_ms": pct(e2e_ms, 0.50),
        "p95_e2el_ms": pct(e2e_ms, 0.95),
        "p99_e2el_ms": pct(e2e_ms, 0.99),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt-file")
    prompts.add_argument("--prompt-dir", type=Path)
    parser.add_argument("--prompt-length", type=int)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--prompt-cycle", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bench-output", type=Path)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args()

    if args.prompt_dir and args.prompt_length is None:
        parser.error("--prompt-dir requires --prompt-length")
    if args.prompt_dir and args.concurrency != 1:
        parser.error("--prompt-dir currently requires --concurrency=1")
    if args.prompt_cycle < 0:
        parser.error("--prompt-cycle must be non-negative")

    def load_prompt(index: int) -> str:
        if args.prompt_cycle:
            index %= args.prompt_cycle
        path = (
            args.prompt_dir
            / f"prompt-{args.prompt_length}-{args.sample_offset + index}.txt"
            if args.prompt_dir
            else Path(args.prompt_file)
        )
        return path.read_text(encoding="utf-8")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")

    rows = []
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    formal_start = 0.0
    for trial in range(args.warmup + args.runs):
        if trial == args.warmup:
            formal_start = time.perf_counter()
        formal_index = max(0, trial - args.warmup)
        prompt_index = args.sample_offset + (
            formal_index % args.prompt_cycle if args.prompt_cycle else formal_index
        )
        prompt = load_prompt(formal_index)
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "ignore_eos": args.ignore_eos,
        }
        batch = run_batch(
            lambda: post_stream(
                args.url.rstrip("/") + "/v1/completions",
                payload,
                args.timeout_s,
            ),
            args.concurrency,
        )
        for request_index, row in enumerate(batch):
            row.update(
                {
                    "index": trial * args.concurrency + request_index,
                    "trial": trial,
                    "request": request_index,
                    "concurrency": args.concurrency,
                    "warmup": trial < args.warmup,
                    "prompt_chars": len(prompt),
                    "prompt_index": prompt_index,
                }
            )
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if not row["warmup"]:
                rows.append(row)

    duration_s = time.perf_counter() - formal_start
    result = summarize(rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.bench_output:
        input_tokens = args.prompt_length
        if input_tokens is None:
            parser.error("--bench-output requires --prompt-length")
        args.bench_output.write_text(
            json.dumps(bench_result(rows, input_tokens, duration_s), sort_keys=True),
            encoding="utf-8",
        )
    return 0 if all(row.get("ok") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
