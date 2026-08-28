#!/usr/bin/env python3
"""Measure Top-16/attention overlap and an ideal codec-fusion upper bound."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics


MODES = (
    "attention_only",
    "encode_only",
    "d2h_only",
    "serial_encode",
    "concurrent_encode",
    "serial_pipeline",
    "concurrent_pipeline",
    "fully_serial_pipeline",
    "ideal_fused_pipeline",
)


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50": statistics.median(ordered),
        "p90": ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)],
        "min": ordered[0],
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--codec-priority", type=int, default=0)
    parser.add_argument("--library")
    args = parser.parse_args()
    if args.library:
        os.environ["UNCHAIN_KV_SPLITZIP_LIB"] = args.library

    import torch
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from unchain_kv import splitzip_cuda

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    main_stream = torch.cuda.current_stream(device)
    codec_stream = torch.cuda.Stream(device=device, priority=args.codec_priority)
    copy_stream = torch.cuda.Stream(device=device)

    query = torch.randn(
        (args.batch, 28, args.tokens, 128), device=device, dtype=torch.bfloat16
    )
    key = torch.randn(
        (args.batch, 4, args.tokens, 128), device=device, dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    source = torch.empty(
        (2, args.batch, args.tokens, 4, 128),
        device=device,
        dtype=torch.bfloat16,
    )
    codebooks = splitzip_cuda._TOP16_CODEBOOKS
    source[0].fill_(2.0 ** (codebooks[args.layer * 32] - 127))
    source[1].fill_(2.0 ** (codebooks[args.layer * 32 + 16] - 127))
    raw_bytes = source.numel() * source.element_size()
    encoded = torch.empty(raw_bytes, dtype=torch.uint8, device=device)
    encoded_bytes = splitzip_cuda.encode_top16(source, encoded, args.layer)
    if not encoded_bytes:
        raise RuntimeError("Top-16 CUDA encoder is unavailable")
    host = torch.empty(encoded_bytes, dtype=torch.uint8, pin_memory=True)
    torch.cuda.synchronize(device)
    if int(encoded[0].item()) != 5:
        raise RuntimeError("Top-16 benchmark payload overflowed")

    serial_encode = {
        "encode_only",
        "serial_encode",
        "serial_pipeline",
        "fully_serial_pipeline",
    }
    concurrent_encode = {"concurrent_encode", "concurrent_pipeline"}
    copy_modes = {
        "d2h_only",
        "serial_pipeline",
        "concurrent_pipeline",
        "fully_serial_pipeline",
        "ideal_fused_pipeline",
    }
    attention_modes = {
        "attention_only",
        "serial_encode",
        "concurrent_encode",
        "serial_pipeline",
        "concurrent_pipeline",
        "fully_serial_pipeline",
        "ideal_fused_pipeline",
    }

    def run_once(mode: str) -> dict[str, float]:
        start = torch.cuda.Event(enable_timing=True)
        start.record(main_stream)
        encode_start = encode_end = None
        copy_start = copy_end = None
        attention_start = attention_end = None

        if mode in serial_encode:
            encode_start = torch.cuda.Event(enable_timing=True)
            encode_end = torch.cuda.Event(enable_timing=True)
            encode_start.record(main_stream)
            actual = splitzip_cuda.encode_top16(source, encoded, args.layer)
            encode_end.record(main_stream)
            if actual != encoded_bytes:
                raise RuntimeError("Top-16 encoded size changed")
        elif mode in concurrent_encode:
            codec_stream.wait_event(start)
            with torch.cuda.stream(codec_stream):
                encode_start = torch.cuda.Event(enable_timing=True)
                encode_end = torch.cuda.Event(enable_timing=True)
                encode_start.record(codec_stream)
                actual = splitzip_cuda.encode_top16(source, encoded, args.layer)
                encode_end.record(codec_stream)
            if actual != encoded_bytes:
                raise RuntimeError("Top-16 encoded size changed")

        if mode in copy_modes:
            dependency = encode_end if encode_end is not None else start
            copy_stream.wait_event(dependency)
            with torch.cuda.stream(copy_stream):
                copy_start = torch.cuda.Event(enable_timing=True)
                copy_end = torch.cuda.Event(enable_timing=True)
                copy_start.record(copy_stream)
                host.copy_(encoded[:encoded_bytes], non_blocking=True)
                copy_end.record(copy_stream)

        if mode == "fully_serial_pipeline":
            main_stream.wait_event(copy_end)

        result = None
        if mode in attention_modes:
            attention_start = torch.cuda.Event(enable_timing=True)
            attention_end = torch.cuda.Event(enable_timing=True)
            attention_start.record(main_stream)
            result = functional.scaled_dot_product_attention(
                query, key, value, is_causal=True, enable_gqa=True
            )
            attention_end.record(main_stream)

        ends = [event for event in (encode_end, copy_end, attention_end) if event]
        for event in ends:
            event.synchronize()
        del result
        row = {"wall_ms": max(start.elapsed_time(event) for event in ends)}
        if encode_end is not None:
            row["encode_ms"] = encode_start.elapsed_time(encode_end)
        if copy_end is not None:
            row["d2h_ms"] = copy_start.elapsed_time(copy_end)
        if attention_end is not None:
            row["attention_ms"] = attention_start.elapsed_time(attention_end)
        return row

    rows = []
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for mode in MODES:
            for _ in range(args.warmup):
                run_once(mode)
        samples_by_mode = {mode: [] for mode in MODES}
        for repeat in range(args.repeats):
            offset = repeat % len(MODES)
            for mode in MODES[offset:] + MODES[:offset]:
                samples_by_mode[mode].append(run_once(mode))
        for mode in MODES:
            samples = samples_by_mode[mode]
            keys = samples[0]
            rows.append(
                {
                    "mode": mode,
                    **{key: _summary([sample[key] for sample in samples]) for key in keys},
                }
            )

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(device),
                "tokens": args.tokens,
                "batch": args.batch,
                "layer": args.layer,
                "codec_priority": args.codec_priority,
                "raw_bytes": raw_bytes,
                "encoded_bytes": encoded_bytes,
                "compression_ratio": raw_bytes / encoded_bytes,
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
