#!/usr/bin/env python3
"""Extract Llama layers 28-31 Top-16 K/V codebooks from exponent traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LAYERS = range(28, 32)


def extract(paths: list[Path]) -> dict[str, object]:
    counts = {layer: {plane: [0] * 256 for plane in ("k", "v")} for layer in LAYERS}
    rows = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            layer = int(row.get("layer", -1))
            if row.get("event") != "bf16_exponent_stats" or layer not in counts:
                continue
            rows += 1
            for plane in ("k", "v"):
                histogram = row[plane]["histogram"]
                if len(histogram) != 256:
                    raise ValueError(f"layer {layer} {plane} histogram has {len(histogram)} bins")
                counts[layer][plane] = [
                    current + int(value)
                    for current, value in zip(counts[layer][plane], histogram, strict=True)
                ]
    codebooks = bytearray()
    layers = []
    for layer in LAYERS:
        current = {"layer": layer}
        for plane in ("k", "v"):
            histogram = counts[layer][plane]
            ranked = sorted(range(256), key=lambda value: (-histogram[value], value))
            selected = [value for value in ranked if histogram[value] > 0][:16]
            if len(selected) != 16:
                raise ValueError(f"layer {layer} {plane} has only {len(selected)} observed exponents")
            total = sum(histogram)
            covered = sum(histogram[value] for value in selected)
            current[plane] = {
                "words": total,
                "top16_exponents": selected,
                "coverage": covered / total if total else 0.0,
            }
            codebooks.extend(selected)
        layers.append(current)
    return {
        "passed": rows >= 4,
        "trace_rows": rows,
        "layers": layers,
        "extension_hex": codebooks.hex(),
        "extension_bytes": len(codebooks),
        "extension_sha256": hashlib.sha256(codebooks).hexdigest(),
        "source_traces": [str(path) for path in paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = extract(args.paths)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
