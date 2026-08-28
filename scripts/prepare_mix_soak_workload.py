#!/usr/bin/env python3
"""Freeze the deterministic 2,000-request MIX-SOAK schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--tokenizer")
    args = parser.parse_args()
    lengths = (1024, 4096, 8192, 16384, 32000)
    outputs = (1, 8, 32, 128, 256)
    cache_modes = ("cold", "hot")
    combinations = [(length, output, cache) for length in lengths for output in outputs for cache in cache_modes]
    if args.requests % len(combinations):
        parser.error(f"--requests must be divisible by {len(combinations)}")
    samples = []
    prompt_hashes = {}
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        (args.output.parent / "prompts").mkdir(parents=True, exist_ok=True)
    for index in range(args.requests):
        length, output, cache = combinations[index % len(combinations)]
        prompt_index = 0 if cache == "hot" else (index // len(combinations)) % 10
        source = args.prompt_dir / f"prompt-{length}-{prompt_index}.txt"
        if not source.is_file():
            parser.error(f"missing prompt: {source}")
        prompt = source
        if cache == "cold" and tokenizer is not None:
            source_ids = tokenizer.encode(source.read_text(encoding="utf-8"), add_special_tokens=False)
            nonce_ids = tokenizer.encode(f" cold-{index:04d}", add_special_tokens=False)
            if len(nonce_ids) >= length:
                parser.error("nonce is longer than requested prompt")
            text = tokenizer.decode(source_ids[: length - len(nonce_ids)] + nonce_ids)
            actual = tokenizer.encode(text, add_special_tokens=False)
            if len(actual) != length:
                parser.error(f"tokenizer round-trip changed {length} to {len(actual)} tokens")
            prompt = args.output.parent / "prompts" / f"cold-{index:04d}.txt"
            prompt.write_text(text, encoding="utf-8")
        prompt_hashes[str(prompt)] = digest_file(prompt)
        samples.append(
            {
                "cache": cache,
                "index": prompt_index,
                "input_tokens": length,
                "output_tokens": output,
                "prompt": str(prompt),
                "sample_id": f"mix-soak-{index:04d}",
            }
        )
    random.Random(args.seed).shuffle(samples)
    payload = {
        "name": "MIX-SOAK",
        "sample_count": len(samples),
        "seed": args.seed,
        "lengths": list(lengths),
        "output_tokens": list(outputs),
        "cache_modes": list(cache_modes),
        "prompt_sha256": prompt_hashes,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
