#!/usr/bin/env python3
"""Freeze 100 public LongBench prompts in the measured 8K--32K range."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import zipfile


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def choose_balanced(
    rows: list[dict[str, object]], count: int, seed: int
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["dataset"]), []).append(row)
    for name, values in groups.items():
        values.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{name}:{row['sample_id']}".encode()
            ).digest()
        )
    chosen = []
    while len(chosen) < count and any(groups.values()):
        for name in sorted(groups):
            if groups[name] and len(chosen) < count:
                chosen.append(groups[name].pop())
    if len(chosen) != count:
        raise ValueError(f"only {len(chosen)} eligible LongBench samples")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--prompt-config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--min-tokens", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if not 0 < args.min_tokens <= args.max_tokens:
        parser.error("invalid token range")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    templates = json.loads(args.prompt_config.read_text(encoding="utf-8"))
    eligible = []
    with zipfile.ZipFile(args.data_zip) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if name.endswith(".jsonl") and not Path(name).stem.endswith("_e")
        )
        for name in names:
            dataset = Path(name).stem
            template = templates.get(dataset)
            if template is None:
                continue
            for index, line in enumerate(
                archive.read(name).decode("utf-8").splitlines()
            ):
                row = json.loads(line)
                prompt = template.format(**row)
                tokens = len(tokenizer(prompt).input_ids)
                if args.min_tokens <= tokens <= args.max_tokens:
                    sample_id = str(row.get("_id", f"{dataset}-{index}"))
                    eligible.append(
                        {
                            "dataset": dataset,
                            "sample_id": sample_id,
                            "input_tokens": tokens,
                            "prompt": prompt,
                        }
                    )
    try:
        selected = choose_balanced(eligible, args.samples, args.seed)
    except ValueError as error:
        parser.error(str(error))

    args.output.mkdir(parents=True)
    samples = []
    for index, row in enumerate(selected):
        name = f"prompt-{index:03d}.txt"
        (args.output / name).write_text(str(row.pop("prompt")), encoding="utf-8")
        samples.append({"index": index, "prompt": name, **row})
    lengths = [int(row["input_tokens"]) for row in samples]
    manifest = {
        "dataset": "THUDM/LongBench",
        "revision": args.revision,
        "data_zip_sha256": sha256(args.data_zip),
        "prompt_config_sha256": sha256(args.prompt_config),
        "model": str(args.model),
        "model_config_sha256": sha256(args.model / "config.json"),
        "tokenizer_sha256": sha256(args.model / "tokenizer.json"),
        "seed": args.seed,
        "sample_count": len(samples),
        "output_tokens": args.output_tokens,
        "input_tokens": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "max": max(lengths),
        },
        "samples": samples,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
