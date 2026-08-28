from __future__ import annotations

from collections.abc import Iterable
import json
import math
from pathlib import Path
import threading
import time


class TraceWriter:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def event(self, name: str, **fields: object) -> None:
        if self.path is None:
            return
        row = {"event": name, "t": time.perf_counter(), **fields}
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")


def summarize_symbol_counts(counts: Iterable[int]) -> dict[str, object]:
    histogram = list(counts)
    words = sum(histogram)
    ranked = sorted(
        (symbol for symbol, count in enumerate(histogram) if count > 0),
        key=lambda symbol: (-histogram[symbol], symbol),
    )
    result: dict[str, object] = {
        "words": words,
        "histogram": histogram,
        "unique_exponents": len(ranked),
        "entropy_bits": -sum(
            (count / words) * math.log2(count / words)
            for count in histogram
            if count > 0
        )
        if words
        else 0.0,
    }
    for size in (8, 16):
        exponents = ranked[:size]
        covered = sum(histogram[symbol] for symbol in exponents)
        result.update(
            {
                f"top{size}_exponents": exponents,
                f"top{size}_coverage": covered / words if words else 0.0,
                f"top{size}_escape_count": words - covered,
                f"top{size}_escape_rate": (words - covered) / words if words else 0.0,
            }
        )
    return result

