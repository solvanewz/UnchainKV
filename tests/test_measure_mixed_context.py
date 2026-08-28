from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.measure_mixed_context import build_schedule, main
from scripts.measure_ttft import bench_result


class MeasureMixedContextTest(unittest.TestCase):
    def test_balanced_schedule_and_per_request_input_totals(self):
        lengths = [1024, 4096, 8192, 16384, 32000]
        schedule = build_schedule(lengths, 100, 7)
        self.assertEqual(Counter(length for length, _index in schedule), {length: 20 for length in lengths})

        rows = [
            {
                "ok": True,
                "ttft_s": 0.1,
                "e2e_s": 0.2,
                "output_tokens": 1,
                "generated_text": "x",
                "input_tokens": length,
            }
            for length in lengths
        ]
        result = bench_result(rows, 0, 1.0)
        self.assertEqual(result["input_lens"], lengths)
        self.assertEqual(result["total_input_tokens"], sum(lengths))

    def test_manifest_controls_output_length(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prompt.txt").write_text("prompt", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {"input_tokens": 8, "index": 0, "prompt": "prompt.txt", "sample_id": "cold-0", "cache": "cold", "output_tokens": 1},
                            {"input_tokens": 8, "index": 1, "prompt": "prompt.txt", "sample_id": "hot-0", "cache": "hot", "output_tokens": 7},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            payloads = []

            def fake_post(_url, payload, _timeout):
                payloads.append(payload)
                return {"ok": True, "ttft_s": 0.1, "e2e_s": 0.2, "output_tokens": payload["max_tokens"], "finish_reason": "length"}

            argv = ["measure_mixed_context.py", "--url", "http://test", "--model", "m", "--manifest", str(manifest), "--requests", "2", "--concurrency", "1", "--output", str(root / "rows.jsonl"), "--bench-output", str(root / "bench.json")]
            with patch.object(sys, "argv", argv), patch("scripts.measure_mixed_context.post_stream", side_effect=fake_post):
                self.assertEqual(main(), 0)

        self.assertEqual([row["max_tokens"] for row in payloads], [1, 7])


if __name__ == "__main__":
    unittest.main()
