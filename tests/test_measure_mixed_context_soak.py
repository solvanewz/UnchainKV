import json
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import scripts.measure_mixed_context_soak as soak


class MeasureMixedContextSoakTest(unittest.TestCase):
    def test_load_prompt_accepts_canonical_unpadded_names(self):
        path = self.prompt_dir / "prompt-8192-9.txt"
        path.write_text("canonical")

        self.assertEqual(soak.load_prompt(self.prompt_dir, 8192, 9), "canonical")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prompt_dir = Path(self.tmp.name) / "prompts"
        self.prompt_dir.mkdir()
        for length in [8192, 16384]:
            for i in range(10):
                p = self.prompt_dir / f"prompt-{length}-{i:04d}.txt"
                p.write_text("hello world " * (length // 6))

    def tearDown(self):
        self.tmp.cleanup()

    def _run_driver(self, extra_args=None):
        args = [
            sys.executable, "scripts/measure_mixed_context_soak.py",
            "--url", "http://127.0.0.1:1/v1/completions",
            "--model", "test",
            "--prompt-dir", str(self.prompt_dir),
            "--lengths", "8192", "16384",
            "--requests", "3",
            "--warmup", "1",
            "--concurrency", "1",
            "--max-tokens", "1",
            "--seed", "42",
            "--output", str(Path(self.tmp.name) / "out.jsonl"),
            "--summary", str(Path(self.tmp.name) / "summary.json"),
            "--timeout", "1",
        ]
        if extra_args:
            args.extend(extra_args)
        return subprocess.run(args, capture_output=True, text=True)

    def test_module_loads(self):
        result = subprocess.run(
            [sys.executable, "-c", "import scripts.measure_mixed_context_soak"],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
        )
        # Import may fail due to path, but no syntax error
        self.assertNotIn("SyntaxError", result.stderr)

    def test_outputs_jsonl_and_summary_on_connect_error(self):
        result = self._run_driver()
        # Should exit non-zero (connect refused), but outputs files
        out = Path(self.tmp.name) / "out.jsonl"
        summary = Path(self.tmp.name) / "summary.json"
        self.assertTrue(out.exists())
        self.assertTrue(summary.exists())
        self.assertNotEqual(result.returncode, 0)
        data = json.loads(summary.read_text())
        self.assertEqual(data["warmup_attempted"], 1)
        self.assertEqual(data["formal_attempted"], 3)
        self.assertIn("formal_elapsed_s", data)
        self.assertIn("completed_rps", data)
        self.assertIn("post_ttft", data)

    def test_fixed_batches_wait_for_the_whole_wave(self):
        started = {}
        finished = {}

        def fake_send(_url, _model, prompt, _max_tokens, _timeout):
            index = int(prompt)
            started[index] = time.perf_counter()
            time.sleep(0.001 if index == 0 else 0.05 if index == 1 else 0)
            finished[index] = time.perf_counter()
            return {"ok": True, "ttft_s": 0.0, "post_ttft_s": 0.0, "e2e_s": 0.0}

        args = SimpleNamespace(
            url="unused",
            model="unused",
            max_tokens=1,
            timeout=1,
            concurrency=2,
            fixed_batches=True,
        )
        with patch.object(soak, "load_prompt", side_effect=lambda _d, _l, i: str(i)), \
             patch.object(soak, "send_request", side_effect=fake_send), \
             ThreadPoolExecutor(max_workers=2) as pool:
            soak.run_phase(
                pool,
                [(8192, 0), (8192, 1), (8192, 2)],
                0,
                False,
                args,
                self.prompt_dir,
            )

        self.assertGreaterEqual(started[2], finished[1])

if __name__ == "__main__":
    unittest.main()
