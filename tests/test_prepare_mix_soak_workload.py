import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class PrepareMixSoakWorkloadTest(unittest.TestCase):
    def test_balanced_frozen_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for length in (1024, 4096, 8192, 16384, 32000):
                for index in range(10):
                    (root / f"prompt-{length}-{index}.txt").write_text(f"{length}-{index}", encoding="utf-8")
            output = root / "manifest.json"
            subprocess.run(["python3", "scripts/prepare_mix_soak_workload.py", str(root), str(output), "--requests", "100"], check=True)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["sample_count"], 100)
        self.assertEqual({row["cache"] for row in payload["samples"]}, {"cold", "hot"})
        self.assertEqual({row["output_tokens"] for row in payload["samples"]}, {1, 8, 32, 128, 256})
