import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "inject_fault.sh"


class InjectTpdsFaultTest(unittest.TestCase):
    def test_rejects_unknown_action_before_injection(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["bash", str(SCRIPT), "unknown", directory],
                env={**os.environ, "UNCHAIN_KV_FAULT_DELAY_S": "0"},
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported fault action", result.stderr)


if __name__ == "__main__":
    unittest.main()
