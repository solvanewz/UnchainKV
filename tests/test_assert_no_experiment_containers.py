from pathlib import Path
import os
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "assert_no_experiment_containers.sh"


class AssertNoExperimentContainersTest(unittest.TestCase):
    def run_check(self, names: list[str]) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            docker = Path(directory) / "docker"
            docker.write_text("#!/bin/sh\nprintf '%s\\n' \"$CONTAINER_NAMES\"\n")
            docker.chmod(0o755)
            env = {**os.environ, "PATH": f"{directory}:{os.environ['PATH']}", "CONTAINER_NAMES": "\n".join(names)}
            return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, check=False)

    def test_allows_infrastructure_and_rejects_experiment_container(self):
        self.assertEqual(self.run_check(["buildx_buildkit_cache0", "gpustack-worker"]).returncode, 0)
        result = self.run_check(["buildx_buildkit_cache0", "kvp-prefill"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("kvp-prefill", result.stderr)


if __name__ == "__main__":
    unittest.main()
