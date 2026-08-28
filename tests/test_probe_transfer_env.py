import importlib.util
from pathlib import Path
import unittest


def load_probe_module():
    path = Path("scripts/probe_transfer_env.py")
    spec = importlib.util.spec_from_file_location("probe_transfer_env", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProbeTransferEnvTest(unittest.TestCase):
    def test_summarize_flags_missing_registered_memory_prereqs(self):
        probe = load_probe_module()
        summary = probe.summarize(
            {
                "gpu_count": 2,
                "infiniband_devices": [],
                "libs": ["libibverbs.so.1", "libgdrapi.so.2", "libcuda.so.1"],
                "modules": ["ib_core"],
                "devices": ["/dev/nvidia0"],
                "network_pci": ["Intel Corporation I350 Gigabit Network Connection"],
            }
        )

        self.assertFalse(summary["rdma_ready"])
        self.assertFalse(summary["gdr_ready"])
        self.assertFalse(summary["gpudirect_rdma_ready"])
        self.assertIn("/dev/infiniband", summary["missing"])
        self.assertIn("/dev/gdrdrv", summary["missing"])
        self.assertIn("nvidia_peermem", summary["missing"])


if __name__ == "__main__":
    unittest.main()
