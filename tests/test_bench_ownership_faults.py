from __future__ import annotations

import unittest
from unittest.mock import patch
import weakref

from scripts import bench_ownership_faults
from scripts.bench_ownership_faults import run_c4, run_c5


class BenchOwnershipFaultsTest(unittest.TestCase):
    def test_bounded_fault_campaigns_fail_closed(self):
        c4 = run_c4(2, "cpu")
        c5 = run_c5(2)

        self.assertTrue(c4["passed"])
        self.assertEqual(c4["ownership"]["double_release"], 0)
        self.assertTrue(c5["passed"])
        self.assertEqual(c5["decode_started"], 0)

    def test_c4_does_not_retain_the_loop_tensor(self):
        live = weakref.WeakSet()

        def tensors(_device):
            baseline = len(live)
            values = tuple(bench_ownership_faults._Tensor(4096) for _ in range(3))
            live.update(values)
            return (*values, baseline)

        with (
            patch.object(bench_ownership_faults, "_tensors", tensors),
            patch.object(bench_ownership_faults, "_gpu_allocated", lambda _device: len(live)),
        ):
            self.assertTrue(run_c4(1, "cuda:0")["passed"])


if __name__ == "__main__":
    unittest.main()
