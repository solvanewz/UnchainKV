import unittest

from scripts.bench_transport_faults import run_s4


class BenchTransportFaultsTest(unittest.TestCase):
    def test_request_failures_are_isolated(self):
        result = run_s4(2)
        self.assertTrue(result["passed"])
        self.assertEqual([row["passed_cycles"] for row in result["scenarios"]], [2, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
