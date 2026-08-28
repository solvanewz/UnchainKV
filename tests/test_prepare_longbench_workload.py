import unittest

from scripts.prepare_longbench_workload import choose_balanced


class PrepareLongBenchWorkloadTest(unittest.TestCase):
    def test_selection_is_deterministic_and_balanced(self):
        rows = [
            {"dataset": dataset, "sample_id": f"{dataset}-{index}"}
            for dataset in ("a", "b")
            for index in range(3)
        ]
        selected = choose_balanced(rows, 4, 7)

        self.assertEqual([row["dataset"] for row in selected], ["a", "b", "a", "b"])
        self.assertEqual(selected, choose_balanced(rows, 4, 7))


if __name__ == "__main__":
    unittest.main()
