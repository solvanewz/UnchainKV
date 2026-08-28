import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_codec_matrix.py"
SPEC = importlib.util.spec_from_file_location("bench_codec_matrix", SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class BenchCodecMatrixTest(unittest.TestCase):
    def test_block_ids_realize_requested_runs_up_to_block_count(self):
        for blocks, requested, expected in ((64, 1, 1), (64, 8, 8), (64, 512, 64), (2000, 512, 512)):
            with self.subTest(blocks=blocks, requested=requested):
                ids = bench.block_ids_for_runs(blocks, requested)
                self.assertEqual(len(ids), blocks)
                self.assertEqual(len(set(ids)), blocks)
                self.assertEqual(bench.count_runs(ids), expected)

    def test_invalid_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            bench.block_ids_for_runs(0, 1)


if __name__ == "__main__":
    unittest.main()
