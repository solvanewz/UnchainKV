import json
from pathlib import Path
import tempfile
import unittest

from unchain_kv.trace import TraceWriter, summarize_symbol_counts


class TraceTest(unittest.TestCase):
    def test_noop_writer(self):
        TraceWriter(None).event("ignored", value=1)

    def test_writer_serializes_event(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            TraceWriter(path).event("ready", layer=3)
            row = json.loads(path.read_text())
            self.assertEqual(row["event"], "ready")
            self.assertEqual(row["layer"], 3)

    def test_summarize_symbol_counts(self):
        result = summarize_symbol_counts([3, 1, 0, 0])
        self.assertEqual(result["words"], 4)
        self.assertEqual(result["top8_exponents"], [0, 1])


if __name__ == "__main__":
    unittest.main()
