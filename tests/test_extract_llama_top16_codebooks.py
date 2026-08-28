import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_llama_top16_codebooks.py"
SPEC = importlib.util.spec_from_file_location("extract_llama_top16_codebooks", SCRIPT)
extractor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extractor)


class ExtractLlamaTop16CodebooksTest(unittest.TestCase):
    def test_extracts_four_kv_codebooks_in_layer_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            with trace.open("w", encoding="utf-8") as handle:
                for layer in range(28, 32):
                    histogram = [0] * 256
                    for exponent in range(16):
                        histogram[exponent] = 16 - exponent
                    handle.write(
                        json.dumps(
                            {
                                "event": "bf16_exponent_stats",
                                "layer": layer,
                                "k": {"histogram": histogram},
                                "v": {"histogram": histogram},
                            }
                        )
                        + "\n"
                    )
            result = extractor.extract([trace])

        self.assertTrue(result["passed"])
        self.assertEqual(result["extension_bytes"], 128)
        self.assertEqual(result["extension_hex"][:32], bytes(range(16)).hex())


if __name__ == "__main__":
    unittest.main()
