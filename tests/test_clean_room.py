from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "import kvx",
    "from kvx",
    "turbo_udp",
    "qwen_vllm_pd_smoke",
    "agent_bench_harness",
    "inject_kvx_runtime",
    "inject_vllm_omni_kvx",
)


class CleanRoomTest(unittest.TestCase):
    def test_source_does_not_reference_old_kvx_code(self):
        offenders = []
        for path in (ROOT / "src").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN:
                if token in text:
                    offenders.append(f"{path.relative_to(ROOT)}: {token}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
