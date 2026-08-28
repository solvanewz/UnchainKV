import importlib.util
import http.client
import io
from pathlib import Path
import threading
import time
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_ttft.py"
spec = importlib.util.spec_from_file_location("measure_ttft", SCRIPT)
measure_ttft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure_ttft)


class MeasureTtftTest(unittest.TestCase):
    class ChunkedRaw(io.RawIOBase):
        def __init__(self, chunks):
            self.chunks = iter(chunks)
            self.pending = b""

        def readable(self):
            return True

        def readinto(self, buffer):
            while not self.pending:
                try:
                    self.pending = next(self.chunks)
                except StopIteration:
                    return 0
            size = min(len(buffer), len(self.pending))
            buffer[:size] = self.pending[:size]
            self.pending = self.pending[size:]
            return size

    class FakeSocket:
        def __init__(self, chunks):
            self.chunks = chunks

        def makefile(self, _mode):
            return io.BufferedReader(MeasureTtftTest.ChunkedRaw(self.chunks))

    def test_prompt_cycle_reuses_frozen_files_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prompt-16-0.txt").write_text("zero")
            (root / "prompt-16-1.txt").write_text("one")
            seen = []

            def fake_post(_url, payload, _timeout):
                seen.append(payload["prompt"])
                return {"ok": True, "ttft_s": 0.1, "e2e_s": 0.2, "chunks": 1,
                        "output_tokens": 1, "generated_text": "x"}

            argv = ["measure_ttft.py", "--url", "http://unused", "--model", "m",
                    "--prompt-dir", directory, "--prompt-length", "16",
                    "--prompt-cycle", "2", "--output", str(root / "rows.jsonl"),
                    "--runs", "3", "--warmup", "1", "--bench-output", str(root / "bench.json")]
            with patch.object(sys, "argv", argv), patch.object(measure_ttft, "post_stream", fake_post):
                self.assertEqual(measure_ttft.main(), 0)
            self.assertEqual(seen, ["zero", "zero", "one", "zero"])

    def test_first_text_from_completion_stream_event(self):
        event = {"choices": [{"text": "hello"}]}
        self.assertEqual(measure_ttft.first_text(event), "hello")

    def test_first_text_from_chat_stream_event(self):
        event = {"choices": [{"delta": {"content": "hello"}}]}
        self.assertEqual(measure_ttft.first_text(event), "hello")

    def test_completion_tokens_from_usage_event(self):
        event = {"choices": [], "usage": {"completion_tokens": 8}}
        self.assertEqual(measure_ttft.completion_tokens(event), 8)

    def test_post_stream_parses_cross_read_multiple_event_timeline(self):
        body = (
            b'data: {"choices":[{"text":"hel"}]}\n\n'
            b'data: {"choices":[{"text":"lo"}]}\n\n'
            b'data: {"choices":[{"text":"","finish_reason":"length"}]}\n\n'
            b'data: {"choices":[],"usage":{"completion_tokens":2}}\n\n'
            b'data: [DONE]\n\n'
        )
        raw = (
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        response = http.client.HTTPResponse(
            self.FakeSocket([raw[:71], raw[71:94], raw[94:123], raw[123:]])
        )
        response.begin()

        with patch.object(measure_ttft.urllib.request, "urlopen", return_value=response), patch.object(
            measure_ttft.time, "perf_counter", side_effect=[10.0, 10.5, 10.7, 10.8, 10.9, 11.0]
        ):
            row = measure_ttft.post_stream("http://unused", {"max_tokens": 2}, 1.0)

        self.assertTrue(row["ok"])
        self.assertEqual(row["generated_text"], "hello")
        self.assertEqual(row["output_tokens"], 2)
        self.assertEqual(row["finish_reason"], "length")
        self.assertEqual(row["ttft_s"], 0.5)
        self.assertEqual(row["e2e_s"], 1.0)
        self.assertEqual(len(row["timeline"]), 4)

    def test_post_stream_records_timeout_as_failure(self):
        with patch.object(
            measure_ttft.urllib.request, "urlopen", side_effect=TimeoutError("late")
        ), patch.object(measure_ttft.time, "perf_counter", side_effect=[1.0, 3.0]):
            row = measure_ttft.post_stream("http://unused", {"max_tokens": 2}, 1.0)

        self.assertFalse(row["ok"])
        self.assertEqual(row["error"], "TimeoutError: late")
        self.assertEqual(row["e2e_s"], 2.0)

    def test_post_stream_records_malformed_event_as_failure(self):
        body = b"data: {bad json}\n\n"
        with patch.object(measure_ttft.urllib.request, "urlopen", return_value=io.BytesIO(body)):
            row = measure_ttft.post_stream("http://unused", {"max_tokens": 2}, 1.0)

        self.assertFalse(row["ok"])
        self.assertIn("JSONDecodeError", row["error"])

    def test_post_stream_rejects_zero_output_and_token_mismatch(self):
        zero = b'data: {"choices":[],"usage":{"completion_tokens":0}}\n\ndata: [DONE]\n\n'
        mismatch = (
            b'data: {"choices":[{"text":"x"}]}\n\n'
            b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
            b'data: [DONE]\n\n'
        )
        with patch.object(measure_ttft.urllib.request, "urlopen", return_value=io.BytesIO(zero)):
            zero_row = measure_ttft.post_stream("http://unused", {"max_tokens": 2}, 1.0)
        with patch.object(measure_ttft.urllib.request, "urlopen", return_value=io.BytesIO(mismatch)):
            mismatch_row = measure_ttft.post_stream(
                "http://unused", {"max_tokens": 2, "ignore_eos": True}, 1.0
            )

        self.assertEqual(zero_row["error"], "no output text")
        self.assertEqual(mismatch_row["error"], "token mismatch: expected 2, got 1")

    def test_summary_uses_sorted_values(self):
        rows = [
            {"ok": True, "ttft_s": 3.0, "e2e_s": 5.0},
            {"ok": True, "ttft_s": 1.0, "e2e_s": 2.0},
            {"ok": True, "ttft_s": 2.0, "e2e_s": 4.0},
        ]

        summary = measure_ttft.summarize(rows)

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["ttft_median_s"], 2.0)
        self.assertEqual(summary["e2e_median_s"], 4.0)
        self.assertEqual(summary["ttft_min_s"], 1.0)
        self.assertEqual(summary["ttft_max_s"], 3.0)
        self.assertEqual(summary["ttft_p90_s"], 3.0)
        self.assertEqual(summary["ttft_p99_s"], 3.0)

    def test_run_batch_starts_requested_number_concurrently(self):
        lock = threading.Lock()
        release = threading.Event()
        active = 0
        peak = 0

        def request():
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            release.wait(1.0)
            with lock:
                active -= 1
            return {"ok": True}

        timer = threading.Timer(0.05, release.set)
        timer.start()
        try:
            rows = measure_ttft.run_batch(request, 2)
        finally:
            timer.cancel()

        self.assertEqual(rows, [{"ok": True}, {"ok": True}])
        self.assertEqual(peak, 2)

    def test_bench_result_matches_matrix_summarizer_schema(self):
        result = measure_ttft.bench_result(
            [
                {
                    "ok": True,
                    "ttft_s": 1.0,
                    "e2e_s": 1.7,
                    "chunks": 7,
                    "output_tokens": 8,
                    "generated_text": "one",
                    "finish_reason": "length",
                },
                {
                    "ok": True,
                    "ttft_s": 2.0,
                    "e2e_s": 2.7,
                    "chunks": 8,
                    "output_tokens": 8,
                    "generated_text": "two",
                    "finish_reason": "length",
                },
            ],
            input_tokens=1024,
            duration_s=4.0,
        )

        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["total_input_tokens"], 2048)
        self.assertEqual(result["output_lens"], [8, 8])
        self.assertEqual(result["generated_texts"], ["one", "two"])
        self.assertEqual(result["finish_reasons"], ["length", "length"])
        self.assertEqual(result["ttfts"], [1.0, 2.0])
        self.assertEqual(result["e2els"], [1.7, 2.7])
        self.assertAlmostEqual(result["p50_tpot_ms"], 100.0)
        self.assertEqual(result["p50_e2el_ms"], 1700.0)

    def test_bench_result_aligns_failures_without_polluting_percentiles(self):
        result = measure_ttft.bench_result(
            [
                {"ok": True, "ttft_s": 1.0, "e2e_s": 2.0, "output_tokens": 2,
                 "generated_text": "ok"},
                {"ok": False, "error": "TimeoutError: late", "ttft_s": None,
                 "e2e_s": 9.0, "output_tokens": None, "generated_text": ""},
            ],
            input_tokens=16,
            duration_s=10.0,
        )

        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["total_input_tokens"], 16)
        self.assertEqual(result["total_output_tokens"], 2)
        self.assertEqual(result["errors"], ["", "TimeoutError: late"])
        self.assertEqual(result["ttfts"], [1.0, 0.0])
        self.assertEqual(result["p50_ttft_ms"], 1000.0)


if __name__ == "__main__":
    unittest.main()
