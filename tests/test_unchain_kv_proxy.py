import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from scripts import unchain_kv_proxy
from scripts.unchain_kv_proxy import Handler, with_transfer


class UnchainKVProxyTest(unittest.TestCase):
    def test_prefill_request_is_non_streaming_and_one_token(self):
        payload = {"stream": True, "max_tokens": 8, "stream_options": {"x": 1}}

        result = with_transfer(payload, "tx", producer=True)

        self.assertFalse(result["stream"])
        self.assertEqual(result["max_tokens"], 1)
        self.assertNotIn("stream_options", result)
        self.assertTrue(result["kv_transfer_params"]["do_remote_decode"])

    def test_decode_slot_backpressures_second_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp) / "proxy.jsonl"
            Handler.configure_admission(1, metrics)
            first = Handler.acquire_decode_slot("first")
            acquired = threading.Event()

            def acquire_second():
                Handler.acquire_decode_slot("second")
                acquired.set()

            thread = threading.Thread(target=acquire_second)
            thread.start()
            time.sleep(0.02)
            self.assertFalse(acquired.is_set())
            Handler.release_decode_slot("first")
            thread.join(1.0)
            self.assertTrue(acquired.is_set())
            Handler.release_decode_slot("second")

            rows = [json.loads(line) for line in metrics.read_text().splitlines()]

        self.assertEqual(first["decode_available"], 0)
        queued = [row for row in rows if row["event"] == "admission_queued"]
        self.assertEqual(max(row["decode_queue"] for row in queued), 1)
        self.assertTrue(
            any(
                row["event"] == "admission_acquired"
                and row["request"] == "second"
                and row["wait_s"] > 0
                for row in rows
            )
        )

    def test_decode_slot_is_held_until_decode_worker_finishes(self):
        Handler.configure_admission(1, None)
        decode_release = threading.Event()
        prefill_started = threading.Event()

        def stream_decode(_url, _payload, _timeout_s, out):
            decode_release.wait(1.0)
            out.put(("end", None))

        def post_json(_url, _payload, _timeout_s, out):
            prefill_started.set()
            out.put(("prefill_error", "failed"))

        payload = json.dumps({"prompt": "x"}).encode()
        handler = object.__new__(Handler)
        handler.path = "/v1/completions"
        handler.headers = {"Content-Length": str(len(payload))}
        handler.rfile = io.BytesIO(payload)
        handler.send_error = lambda *_args: None
        handler.decode_lead_s = 0.0

        with patch.object(unchain_kv_proxy, "stream_decode", stream_decode), patch.object(
            unchain_kv_proxy, "post_json", post_json
        ):
            thread = threading.Thread(target=handler.do_POST)
            thread.start()
            self.assertTrue(prefill_started.wait(1.0))
            time.sleep(0.02)
            self.assertTrue(thread.is_alive())
            self.assertEqual(Handler.decode_inflight, 1)
            decode_release.set()
            thread.join(1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(Handler.decode_inflight, 0)

if __name__ == "__main__":
    unittest.main()
