import unittest

from scripts.nixl_proxy import prefill_payload


class NixlProxyTest(unittest.TestCase):
    def test_prefill_request_is_one_token_and_keeps_decode_request_unchanged(self):
        payload = {
            "stream": True,
            "max_tokens": 128,
            "min_tokens": 8,
            "stream_options": {"include_usage": True},
        }

        result = prefill_payload(payload)

        self.assertEqual(payload["max_tokens"], 128)
        self.assertEqual(result["max_tokens"], 1)
        self.assertFalse(result["stream"])
        self.assertNotIn("min_tokens", result)
        self.assertNotIn("stream_options", result)
        self.assertTrue(result["kv_transfer_params"]["do_remote_decode"])


if __name__ == "__main__":
    unittest.main()
