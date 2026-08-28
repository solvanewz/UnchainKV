import os
import unittest

from unchain_kv.layer_state import LayerStore
from unchain_kv.transport import make_receiver, transport_name


class TransportTest(unittest.TestCase):
    def test_defaults_to_tcp(self):
        old = os.environ.pop("UNCHAIN_KV_TRANSPORT", None)
        receiver = None
        try:
            self.assertEqual(transport_name(), "tcp")
            receiver = make_receiver(("127.0.0.1", 0), LayerStore())
            self.assertEqual(receiver.__class__.__name__, "TcpReceiver")
        finally:
            if receiver is not None:
                receiver.close()
            if old is not None:
                os.environ["UNCHAIN_KV_TRANSPORT"] = old

    def test_selects_tcp(self):
        old = os.environ.get("UNCHAIN_KV_TRANSPORT")
        receiver = None
        try:
            os.environ["UNCHAIN_KV_TRANSPORT"] = "tcp"
            self.assertEqual(transport_name(), "tcp")
            receiver = make_receiver(("127.0.0.1", 0), LayerStore())
            self.assertEqual(receiver.__class__.__name__, "TcpReceiver")
        finally:
            if receiver is not None:
                receiver.close()
            if old is None:
                os.environ.pop("UNCHAIN_KV_TRANSPORT", None)
            else:
                os.environ["UNCHAIN_KV_TRANSPORT"] = old

    def test_rejects_unknown_transport(self):
        old = os.environ.get("UNCHAIN_KV_TRANSPORT")
        try:
            os.environ["UNCHAIN_KV_TRANSPORT"] = "quic"
            with self.assertRaisesRegex(ValueError, "unsupported transport"):
                transport_name()
        finally:
            if old is None:
                os.environ.pop("UNCHAIN_KV_TRANSPORT", None)
            else:
                os.environ["UNCHAIN_KV_TRANSPORT"] = old

if __name__ == "__main__":
    unittest.main()
