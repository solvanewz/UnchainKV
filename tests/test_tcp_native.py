import ctypes
import gc
import os
import socket
import threading
import time
import unittest
from unittest.mock import patch

from unchain_kv import tcp_data, tcp_native
from unchain_kv.layer_state import LayerStore
from unchain_kv.protocol import Chunk, ChunkHeader


class FakeLib:
    def __init__(self):
        self.calls = []
        self.freed = []

    def kvq_tcp_send_block_layer(
        self, peer, transfer, request, layer, data, data_len, block_size, block_count
    ):
        self.calls.append(
            (
                peer.decode(),
                transfer.decode(),
                request.decode(),
                layer,
                ctypes.string_at(data, data_len),
                block_size,
                block_count,
            )
        )
        return 0

    def kvq_tcp_stop_receiver(self):
        return 0

    def kvq_tcp_free(self, ptr):
        self.freed.append(ptr.value)

    def kvq_tcp_last_error(self):
        return b"fake error"


class FakePersistentLib(FakeLib):
    def __init__(self):
        super().__init__()
        self.connects = 0
        self.fd_calls = []

    def kvq_tcp_connect(self, peer):
        self.connects += 1
        self.peer = peer.decode()
        return 42

    def kvq_tcp_send_block_layer_fd(
        self, fd, transfer, request, layer, data, data_len, block_size, block_count
    ):
        self.fd_calls.append(
            (
                fd,
                transfer.decode(),
                request.decode(),
                layer,
                ctypes.string_at(data, data_len),
                block_size,
                block_count,
            )
        )
        return 0


class TcpNativeTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("UNCHAIN_KV_TCP_NATIVE_INTEGRATION"),
        "requires the built native TCP library",
    )
    def test_built_native_library_round_trip(self):
        store = LayerStore()
        receiver = tcp_native.NativeTcpReceiver(("127.0.0.1", 29799), store)
        thread = threading.Thread(target=receiver.serve)
        thread.start()
        try:
            for _ in range(100):
                try:
                    with socket.create_connection(("127.0.0.1", 29799), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.01)
            else:
                self.fail("native receiver did not start")
            tcp_data.send_layer_blocks(
                ("127.0.0.1", 29799), "tx", "req", 0, b"abcdefgh", 4, 2
            )
            store.wait("tx", 0, 2.0)
            self.assertEqual(bytes(store.layer_payloads("tx", 0)[0]), b"abcdefgh")
        finally:
            tcp_native._drop_connection(receiver.lib, ("127.0.0.1", 29799))
            receiver.close()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())

    def test_callback_error_fails_active_transfer_without_escaping(self):
        fake = FakeLib()
        store = LayerStore()
        store.add(Chunk(ChunkHeader("tx", "req", 0, 0, 0, 2, 0, 1, 0), b"x"))
        with patch("unchain_kv.tcp_native.load_library", return_value=fake):
            receiver = tcp_native.NativeTcpReceiver(("127.0.0.1", 1234), store)

        receiver._on_frame(None, 0, None)

        self.assertEqual(store.payload_bytes("tx"), 0)
        with self.assertRaisesRegex(ValueError, "null frame"):
            store.wait("tx", 0, 0)

    def test_tcp_data_uses_native_sender_when_configured(self):
        old = os.environ.get("UNCHAIN_KV_TCP_LIB")
        fake = FakeLib()
        try:
            os.environ["UNCHAIN_KV_TCP_LIB"] = "/tmp/libunchain_kv_tcp.so"
            with patch("unchain_kv.tcp_native.load_library", return_value=fake):
                tcp_data.send_layer_blocks(
                    ("127.0.0.1", 1234), "tx", "req", 2, b"abcdefgh", 4, 2
                )

            self.assertEqual(
                fake.calls,
                [("127.0.0.1:1234", "tx", "req", 2, b"abcdefgh", 4, 2)],
            )
        finally:
            if old is None:
                os.environ.pop("UNCHAIN_KV_TCP_LIB", None)
            else:
                os.environ["UNCHAIN_KV_TCP_LIB"] = old

    def test_native_sender_reuses_thread_local_connection(self):
        old = os.environ.get("UNCHAIN_KV_TCP_LIB")
        fake = FakePersistentLib()
        tcp_native._tls.fds = {}
        try:
            os.environ["UNCHAIN_KV_TCP_LIB"] = "/tmp/libunchain_kv_tcp.so"
            with patch("unchain_kv.tcp_native.load_library", return_value=fake):
                for layer in (2, 3):
                    tcp_data.send_layer_blocks(
                        ("127.0.0.1", 1234), "tx", "req", layer, b"abcdefgh", 4, 2
                    )

            self.assertEqual(fake.connects, 1)
            self.assertEqual(fake.peer, "127.0.0.1:1234")
            self.assertEqual([call[3] for call in fake.fd_calls], [2, 3])
        finally:
            tcp_native._tls.fds = {}
            if old is None:
                os.environ.pop("UNCHAIN_KV_TCP_LIB", None)
            else:
                os.environ["UNCHAIN_KV_TCP_LIB"] = old

    def test_native_receiver_stores_callback_payload_without_bytearray_copy(self):
        fake = FakeLib()
        store = LayerStore()
        with patch("unchain_kv.tcp_native.load_library", return_value=fake):
            receiver = tcp_native.NativeTcpReceiver(("127.0.0.1", 1234), store)

        frame = bytearray(b"KVB1")
        frame.extend((2).to_bytes(4, "big"))
        frame.extend(b"tx")
        frame.extend((3).to_bytes(4, "big"))
        frame.extend(b"req")
        frame.extend((7).to_bytes(4, "big"))
        frame.extend((2).to_bytes(4, "big"))
        frame.extend((4).to_bytes(4, "big"))
        frame.extend(b"abcdefgh")
        buf = ctypes.create_string_buffer(bytes(frame))

        receiver._on_frame(ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte)), len(frame), None)

        payload = store.layer_payloads("tx", 7)[0]
        self.assertEqual(store.layer_layout("tx", 7), "block_major")
        self.assertIsInstance(payload, memoryview)
        self.assertNotIsInstance(payload.obj, bytearray)
        self.assertEqual(bytes(payload), b"abcdefgh")
        self.assertEqual(fake.freed, [])

        store.release_payloads("tx", 7)
        self.assertEqual(fake.freed, [])
        del payload
        gc.collect()
        self.assertEqual(fake.freed, [ctypes.addressof(buf)])


if __name__ == "__main__":
    unittest.main()
