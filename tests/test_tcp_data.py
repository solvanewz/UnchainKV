import os
import threading
import socket
import time
from types import SimpleNamespace
import unittest
from zlib import crc32

from unchain_kv.grant_state import GrantStore
from unchain_kv.layer_state import LayerStore
from unchain_kv.protocol import Chunk, ChunkHeader
from unchain_kv import tcp_data
from unchain_kv.tcp_data import (
    TcpReceiver,
    send_chunks,
    send_grant,
    send_compressed_native_layer_blocks,
    send_layer_group_blocks,
    send_kv_layer_blocks,
    send_layer_blocks,
    send_native_layer_blocks,
)


def make_chunk(index: int) -> Chunk:
    payload = f"payload-{index}".encode()
    return Chunk(
        ChunkHeader(
            "tx",
            "req",
            0,
            index,
            index,
            3,
            0,
            len(payload),
            crc32(payload) & 0xFFFFFFFF,
        ),
        payload,
    )


class TcpDataTest(unittest.TestCase):
    def test_recv_exact_allows_clean_eof_but_rejects_partial_data(self):
        left, right = socket.socketpair()
        try:
            left.close()
            self.assertIsNone(tcp_data._recv_exact(right, 8, clean_eof=True))
        finally:
            right.close()

        left, right = socket.socketpair()
        try:
            left.sendall(b"half")
            left.close()
            with self.assertRaisesRegex(ConnectionError, "truncated"):
                tcp_data._recv_exact(right, 8, clean_eof=True)
        finally:
            right.close()

    def test_frame_checksum_rejects_corruption(self):
        checksum = crc32(b"payload") & 0xFFFFFFFF

        tcp_data._validate_frame(b"payload", checksum)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            tcp_data._validate_frame(b"payloax", checksum)

    def test_receiver_accepts_out_of_order_chunks(self):
        store = LayerStore()
        receiver = TcpReceiver(("127.0.0.1", 0), store)
        port = receiver.port
        thread = threading.Thread(target=receiver.serve, daemon=True)
        thread.start()

        send_chunks(("127.0.0.1", port), [make_chunk(2), make_chunk(0), make_chunk(1)])
        deadline = time.time() + 2
        while time.time() < deadline and not store.is_ready("tx", 0):
            time.sleep(0.01)
        receiver.close()

        self.assertTrue(store.is_ready("tx", 0))

    def test_receiver_fails_active_transfer_on_truncated_prefix(self):
        store = LayerStore()
        store.add(make_chunk(0))
        events = []
        receiver = TcpReceiver(
            ("127.0.0.1", 0),
            store,
            trace=SimpleNamespace(event=lambda name, **fields: events.append((name, fields))),
        )
        thread = threading.Thread(target=receiver.serve, daemon=True)
        thread.start()
        with socket.create_connection(("127.0.0.1", receiver.port)) as sock:
            sock.sendall(b"half")
        deadline = time.time() + 2
        while time.time() < deadline and store.payload_bytes("tx"):
            time.sleep(0.01)
        receiver.close()

        self.assertEqual(store.payload_bytes("tx"), 0)
        with self.assertRaisesRegex(ConnectionError, "truncated"):
            store.wait("tx", 0, 0)
        self.assertEqual(events[-1][0], "tcp_receive_error")
        self.assertEqual(events[-1][1]["error"], "ConnectionError")

    def test_receiver_accepts_grants(self):
        store = LayerStore()
        grants = GrantStore()
        receiver = TcpReceiver(("127.0.0.1", 0), store, grants=grants)
        port = receiver.port
        thread = threading.Thread(target=receiver.serve, daemon=True)
        thread.start()

        send_grant(("127.0.0.1", port), 7)
        try:
            grants.wait(7, 2)
        finally:
            receiver.close()

    def test_send_grant_waits_for_delayed_receiver(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        store = LayerStore()
        grants = GrantStore()
        receiver = None

        def start_receiver():
            nonlocal receiver
            time.sleep(0.05)
            receiver = TcpReceiver(("127.0.0.1", port), store, grants=grants)
            receiver.serve()

        thread = threading.Thread(target=start_receiver, daemon=True)
        thread.start()
        send_grant(("127.0.0.1", port), 3)
        try:
            grants.wait(3, 2)
        finally:
            if receiver is not None:
                receiver.close()

    def test_bulk_block_layer_is_stored_as_direct_payload_view(self):
        store = LayerStore()
        receiver = TcpReceiver(("127.0.0.1", 0), store)
        thread = threading.Thread(target=receiver.serve, daemon=True)
        thread.start()

        payload = b"abcdefgh"
        send_layer_blocks(("127.0.0.1", receiver.port), "tx", "req", 2, payload, 4, 2)
        deadline = time.time() + 2
        while time.time() < deadline and not store.is_ready("tx", 2):
            time.sleep(0.01)
        receiver.close()

        self.assertEqual(store.layer_layout("tx", 2), "block_major")
        stored = store.layer_payloads("tx", 2)[0]
        self.assertIsInstance(stored, memoryview)
        self.assertIsInstance(stored.obj, bytearray)
        self.assertEqual(bytes(stored), payload)

    def test_bulk_native_and_kv_major_layers(self):
        store = LayerStore()
        receiver = TcpReceiver(("127.0.0.1", 0), store)
        thread = threading.Thread(target=receiver.serve, daemon=True)
        thread.start()

        send_native_layer_blocks(
            ("127.0.0.1", receiver.port), "tx", "req", 3, b"native", 3, 2
        )
        send_kv_layer_blocks(
            ("127.0.0.1", receiver.port), "tx", "req", 4, b"keykey", b"valval", 3, 2
        )
        deadline = time.time() + 2
        while time.time() < deadline and (
            not store.is_ready("tx", 3) or not store.is_ready("tx", 4)
        ):
            time.sleep(0.01)
        receiver.close()

        self.assertEqual(store.layer_layout("tx", 3), "native")
        self.assertEqual(bytes(store.layer_payloads("tx", 3)[0]), b"native")
        self.assertEqual(store.layer_layout("tx", 4), "kv_major")
        self.assertEqual(
            [bytes(payload) for payload in store.layer_payloads("tx", 4)],
            [b"keykey", b"valval"],
        )

    def test_compressed_native_layer_keeps_codec_metadata(self):
        store = LayerStore()
        receiver = object.__new__(TcpReceiver)
        receiver.store = store
        receiver.trace = None
        receiver.grants = None
        frame = bytearray(b"KVC1")
        frame.extend(tcp_data._write_string("tx"))
        frame.extend(tcp_data._write_string("req"))
        frame.extend(tcp_data._U32.pack(5))
        frame.extend(tcp_data._U32.pack(2))
        frame.extend(tcp_data._U32.pack(3))
        frame.extend(tcp_data._U32.pack(6))
        frame.extend(tcp_data._U32.pack(tcp_data._CODEC_CODES["raw_passthrough"]))
        frame.extend(tcp_data._U32.pack(6))
        frame.extend(b"native")

        receiver._handle_frame(frame)

        self.assertEqual(store.layer_layout("tx", 5), "compressed_native")
        self.assertEqual(bytes(store.layer_payloads("tx", 5)[0]), b"native")
        self.assertEqual(
            store.layer_metadata("tx", 5),
            {
                "block_count": 2,
                "raw_block_size": 3,
                "raw_bytes": 6,
                "encoded_bytes": 6,
                "codec": "raw_passthrough",
            },
        )

    def test_compressed_native_two_chunk_layer_waits_for_both_frames(self):
        magic = getattr(tcp_data, "_COMPRESSED_NATIVE_CHUNK_MAGIC", None)
        self.assertIsNotNone(magic)
        if magic is None:
            return
        store = LayerStore()
        receiver = object.__new__(TcpReceiver)
        receiver.store = store
        receiver.trace = None
        receiver.grants = None

        for chunk_index, payload in enumerate((b"zip", b"zap")):
            frame = bytearray(magic)
            frame.extend(tcp_data._write_string("tx"))
            frame.extend(tcp_data._write_string("req"))
            for value in (
                5,
                4,
                3,
                12,
                tcp_data._CODEC_CODES["splitzip_bf16"],
                chunk_index,
                2,
                len(payload),
            ):
                frame.extend(tcp_data._U32.pack(value))
            receiver._handle_frame(frame + payload)
            self.assertEqual(store.is_ready("tx", 5), chunk_index == 1)

        self.assertEqual(
            [bytes(payload) for payload in store.layer_payloads("tx", 5)],
            [b"zip", b"zap"],
        )
        self.assertEqual(
            store.layer_metadata("tx", 5),
            {
                "block_count": 4,
                "raw_block_size": 3,
                "raw_bytes": 12,
                "codec": "splitzip_bf16",
                "chunks_in_layer": 2,
            },
        )

    def test_layer_group_frame_splits_payloads_by_layer(self):
        store = LayerStore()
        receiver = TcpReceiver(("127.0.0.1", 0), store)
        thread = threading.Thread(target=receiver.serve, daemon=True)
        thread.start()

        send_layer_group_blocks(
            ("127.0.0.1", receiver.port),
            "tx",
            "req",
            2,
            [b"abcd", b"efgh"],
            2,
            2,
            layout="block_major",
        )
        deadline = time.time() + 2
        while time.time() < deadline and (
            not store.is_ready("tx", 2) or not store.is_ready("tx", 3)
        ):
            time.sleep(0.01)
        receiver.close()

        self.assertEqual(bytes(store.layer_payloads("tx", 2)[0]), b"abcd")
        self.assertEqual(bytes(store.layer_payloads("tx", 3)[0]), b"efgh")
        self.assertEqual(store.layer_layout("tx", 2), "block_major")

    def test_connect_uses_separate_io_timeout_after_connect(self):
        old = os.environ.get("UNCHAIN_KV_TCP_IO_TIMEOUT_S")
        store = LayerStore()
        receiver = TcpReceiver(("127.0.0.1", 0), store)
        thread = threading.Thread(target=receiver.serve, daemon=True)
        thread.start()
        sock = None
        try:
            os.environ["UNCHAIN_KV_TCP_IO_TIMEOUT_S"] = "7.5"
            sock = tcp_data._connect(("127.0.0.1", receiver.port))
            self.assertEqual(sock.gettimeout(), 7.5)
        finally:
            if sock is not None:
                sock.close()
            receiver.close()
            if old is None:
                os.environ.pop("UNCHAIN_KV_TCP_IO_TIMEOUT_S", None)
            else:
                os.environ["UNCHAIN_KV_TCP_IO_TIMEOUT_S"] = old


if __name__ == "__main__":
    unittest.main()
