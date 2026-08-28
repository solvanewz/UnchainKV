import threading
import time
import unittest
from zlib import crc32

from unchain_kv.layer_state import LayerStore
from unchain_kv.protocol import Chunk, ChunkHeader


def chunk(layer: int, index: int, total: int, payload: bytes = b"x") -> Chunk:
    return Chunk(
        ChunkHeader(
            transfer_id="tx",
            request_id="req",
            layer_index=layer,
            block_index=index,
            chunk_index=index,
            chunks_in_layer=total,
            offset=0,
            payload_len=len(payload),
            checksum=crc32(payload) & 0xFFFFFFFF,
        ),
        payload,
    )


class LayerStoreTest(unittest.TestCase):
    def test_layer_ready_after_out_of_order_chunks(self):
        store = LayerStore()
        self.assertFalse(store.add(chunk(0, 1, 3)))
        self.assertFalse(store.add(chunk(0, 0, 3)))
        self.assertTrue(store.add(chunk(0, 2, 3)))
        self.assertTrue(store.is_ready("tx", 0))

    def test_duplicate_chunk_fails_transfer_and_clears_partial_payload(self):
        store = LayerStore()
        self.assertFalse(store.add(chunk(0, 0, 2)))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            store.add(chunk(0, 0, 2))
        self.assertEqual(store.payload_bytes("tx"), 0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            store.wait("tx", 0, 0)

    def test_wait_can_start_before_first_chunk(self):
        store = LayerStore()
        done = threading.Event()

        def wait():
            store.wait("tx", 0, 1.0)
            done.set()

        thread = threading.Thread(target=wait)
        thread.start()
        time.sleep(0.01)
        store.add(chunk(0, 0, 1))
        thread.join(1.0)
        self.assertTrue(done.is_set())

    def test_observed_session_comes_from_chunk_header(self):
        store = LayerStore()
        store.add(chunk(0, 0, 1))
        self.assertEqual(store.observed_session(), ("tx", "req"))

    def test_layer_layout_is_recorded(self):
        store = LayerStore()
        store.add(chunk(0, 0, 1), layout="kv_major")

        self.assertEqual(store.layer_layout("tx", 0), "kv_major")

    def test_release_payloads_makes_late_reads_and_writes_fail(self):
        store = LayerStore()
        store.add(chunk(0, 0, 1, b"abc"))

        self.assertEqual(store.payload_bytes("tx"), 3)
        self.assertEqual(store.release_payloads("tx", 0), 3)
        self.assertTrue(store.is_ready("tx", 0))
        with self.assertRaisesRegex(RuntimeError, "not available"):
            store.layer_payloads("tx", 0)
        with self.assertRaisesRegex(RuntimeError, "released"):
            store.add(chunk(0, 0, 1))
        self.assertEqual(store.payload_bytes("tx"), 0)

    def test_discard_rejects_late_data(self):
        store = LayerStore()
        store.add(chunk(0, 0, 1, b"abc"))
        self.assertEqual(store.discard_transfer("tx"), 3)
        self.assertFalse(store.is_ready("tx", 0))
        self.assertIsNone(store.observed_session())
        with self.assertRaisesRegex(RuntimeError, "discarded"):
            store.add(chunk(0, 0, 1))

    def test_transfer_id_cannot_cross_requests(self):
        store = LayerStore()
        store.add(chunk(0, 0, 1))
        other = Chunk(
            ChunkHeader("tx", "other", 1, 0, 0, 1, 0, 1, 0), b"x"
        )

        with self.assertRaisesRegex(ValueError, "another request"):
            store.add(other)
        self.assertEqual(store.payload_bytes("tx"), 0)

    def test_chunk_bounds_and_length_are_rejected(self):
        for value, message in (
            (Chunk(ChunkHeader("tx", "req", -1, 0, 0, 1, 0, 1, 0), b"x"), "layer"),
            (Chunk(ChunkHeader("tx", "req", 0, 0, 1, 1, 0, 1, 0), b"x"), "chunk_index"),
            (Chunk(ChunkHeader("tx", "req", 0, 0, 0, 0, 0, 1, 0), b"x"), "positive"),
            (Chunk(ChunkHeader("tx", "req", 0, 0, 0, 1, 0, 2, 0), b"x"), "length"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                LayerStore().add(value)

    def test_timeout_is_terminal_and_rejects_late_data(self):
        store = LayerStore()
        with self.assertRaises(TimeoutError):
            store.wait("tx", 0, 0)
        with self.assertRaises(TimeoutError):
            store.add(chunk(0, 0, 1))

    def test_failure_wakes_waiter_and_releases_partial_payload(self):
        store = LayerStore()
        store.add(chunk(0, 0, 2, b"abc"))
        errors = []

        def wait():
            try:
                store.wait("tx", 0, 1)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=wait)
        thread.start()
        time.sleep(0.01)
        released = store.fail(ConnectionError("truncated"), "tx")
        thread.join(1)

        self.assertEqual(released, 3)
        self.assertEqual(store.payload_bytes("tx"), 0)
        self.assertEqual([str(error) for error in errors], ["truncated"])


if __name__ == "__main__":
    unittest.main()
