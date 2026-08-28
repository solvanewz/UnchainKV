import unittest
from zlib import crc32

from unchain_kv.protocol import Chunk, ChunkHeader, Grant, HEADER_SIZE


class ProtocolTest(unittest.TestCase):
    def test_chunk_round_trips(self):
        payload = b"abc"
        header = ChunkHeader(
            transfer_id="tx1",
            request_id="req1",
            layer_index=2,
            block_index=3,
            chunk_index=4,
            chunks_in_layer=5,
            offset=6,
            payload_len=len(payload),
            checksum=crc32(payload) & 0xFFFFFFFF,
        )
        raw = Chunk(header, payload).pack()
        self.assertGreater(len(raw), HEADER_SIZE)
        parsed = Chunk.unpack(raw)
        self.assertEqual(parsed.header, header)
        self.assertEqual(parsed.payload, payload)

    def test_crc_mismatch_raises(self):
        payload = b"abc"
        header = ChunkHeader("tx1", "req1", 0, 0, 0, 1, 0, len(payload), 0)
        raw = Chunk(header, payload).pack()
        with self.assertRaises(ValueError):
            Chunk.unpack(raw)

    def test_grant_round_trips(self):
        raw = Grant(layer_index=3).pack()

        self.assertTrue(Grant.is_grant(raw))
        self.assertEqual(Grant.unpack(raw), Grant(layer_index=3))

    def test_typed_grant_round_trips(self):
        grant = Grant(layer_index=3, kind="restore_ack", transfer_id="tx")

        self.assertEqual(Grant.unpack(grant.pack()), grant)


if __name__ == "__main__":
    unittest.main()
