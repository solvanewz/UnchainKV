import unittest

from unchain_kv.tensor_blocks import merge_block_bytes, split_block_bytes


class TensorBlocksTest(unittest.TestCase):
    def test_split_and_merge_block_bytes(self):
        data = b"abcdefghijkl"
        chunks = list(split_block_bytes(data, chunk_size=5))
        self.assertEqual(chunks, [(0, b"abcde"), (5, b"fghij"), (10, b"kl")])
        self.assertEqual(merge_block_bytes(len(data), chunks), data)


if __name__ == "__main__":
    unittest.main()
