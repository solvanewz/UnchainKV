import unittest

from unchain_kv.grant_state import GrantStore


class GrantStoreTest(unittest.TestCase):
    def test_grant_is_scoped_to_transfer(self):
        store = GrantStore()
        store.add(2, transfer_id="tx-a")

        store.wait(2, 0, transfer_id="tx-a")
        with self.assertRaises(TimeoutError):
            store.wait(2, 0, transfer_id="tx-b")

    def test_wait_distinguishes_kind_and_transfer(self):
        store = GrantStore()
        store.add(2)
        store.add(2, kind="restore_ack", transfer_id="tx-a")

        store.wait(2, 0, kind="grant")
        store.wait(2, 0, kind="restore_ack", transfer_id="tx-a")
        with self.assertRaises(TimeoutError):
            store.wait(2, 0, kind="restore_ack", transfer_id="tx-b")

    def test_wait_value_returns_and_consumes_matching_value(self):
        store = GrantStore()
        store.add(34, kind="prefix_tokens", transfer_id="tx")

        self.assertEqual(store.wait_value(0, "prefix_tokens", "tx"), 34)
        with self.assertRaises(TimeoutError):
            store.wait_value(0, "prefix_tokens", "tx")


if __name__ == "__main__":
    unittest.main()
