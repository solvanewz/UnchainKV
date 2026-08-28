from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from unchain_kv.patch_vllm import (
    patch_attention_kv_update,
    patch_factory,
    patch_kv_transfer_utils,
    patch_rope_kvcache_fusion,
    write_connector,
)
import os
from unittest.mock import patch


class PatchVllmTest(unittest.TestCase):
    def test_writes_connector_and_patches_factory_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            factory = root / "vllm/distributed/kv_transfer/kv_connector/factory.py"
            factory.parent.mkdir(parents=True)
            factory.write_text("class KVConnectorFactory:\n    pass\n", encoding="utf-8")

            connector = write_connector(root)
            patch_factory(root)
            once = factory.read_text(encoding="utf-8")
            patch_factory(root)

            self.assertTrue(connector.exists())
            self.assertIn("class UnchainKVConnector", connector.read_text("utf-8"))
            self.assertEqual(once, factory.read_text(encoding="utf-8"))
            self.assertEqual(
                once.count('KVConnectorFactory.register_connector("UnchainKVConnector"'),
                1,
            )

    def test_patches_kv_transfer_utils_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "vllm/model_executor/layers/attention/kv_transfer_utils.py"
            target.parent.mkdir(parents=True)
            target.write_text(
                "def wrapper(*args, **kwargs):\n"
                "        # Execute the function\n"
                "        result = func(*args, **kwargs)\n"
                "        # Save KV cache layer on exit\n"
                "        connector.save_kv_layer(layer_name, kv_cache, attn_metadata)\n"
                "        return result\n",
                encoding="utf-8",
            )

            patch_kv_transfer_utils(root)
            once = target.read_text(encoding="utf-8")
            patch_kv_transfer_utils(root)

            self.assertEqual(once, target.read_text(encoding="utf-8"))
            self.assertIn("trace_prefill_attention_start", once)
            self.assertIn("trace_prefill_attention_done", once)

    def test_patches_kv_update_trace_points_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attention = root / "vllm/model_executor/layers/attention/attention.py"
            fusion = root / "vllm/compilation/passes/fusion/rope_kvcache_fusion.py"
            attention.parent.mkdir(parents=True)
            fusion.parent.mkdir(parents=True)
            attention.write_text(
                "        attn_layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]\n"
                "            attn_layer,\n"
                "            key,\n"
                "            value,\n"
                "            kv_cache,\n"
                "            layer_slot_mapping,\n"
                "        )\n",
                encoding="utf-8",
            )
            fusion.write_text(
                "        attn_layer.impl.do_rope_and_kv_cache_update(\n"
                "            attn_layer,\n"
                "            query,\n"
                "            key,\n"
                "            value,\n"
                "            positions,\n"
                "            cos_sin_cache,\n"
                "            is_neox,\n"
                "            kv_cache,\n"
                "            layer_slot_mapping,\n"
                "        )\n",
                encoding="utf-8",
            )

            patch_attention_kv_update(root)
            patch_rope_kvcache_fusion(root)
            attention_once = attention.read_text(encoding="utf-8")
            fusion_once = fusion.read_text(encoding="utf-8")
            patch_attention_kv_update(root)
            patch_rope_kvcache_fusion(root)

            self.assertEqual(attention_once, attention.read_text(encoding="utf-8"))
            self.assertEqual(fusion_once, fusion.read_text(encoding="utf-8"))
            self.assertIn("trace_prefill_kv_update_done", attention_once)
            self.assertIn("trace_prefill_kv_update_done", fusion_once)



    # ── Extent-order helpers ──

    def test_extent_order_defaults_off(self):
        from unchain_kv.patch_vllm import extent_order_enabled
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(extent_order_enabled())
        with patch.dict(os.environ, {"UNCHAIN_KV_EXTENT_ALLOC": "off"}, clear=True):
            self.assertFalse(extent_order_enabled())
        with patch.dict(os.environ, {"UNCHAIN_KV_EXTENT_ALLOC": ""}, clear=True):
            self.assertFalse(extent_order_enabled())

    def test_extent_order_accepts_normalize(self):
        from unchain_kv.patch_vllm import extent_order_enabled
        with patch.dict(os.environ, {"UNCHAIN_KV_EXTENT_ALLOC": "normalize"}, clear=True):
            self.assertTrue(extent_order_enabled())

    def test_extent_order_accepts_prefer(self):
        from unchain_kv.patch_vllm import extent_alloc_mode, extent_order_enabled
        with patch.dict(os.environ, {"UNCHAIN_KV_EXTENT_ALLOC": "prefer"}, clear=True):
            self.assertEqual(extent_alloc_mode(), "prefer")
            self.assertTrue(extent_order_enabled())

    def test_extent_order_rejects_unknown_mode(self):
        from unchain_kv.patch_vllm import extent_order_enabled
        with patch.dict(os.environ, {"UNCHAIN_KV_EXTENT_ALLOC": "compact"}, clear=True):
            with self.assertRaises(ValueError):
                extent_order_enabled()

    def test_prefers_adjacent_then_first_full_extent(self):
        from unchain_kv.patch_vllm import FakeBlock, select_preferred_extent

        class Queue:
            def remove(self, block):
                block.prev_free_block = block.next_free_block = None

            def append_n(self, blocks):
                for block in blocks:
                    block.prev_free_block = block.next_free_block = object()

        blocks = [FakeBlock(i, ref_cnt=0) for i in range(10)]
        pool = SimpleNamespace(
            blocks=blocks,
            free_block_queue=Queue(),
            unchain_kv_extent_alloc="prefer",
        )
        adjacent = select_preferred_extent(pool, 3, preferred_after=3)
        self.assertEqual([block.block_id for block in adjacent], [4, 5, 6])

        blocks[0].is_null = True
        blocks[1].block_hash = 1
        blocks[2].ref_cnt = 1
        first_fit = select_preferred_extent(pool, 2, preferred_after=8)
        self.assertEqual([block.block_id for block in first_fit], [7, 8])

    def test_extent_miss_preserves_free_blocks(self):
        from unchain_kv.patch_vllm import FakeBlock, select_preferred_extent

        blocks = [FakeBlock(i, block_hash=i) for i in range(4)]
        queue = SimpleNamespace(
            remove=lambda _block: self.fail("extent miss must not remove"),
            append_n=lambda _blocks: None,
        )
        pool = SimpleNamespace(
            blocks=blocks,
            free_block_queue=queue,
            unchain_kv_extent_alloc="prefer",
        )
        self.assertIsNone(select_preferred_extent(pool, 2))

    def test_short_codec_fast_path_uses_normalized_queue_allocation(self):
        from unchain_kv.patch_vllm import FakeBlock, select_preferred_extent

        pool = SimpleNamespace(
            blocks=[FakeBlock(i, ref_cnt=0) for i in range(8)],
            free_block_queue=SimpleNamespace(
                remove=lambda _block: self.fail("short path must use free queue")
            ),
            unchain_kv_extent_alloc="prefer",
        )
        with patch.dict(
            os.environ, {"UNCHAIN_KV_CODEC_MIN_BLOCKS": "4"}, clear=False
        ):
            self.assertIsNone(select_preferred_extent(pool, 3))

    def test_reserved_extent_activates_and_releases_without_cache_eviction(self):
        from unchain_kv.patch_vllm import (
            FakeBlock,
            activate_reserved_blocks,
            release_reserved_blocks,
            select_preferred_extent,
        )

        class Queue:
            def remove(self, block):
                block.prev_free_block = block.next_free_block = None

            def append_n(self, blocks):
                for block in blocks:
                    block.prev_free_block = block.next_free_block = object()

        blocks = [FakeBlock(i, ref_cnt=0) for i in range(6)]
        pool = SimpleNamespace(
            blocks=blocks,
            free_block_queue=Queue(),
            unchain_kv_extent_alloc="prefer",
            metrics_collector=None,
        )
        reserved = select_preferred_extent(pool, 2, preferred_after=1)
        activate_reserved_blocks(pool, reserved)
        self.assertEqual([block.ref_cnt for block in reserved], [1, 1])
        for block in reserved:
            block.ref_cnt = 0
        release_reserved_blocks(pool, reserved)
        self.assertTrue(all(block.prev_free_block is not None for block in reserved))

    def test_extent_lease_evicts_idle_cache_to_form_largest_run(self):
        from unchain_kv.patch_vllm import FakeBlock, select_extent_lease

        class Queue:
            def remove(self, block):
                block.prev_free_block = block.next_free_block = None

        blocks = [FakeBlock(i, ref_cnt=0) for i in range(12)]
        blocks[3].block_hash = 1
        blocks[9].ref_cnt = 1

        def evict(block_ids):
            for block_id in block_ids:
                blocks[block_id].block_hash = None

        pool = SimpleNamespace(
            blocks=blocks,
            free_block_queue=Queue(),
            unchain_kv_extent_alloc="prefer",
            evict_blocks=evict,
        )

        lease = select_extent_lease(pool, 2, 8)

        self.assertEqual([block.block_id for block in lease], list(range(8)))
        self.assertIsNone(blocks[3].block_hash)

    def test_normalizes_allocated_blocks(self):
        from unchain_kv.patch_vllm import normalize_allocated_blocks, FakeBlock
        blocks = [FakeBlock(9), FakeBlock(2), FakeBlock(5)]
        result = normalize_allocated_blocks(blocks)
        self.assertIs(result, blocks)
        self.assertEqual([b.block_id for b in blocks], [2, 5, 9])

    def test_normalize_release_is_opt_in(self):
        from unchain_kv.patch_vllm import normalize_release_enabled
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(normalize_release_enabled())
        with patch.dict(os.environ, {"UNCHAIN_KV_NORMALIZE_RELEASE": "1"}):
            self.assertTrue(normalize_release_enabled())

    def test_normalizes_only_recyclable_blocks(self):
        from unchain_kv.patch_vllm import normalize_released_blocks, FakeBlock
        blocks = [
            FakeBlock(9), FakeBlock(8, block_hash=0xCAFE),
            FakeBlock(2), FakeBlock(7, ref_cnt=2), FakeBlock(0, is_null=True),
        ]
        normalize_released_blocks(blocks)
        self.assertEqual([b.block_id for b in blocks], [2, 8, 9, 7, 0])
        self.assertIsNone(blocks[0].block_hash)
        self.assertEqual(blocks[1].block_hash, 0xCAFE)
        self.assertEqual(blocks[3].ref_cnt, 2)
        self.assertTrue(blocks[4].is_null)

    def test_normalizes_released_from_reversed_input(self):
        from unchain_kv.patch_vllm import normalize_released_blocks, FakeBlock
        blocks = [FakeBlock(9), FakeBlock(5), FakeBlock(2)]
        normalize_released_blocks(blocks)
        self.assertEqual([b.block_id for b in blocks], [2, 5, 9])

    def test_normalize_released_returns_materialized_list(self):
        from unchain_kv.patch_vllm import normalize_released_blocks, FakeBlock
        blocks = [FakeBlock(i) for i in range(5)]
        result = normalize_released_blocks(iter(reversed(blocks)))
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)

    # ── Task 2: patch_extent_order ──

    def _vllm_fixtures(self, root):
        bp = root / "vllm/v1/core/block_pool.py"
        mgr = root / "vllm/v1/core/single_type_kv_cache_manager.py"
        bp.parent.mkdir(parents=True, exist_ok=True)
        mgr.parent.mkdir(parents=True, exist_ok=True)
        bp.write_text("""from collections.abc import Iterable, Sequence
from typing import Any

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock, FreeKVCacheBlockQueue

logger = init_logger(__name__)

class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching, hash_block_size):
        self.blocks = [KVCacheBlock(i) for i in range(num_gpu_blocks)]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)
        for block in ret:
            block.ref_cnt += 1
        return ret

    def free_blocks(self, ordered_blocks):
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n(
            [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
        )
""", encoding="utf-8")
        mgr.write_text("""import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock

class SingleTypeKVCacheManager:
    def __init__(self, block_pool):
        self.block_pool = block_pool
        self.new_block_ids: list[int] = []

    def get_num_blocks_to_allocate(self):
        num_required_blocks = 0
        num_req_blocks = 0
        if request_id in self.num_cached_block:
            return max(num_required_blocks - num_req_blocks, 0)
        num_new_blocks = 0
        num_evictable_blocks = 0
        return num_new_blocks + num_evictable_blocks

    def allocate_new_computed_blocks(self):
        req_blocks = self.req_to_blocks[request_id]
        if num_external_computed_tokens > 0:
            allocated_blocks = self.block_pool.get_new_blocks(
                cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
            )
            req_blocks.extend(allocated_blocks)

    def allocate_new_blocks(self):
        req_blocks = self.req_to_blocks[request_id]
        if num_new_blocks > 0:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)

    def unrelated_nested_allocation(self):
        if num_new_blocks > 0:
                new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
                req_blocks.extend(new_blocks)

    def free(self, request_id: str) -> None:
        req_blocks = self.req_to_blocks.pop(request_id, [])
        ordered_blocks = reversed(req_blocks)
        self.block_pool.free_blocks(ordered_blocks)
""", encoding="utf-8")

    def test_patches_extent_order_once(self):
        from unchain_kv.patch_vllm import patch_extent_order
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._vllm_fixtures(root)
            paths = patch_extent_order(root)
            self.assertEqual(len(paths), 2)
            bp_text = (root / "vllm/v1/core/block_pool.py").read_text("utf-8")
            mgr_text = (root / "vllm/v1/core/single_type_kv_cache_manager.py").read_text("utf-8")
            paths2 = patch_extent_order(root)
            self.assertEqual(bp_text, (root / "vllm/v1/core/block_pool.py").read_text("utf-8"))
            self.assertEqual(mgr_text, (root / "vllm/v1/core/single_type_kv_cache_manager.py").read_text("utf-8"))
            self.assertIn("from unchain_kv.patch_vllm import", bp_text)
            self.assertIn("unchain_kv_extent_alloc", bp_text)
            self.assertIn("select_preferred_extent", bp_text)
            self.assertIn("preferred_after: int | None", bp_text)
            self.assertIn("normalize_allocated_blocks(ret)", bp_text)
            self.assertIn("from unchain_kv.patch_vllm import", mgr_text)
            self.assertIn("normalize_released_blocks(ordered_blocks)", mgr_text)
            self.assertIn("or normalize_release_enabled()", mgr_text)
            self.assertEqual(mgr_text.count("preferred_after=preferred_after"), 2)
            self.assertIn("_unchain_kv_extent_reservations", mgr_text)
            self.assertIn("activate_reserved_blocks", mgr_text)
            self.assertIn("release_reserved_blocks", mgr_text)
            self.assertIn(
                "                new_blocks = self.block_pool.get_new_blocks(num_new_blocks)",
                mgr_text,
            )
            self.assertLess(
                mgr_text.index("ordered_blocks = reversed(req_blocks)"),
                mgr_text.index("normalize_released_blocks(ordered_blocks)"),
            )
            self.assertLess(
                mgr_text.index("        release_reserved_blocks("),
                mgr_text.index("        self.block_pool.free_blocks(ordered_blocks)"),
            )
            compile(bp_text, "block_pool.py", "exec")
            compile(mgr_text, "single_type_kv_cache_manager.py", "exec")

    def test_extent_patch_preflights_both_files_before_writing(self):
        from unchain_kv.patch_vllm import patch_extent_order
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bp = root / "vllm/v1/core/block_pool.py"
            mgr = root / "vllm/v1/core/single_type_kv_cache_manager.py"
            bp.parent.mkdir(parents=True)
            bp.write_text("class BlockPool:\n    pass\n", encoding="utf-8")
            mgr.write_text(
                "class SingleTypeKVCacheManager:\n"
                "    def free(self, request_id):\n"
                "        req_blocks = []\n"
                "        ordered_blocks = reversed(req_blocks)\n"
                "        self.block_pool.free_blocks(ordered_blocks)\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                patch_extent_order(root)
            self.assertIn("class BlockPool:", bp.read_text("utf-8"))

    def test_extent_patch_rejects_partial_patch_state(self):
        from unchain_kv.patch_vllm import patch_extent_order
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._vllm_fixtures(root)
            patch_extent_order(root)
            mgr = root / "vllm/v1/core/single_type_kv_cache_manager.py"
            mgr.write_text("broken", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                patch_extent_order(root)

if __name__ == "__main__":
    unittest.main()
