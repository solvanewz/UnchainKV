from __future__ import annotations

import os

# ── Extent-order helpers ──

class FakeBlock:
    """Minimal fake for testing extent order helpers without vLLM imports."""
    __slots__ = (
        "block_id",
        "block_hash",
        "ref_cnt",
        "is_null",
        "prev_free_block",
        "next_free_block",
    )

    def __init__(self, block_id, *, block_hash=None, ref_cnt=1, is_null=False):
        self.block_id = block_id
        self.block_hash = block_hash
        self.ref_cnt = ref_cnt
        self.is_null = is_null
        self.prev_free_block = object()
        self.next_free_block = object()


def extent_alloc_mode() -> str:
    mode = os.environ.get("UNCHAIN_KV_EXTENT_ALLOC", "off").strip() or "off"
    if mode not in {"off", "normalize", "prefer"}:
        raise ValueError(
            f"UNCHAIN_KV_EXTENT_ALLOC={mode!r} not supported; "
            "expected 'off', 'normalize', or 'prefer'"
        )
    return mode


def extent_order_enabled() -> bool:
    return extent_alloc_mode() != "off"


def normalize_release_enabled() -> bool:
    return os.environ.get("UNCHAIN_KV_NORMALIZE_RELEASE", "0") == "1"


def normalize_allocated_blocks(blocks: list) -> list:
    """Sort blocks in-place by block_id ascending; return the same list object."""
    blocks.sort(key=lambda b: b.block_id)
    return blocks


def select_preferred_extent(block_pool, num_blocks: int, preferred_after=None):
    """Remove and return one free uncached physical extent, or None."""
    if getattr(block_pool, "unchain_kv_extent_alloc", "off") != "prefer":
        return None
    if num_blocks <= 0:
        return []
    prefer_min_blocks = max(
        0, int(os.environ.get("UNCHAIN_KV_CODEC_MIN_BLOCKS", "0") or 0)
    )
    if num_blocks < prefer_min_blocks:
        return None

    blocks = block_pool.blocks

    def eligible(block) -> bool:
        return (
            not block.is_null
            and block.block_hash is None
            and block.ref_cnt == 0
            and block.prev_free_block is not None
            and block.next_free_block is not None
        )

    selected = None
    if preferred_after is not None:
        start = int(preferred_after) + 1
        stop = start + num_blocks
        if 0 <= start and stop <= len(blocks):
            candidate = blocks[start:stop]
            if len(candidate) == num_blocks and all(map(eligible, candidate)):
                selected = candidate

    if selected is None:
        run_start = 0
        run_length = 0
        for index, block in enumerate(blocks):
            if eligible(block):
                if run_length == 0:
                    run_start = index
                run_length += 1
                if run_length == num_blocks:
                    selected = blocks[run_start : index + 1]
                    break
            else:
                run_length = 0

    if selected is None:
        return None

    removed = []
    try:
        for block in selected:
            block_pool.free_block_queue.remove(block)
            removed.append(block)
    except BaseException:
        block_pool.free_block_queue.append_n(removed)
        raise
    return selected


def select_extent_lease(
    block_pool, min_blocks: int, max_blocks: int, preferred_after=None
):
    """Lease the largest free physical extent, evicting idle cache entries."""
    if getattr(block_pool, "unchain_kv_extent_alloc", "off") != "prefer":
        return None
    if min_blocks <= 0 or max_blocks < min_blocks:
        return None
    blocks = block_pool.blocks

    def eligible(block) -> bool:
        return (
            not block.is_null
            and block.ref_cnt == 0
            and block.prev_free_block is not None
            and block.next_free_block is not None
        )

    candidates = []
    preferred = None
    if preferred_after is not None:
        start = int(preferred_after) + 1
        length = 0
        while start + length < len(blocks) and eligible(blocks[start + length]):
            length += 1
        if length >= min_blocks:
            preferred = (length, start)

    run_start = 0
    run_length = 0
    for index, block in enumerate(blocks + [None]):
        if index < len(blocks) and eligible(block):
            if run_length == 0:
                run_start = index
            run_length += 1
            continue
        if run_length >= min_blocks:
            candidates.append((run_length, run_start))
        run_length = 0
    if preferred is None and not candidates:
        return None
    length, start = preferred or max(
        candidates, key=lambda item: (item[0], -item[1])
    )
    selected = blocks[start : start + min(length, max_blocks)]
    removed = []
    try:
        for block in selected:
            block_pool.free_block_queue.remove(block)
            removed.append(block)
    except BaseException:
        block_pool.free_block_queue.append_n(removed)
        raise
    cached_ids = {block.block_id for block in selected if block.block_hash is not None}
    if cached_ids:
        try:
            block_pool.evict_blocks(cached_ids)
        except BaseException:
            block_pool.free_block_queue.append_n(selected)
            raise
    return selected


def extent_reserve_blocks() -> int:
    return max(0, int(os.environ.get("UNCHAIN_KV_EXTENT_RESERVE_BLOCKS", "0") or 0))


def activate_reserved_blocks(block_pool, blocks: list) -> None:
    for block in blocks:
        if block.ref_cnt != 0 or block.block_hash is not None or block.is_null:
            raise RuntimeError("invalid reserved KV block")
        block.ref_cnt = 1
        if block_pool.metrics_collector:
            block_pool.metrics_collector.on_block_allocated(block)


def release_reserved_blocks(block_pool, blocks: list) -> None:
    if not blocks:
        return
    if any(block.ref_cnt != 0 or block.block_hash is not None for block in blocks):
        raise RuntimeError("cannot release active or cached KV reservation")
    block_pool.free_block_queue.append_n(blocks)


# ── Extent-order source patching ──

_EXTENT_MARKER = "unchain_kv_extent_alloc"

_BLOCK_POOL_IMPORT = (
    "from vllm.logger import init_logger"
)
_BLOCK_POOL_IMPORT_INSERT = (
    "from unchain_kv.patch_vllm import (extent_alloc_mode, "
    "normalize_allocated_blocks, select_preferred_extent)"
)
_BLOCK_POOL_INIT_ANCHOR = (
    "self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)"
)
_BLOCK_POOL_INIT_INSERT = (
    "        self.unchain_kv_extent_alloc = extent_alloc_mode()\n"
)
_BLOCK_POOL_SIGNATURE_ANCHOR = (
    "    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:"
)
_BLOCK_POOL_SIGNATURE_INSERT = (
    "    def get_new_blocks(\n"
    "        self, num_blocks: int, preferred_after: int | None = None\n"
    "    ) -> list[KVCacheBlock]:"
)
_BLOCK_POOL_POP_ANCHOR = (
    "        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)"
)
_BLOCK_POOL_POP_INSERT = (
    "        ret = select_preferred_extent(self, num_blocks, preferred_after)\n"
    "        if ret is None:\n"
    "            ret = self.free_block_queue.popleft_n(num_blocks)"
)
_BLOCK_POOL_RETURN_ANCHOR = (
    "        return ret"
)
_BLOCK_POOL_RETURN_INSERT = (
    "        if self.unchain_kv_extent_alloc != \"off\":\n"
    "            normalize_allocated_blocks(ret)\n"
)

_MANAGER_IMPORT_ANCHOR = (
    "from vllm.v1.core.block_pool import"
)
_MANAGER_IMPORT_INSERT = (
    "from unchain_kv.patch_vllm import (activate_reserved_blocks, "
    "extent_reserve_blocks, normalize_release_enabled, normalize_released_blocks, "
    "release_reserved_blocks, select_extent_lease, select_preferred_extent)"
)
_MANAGER_INIT_ANCHOR = "        self.new_block_ids: list[int] = []"
_MANAGER_INIT_INSERT = (
    "        self.unchain_kv_extent_reserve_blocks = extent_reserve_blocks()\n"
    "        self._unchain_kv_extent_reservations = {}\n"
)
_MANAGER_FAST_COUNT_ANCHOR = (
    "            return max(num_required_blocks - num_req_blocks, 0)"
)
_MANAGER_FAST_COUNT_INSERT = (
    "            reserved = len(\n"
    "                self._unchain_kv_extent_reservations.get(request_id, ())\n"
    "            )\n"
    "            return max(num_required_blocks - num_req_blocks - reserved, 0)"
)
_MANAGER_COUNT_ANCHOR = "        return num_new_blocks + num_evictable_blocks"
_MANAGER_COUNT_INSERT = (
    "        reserved = len(\n"
    "            self._unchain_kv_extent_reservations.get(request_id, ())\n"
    "        )\n"
    "        return max(num_new_blocks - reserved, 0) + num_evictable_blocks"
)
_MANAGER_FREE_ANCHOR = (
    "        ordered_blocks = reversed(req_blocks)"
)
_MANAGER_FREE_INSERT = (
    "        release_reserved_blocks(\n"
    "            self.block_pool,\n"
    "            self._unchain_kv_extent_reservations.pop(request_id, []),\n"
    "        )\n"
    "        if (\n"
    "            getattr(self.block_pool, \"unchain_kv_extent_alloc\", \"off\") != \"off\"\n"
    "            or normalize_release_enabled()\n"
    "        ):\n"
    "            ordered_blocks = normalize_released_blocks(ordered_blocks)\n"
)
_MANAGER_COMPUTED_ALLOC_ANCHOR = (
    "            allocated_blocks = self.block_pool.get_new_blocks(\n"
    "                cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)\n"
    "            )"
)
_MANAGER_COMPUTED_ALLOC_INSERT = (
    "            preferred_after = (\n"
    "                req_blocks[-1].block_id\n"
    "                if req_blocks and not req_blocks[-1].is_null\n"
    "                else None\n"
    "            )\n"
    "            allocated_blocks = self.block_pool.get_new_blocks(\n"
    "                cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks),\n"
    "                preferred_after=preferred_after,\n"
    "            )"
)
_MANAGER_NEW_ALLOC_ANCHOR = (
    "            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)"
)
_MANAGER_NEW_ALLOC_INSERT = (
    "            reserved_blocks = self._unchain_kv_extent_reservations.pop(\n"
    "                request_id, []\n"
    "            )\n"
    "            if (\n"
    "                not reserved_blocks\n"
    "                and self.unchain_kv_extent_reserve_blocks > 0\n"
    "            ):\n"
    "                reserve_after = (\n"
    "                    req_blocks[-1].block_id\n"
    "                    if req_blocks and not req_blocks[-1].is_null\n"
    "                    else None\n"
    "                )\n"
    "                reserved_blocks = select_extent_lease(\n"
    "                    self.block_pool,\n"
    "                    num_new_blocks,\n"
    "                    self.unchain_kv_extent_reserve_blocks,\n"
    "                    reserve_after,\n"
    "                ) or []\n"
    "            consumed = reserved_blocks[:num_new_blocks]\n"
    "            unused = reserved_blocks[num_new_blocks:]\n"
    "            preferred_after = (\n"
    "                consumed[-1].block_id\n"
    "                if consumed\n"
    "                else req_blocks[-1].block_id\n"
    "                if req_blocks and not req_blocks[-1].is_null\n"
    "                else None\n"
    "            )\n"
    "            allocated_blocks = []\n"
    "            try:\n"
    "                remaining = num_new_blocks - len(consumed)\n"
    "                if remaining:\n"
    "                    allocated_blocks = self.block_pool.get_new_blocks(\n"
    "                        remaining, preferred_after=preferred_after\n"
    "                    )\n"
    "                activate_reserved_blocks(self.block_pool, consumed)\n"
    "            except BaseException:\n"
    "                if allocated_blocks:\n"
    "                    self.block_pool.free_blocks(reversed(allocated_blocks))\n"
    "                release_reserved_blocks(\n"
    "                    self.block_pool, consumed + unused\n"
    "                )\n"
    "                raise\n"
    "            new_blocks = consumed + allocated_blocks\n"
    "            if unused:\n"
    "                self._unchain_kv_extent_reservations[request_id] = unused\n"
    "            elif (\n"
    "                self.unchain_kv_extent_reserve_blocks > 0 and new_blocks\n"
    "            ):\n"
    "                reserve_count = self.unchain_kv_extent_reserve_blocks\n"
    "                reservation = select_extent_lease(\n"
    "                    self.block_pool, 1, reserve_count, new_blocks[-1].block_id\n"
    "                )\n"
    "                if reservation:\n"
    "                    self._unchain_kv_extent_reservations[request_id] = (\n"
    "                        reservation\n"
    "                    )"
)


def _check_marker_state(has_bp: bool, has_mgr: bool) -> None:
    if has_bp != has_mgr:
        raise RuntimeError(
            f"extent patch marker mismatch: "
            f"block_pool={'patched' if has_bp else 'clean'}, "
            f"manager={'patched' if has_mgr else 'clean'}"
        )


def patch_extent_order(vllm_root: Path) -> list[Path]:
    bp_path = vllm_root / "vllm/v1/core/block_pool.py"
    mgr_path = vllm_root / "vllm/v1/core/single_type_kv_cache_manager.py"
    bp_text = bp_path.read_text(encoding="utf-8")
    mgr_text = mgr_path.read_text(encoding="utf-8")

    has_bp = _EXTENT_MARKER in bp_text
    has_mgr = _EXTENT_MARKER in mgr_text
    _check_marker_state(has_bp, has_mgr)
    if has_bp and has_mgr:
        return [bp_path, mgr_path]

    # Validate all anchors before any write
    bp_anchors = [
        ("import", _BLOCK_POOL_IMPORT),
        ("init", _BLOCK_POOL_INIT_ANCHOR),
        ("signature", _BLOCK_POOL_SIGNATURE_ANCHOR),
        ("pop", _BLOCK_POOL_POP_ANCHOR),
        ("return", _BLOCK_POOL_RETURN_ANCHOR),
    ]
    mgr_anchors = [
        ("import", _MANAGER_IMPORT_ANCHOR),
        ("init", _MANAGER_INIT_ANCHOR),
        ("fast count", _MANAGER_FAST_COUNT_ANCHOR),
        ("count", _MANAGER_COUNT_ANCHOR),
        ("free", _MANAGER_FREE_ANCHOR),
        ("computed alloc", _MANAGER_COMPUTED_ALLOC_ANCHOR),
        ("new alloc", _MANAGER_NEW_ALLOC_ANCHOR),
    ]
    for name, anchor in bp_anchors:
        if anchor not in bp_text:
            raise RuntimeError(
                f"cannot patch {bp_path}: {name} anchor not found"
            )
    for name, anchor in mgr_anchors:
        if anchor not in mgr_text:
            raise RuntimeError(
                f"cannot patch {mgr_path}: {name} anchor not found"
            )

    # Apply block_pool patches
    new_bp = bp_text.replace(
        _BLOCK_POOL_IMPORT,
        _BLOCK_POOL_IMPORT + "\n" + _BLOCK_POOL_IMPORT_INSERT,
    )
    new_bp = new_bp.replace(
        _BLOCK_POOL_INIT_ANCHOR,
        _BLOCK_POOL_INIT_ANCHOR + "\n" + _BLOCK_POOL_INIT_INSERT,
    )
    new_bp = new_bp.replace(
        _BLOCK_POOL_SIGNATURE_ANCHOR,
        _BLOCK_POOL_SIGNATURE_INSERT,
    )
    new_bp = new_bp.replace(_BLOCK_POOL_POP_ANCHOR, _BLOCK_POOL_POP_INSERT)
    new_bp = new_bp.replace(
        _BLOCK_POOL_RETURN_ANCHOR,
        _BLOCK_POOL_RETURN_INSERT + _BLOCK_POOL_RETURN_ANCHOR,
    )

    # Apply manager patches
    new_mgr = mgr_text.replace(
        _MANAGER_IMPORT_ANCHOR,
        _MANAGER_IMPORT_INSERT + "\n" + _MANAGER_IMPORT_ANCHOR,
    )
    new_mgr = new_mgr.replace(
        _MANAGER_INIT_ANCHOR,
        _MANAGER_INIT_ANCHOR + "\n" + _MANAGER_INIT_INSERT,
    )
    new_mgr = new_mgr.replace(
        _MANAGER_FAST_COUNT_ANCHOR,
        _MANAGER_FAST_COUNT_INSERT,
    )
    new_mgr = new_mgr.replace(
        _MANAGER_COUNT_ANCHOR,
        _MANAGER_COUNT_INSERT,
    )
    # Insert before free_blocks call, preserving indentation
    new_mgr = new_mgr.replace(
        _MANAGER_FREE_ANCHOR,
        _MANAGER_FREE_ANCHOR + "\n" + _MANAGER_FREE_INSERT,
    )
    new_mgr = new_mgr.replace(
        _MANAGER_COMPUTED_ALLOC_ANCHOR,
        _MANAGER_COMPUTED_ALLOC_INSERT,
        1,
    )
    new_mgr = new_mgr.replace(
        _MANAGER_NEW_ALLOC_ANCHOR,
        _MANAGER_NEW_ALLOC_INSERT,
        1,
    )

    # Atomic: write both
    bp_path.write_text(new_bp, encoding="utf-8")
    mgr_path.write_text(new_mgr, encoding="utf-8")
    return [bp_path, mgr_path]


def normalize_released_blocks(ordered_blocks) -> list:
    """Normalize released block order for extent-aware reuse.

    Takes an iterable of blocks (from reversed(request_blocks)) and returns a
    materialized list.  Uncached, ref_cnt==1, non-null blocks are arranged in
    ascending block_id order at those positions; cached, shared, and null blocks
    stay at their original positions.
    """
    blocks = ordered_blocks if isinstance(ordered_blocks, list) else list(ordered_blocks)
    # Collect indices and blocks that are movable
    movable_indices = []
    movable_blocks = []
    for i, b in enumerate(blocks):
        if not b.is_null and b.block_hash is None and b.ref_cnt == 1:
            movable_indices.append(i)
            movable_blocks.append(b)
    movable_blocks.sort(key=lambda b: b.block_id)
    for idx, b in zip(movable_indices, movable_blocks):
        blocks[idx] = b
    return blocks

import argparse
from pathlib import Path


CONNECTOR_MODULE = '''from unchain_kv.vllm_connector import UnchainKVConnector as _UnchainKVConnector


class UnchainKVConnector(_UnchainKVConnector):
    pass
'''


KV_UPDATE_TRACE = (
    "        try:\n"
    "            from vllm.distributed.kv_transfer import (\n"
    "                get_kv_transfer_group,\n"
    "                has_kv_transfer_group,\n"
    "                is_v1_kv_transfer_group,\n"
    "            )\n"
    "            if has_kv_transfer_group() and is_v1_kv_transfer_group():\n"
    "                connector = get_kv_transfer_group()\n"
    "                if connector.has_connector_metadata():\n"
    "                    trace_prefill_kv_update_done = getattr(\n"
    "                        connector,\n"
    "                        \"trace_prefill_kv_update_done\",\n"
    "                        None,\n"
    "                    )\n"
    "                    if trace_prefill_kv_update_done is not None:\n"
    "                        trace_prefill_kv_update_done(layer_name, kv_cache)\n"
    "        except Exception:\n"
    "            pass\n"
)


def write_connector(vllm_root: Path) -> Path:
    target = vllm_root / "vllm/distributed/kv_transfer/kv_connector/v1/unchain_kv_connector.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(CONNECTOR_MODULE, encoding="utf-8")
    return target


def patch_factory(vllm_root: Path) -> Path:
    factory = vllm_root / "vllm/distributed/kv_transfer/kv_connector/factory.py"
    text = factory.read_text(encoding="utf-8")
    marker = 'KVConnectorFactory.register_connector("UnchainKVConnector"'
    if marker in text:
        return factory
    line = (
        '\nKVConnectorFactory.register_connector('
        '"UnchainKVConnector", '
        '"vllm.distributed.kv_transfer.kv_connector.v1.unchain_kv_connector", '
        '"UnchainKVConnector")\n'
    )
    factory.write_text(text + line, encoding="utf-8")
    return factory


def patch_kv_transfer_utils(vllm_root: Path) -> Path:
    target = vllm_root / "vllm/model_executor/layers/attention/kv_transfer_utils.py"
    text = target.read_text(encoding="utf-8")
    if "trace_prefill_attention_start" in text:
        return target
    variants = (
        (
            "        # Execute the function\n"
            "        result = func(*args, **kwargs)\n"
            "        # Save KV cache layer on exit\n"
            "        connector.save_kv_layer(layer_name, kv_cache, attn_metadata)\n"
        ),
        (
            "        # Execute the function\n"
            "        result = func(*args, **kwargs)\n"
            "\n"
            "        # Save KV cache layer on exit\n"
            "        connector.save_kv_layer(layer_name, kv_cache, attn_metadata)\n"
        ),
    )
    new = (
        "        # Execute the function\n"
        "        trace_prefill_attention_start = getattr(\n"
        "            connector, \"trace_prefill_attention_start\", None)\n"
        "        if trace_prefill_attention_start is not None:\n"
        "            trace_prefill_attention_start(layer_name)\n"
        "        result = func(*args, **kwargs)\n"
        "        trace_prefill_attention_done = getattr(\n"
        "            connector, \"trace_prefill_attention_done\", None)\n"
        "        if trace_prefill_attention_done is not None:\n"
        "            trace_prefill_attention_done(layer_name)\n"
        "        # Save KV cache layer on exit\n"
        "        connector.save_kv_layer(layer_name, kv_cache, attn_metadata)\n"
    )
    old = next((variant for variant in variants if variant in text), "")
    if not old:
        raise RuntimeError(f"cannot patch {target}: wrapper shape changed")
    target.write_text(text.replace(old, new), encoding="utf-8")
    return target


def patch_attention_kv_update(vllm_root: Path) -> Path:
    target = vllm_root / "vllm/model_executor/layers/attention/attention.py"
    text = target.read_text(encoding="utf-8")
    if "trace_prefill_kv_update_done" in text:
        return target
    old = (
        "        attn_layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]\n"
        "            attn_layer,\n"
        "            key,\n"
        "            value,\n"
        "            kv_cache,\n"
        "            layer_slot_mapping,\n"
        "        )\n"
    )
    if old not in text:
        raise RuntimeError(f"cannot patch {target}: wrapper shape changed")
    target.write_text(text.replace(old, old + KV_UPDATE_TRACE), encoding="utf-8")
    return target


def patch_rope_kvcache_fusion(vllm_root: Path) -> Path:
    target = vllm_root / "vllm/compilation/passes/fusion/rope_kvcache_fusion.py"
    text = target.read_text(encoding="utf-8")
    if "trace_prefill_kv_update_done" in text:
        return target
    old = (
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
        "        )\n"
    )
    if old not in text:
        raise RuntimeError(f"cannot patch {target}: wrapper shape changed")
    target.write_text(text.replace(old, old + KV_UPDATE_TRACE), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm_root")
    args = parser.parse_args()
    root = Path(args.vllm_root)
    print(write_connector(root))
    print(patch_factory(root))
    print(patch_kv_transfer_utils(root))
    print(patch_attention_kv_update(root))
    print(patch_rope_kvcache_fusion(root))
    for p in patch_extent_order(root):
        print(p)


if __name__ == "__main__":
    main()
