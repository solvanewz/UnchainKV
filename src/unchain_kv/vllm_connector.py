from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
import queue
import random
import threading
import time
from zlib import crc32

from .grant_state import GrantStore
from .layer_state import LayerStore
from .protocol import Chunk, ChunkHeader
from .tensor_blocks import split_block_bytes
from .trace import TraceWriter, summarize_symbol_counts
from .transport import make_receiver, send_chunks, send_grant, transport_name

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1 as _ConnectorBase,
        KVConnectorMetadata as _ConnectorMetadata,
    )
except Exception:
    class _ConnectorMetadata:
        pass

    class _ConnectorBase:
        def __init__(self, vllm_config, role, kv_cache_config=None):
            self._role = role

        @property
        def role(self):
            return self._role


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in {None, ""} else int(value)


def _env_int_set(name: str) -> set[int]:
    value = os.environ.get(name, "")
    return {int(item) for item in value.replace(" ", ",").split(",") if item}


def _env_addr(name: str, default: str) -> tuple[str, int]:
    host, port = os.environ.get(name, default).rsplit(":", 1)
    return host, int(port)


def _align_down(value: int, alignment: int) -> int:
    return max(0, int(value)) // alignment * alignment


def _align_up(value: int, alignment: int) -> int:
    return (max(0, int(value)) + alignment - 1) // alignment * alignment


def _host_memory_headroom(
    meminfo_path: Path = Path("/proc/meminfo"),
    cgroup_path: Path = Path("/sys/fs/cgroup"),
) -> tuple[int, int | None, int | None, int]:
    mem_available = None
    for line in meminfo_path.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            mem_available = int(line.split()[1]) * 1024
            break
    if mem_available is None:
        raise RuntimeError("MemAvailable is unavailable")

    available = mem_available
    cgroup_max = None
    cgroup_current = None
    try:
        maximum = (cgroup_path / "memory.max").read_text().strip()
        current = (cgroup_path / "memory.current").read_text().strip()
        if maximum != "max":
            cgroup_max = int(maximum)
            cgroup_current = int(current)
            available = min(available, max(0, cgroup_max - cgroup_current))
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return mem_available, cgroup_max, cgroup_current, available


def _sample_percentile(samples, fraction: float) -> float:
    values = sorted(float(value) for value in samples)
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, math.ceil(fraction * len(values)) - 1))
    return values[index]


def _adaptive_spool_target(
    batch_bound: int,
    live_cap: int,
    compression_ratio: float,
    fill_bytes_s: float,
    network_bytes_s: float,
    sender_gap_s: float,
) -> int:
    if (
        batch_bound <= 0
        or live_cap < batch_bound
        or not 0 < compression_ratio <= 1
        or fill_bytes_s <= 0
        or network_bytes_s < 0
    ):
        return 0
    backlog = compression_ratio * batch_bound * max(
        0.0, 1.0 - network_bytes_s / fill_bytes_s
    )
    encoded_target = min(
        compression_ratio * live_cap,
        max(
            compression_ratio * batch_bound,
            backlog + network_bytes_s * max(0.0, sender_gap_s),
        ),
    )
    return min(live_cap, math.ceil(encoded_target / compression_ratio))


def _kv_block_size_tokens(vllm_config, kv_cache_config) -> int:
    groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
    if groups:
        spec = getattr(groups[0], "kv_cache_spec", None)
        block_size = getattr(spec, "block_size", None)
        if block_size:
            return int(block_size)
    cache_config = getattr(vllm_config, "cache_config", None)
    return max(1, int(getattr(cache_config, "block_size", 16) or 16))


def _contiguous_span(values: list[int]) -> tuple[int, int] | None:
    if not values:
        return None
    start = values[0]
    if all(value == start + index for index, value in enumerate(values)):
        return start, len(values)
    return None


def _contiguous_runs(values: list[int]) -> tuple[tuple[int, int], ...]:
    if not values:
        return ()
    runs = []
    start = values[0]
    count = 1
    for value in values[1:]:
        if value == start + count:
            count += 1
        else:
            runs.append((start, count))
            start = value
            count = 1
    runs.append((start, count))
    return tuple(runs)


def _block_id_digest(values: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, values)).encode()).hexdigest()


def _tensor_bytes(tensor: object) -> int:
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except (AttributeError, TypeError, ValueError):
        return 0


def _decode_splitzip_top16_payload(
    data: object, raw_bytes: int, codebooks: bytes
) -> bytes | None:
    view = memoryview(data)
    if view.ndim != 1 or view.format != "B":
        view = view.cast("B")
    if (
        raw_bytes <= 0
        or raw_bytes % 4
        or len(codebooks) != 32
        or len(view) < 5
        or view[0] != 5
    ):
        return None
    count = raw_bytes // 2
    code_bytes = (count + 1) // 2
    escape_capacity = (count + 199) // 200
    if len(view) != 5 + count + code_bytes + escape_capacity * 5:
        return None
    escape_count = int.from_bytes(view[1:5], "little")
    if escape_count > escape_capacity:
        return None
    body = view[5 : 5 + count]
    codes = view[5 + count : 5 + count + code_bytes]
    raw = bytearray(raw_bytes)
    plane_words = count // 2
    for index in range(count):
        code = (codes[index // 2] >> ((index % 2) * 4)) & 0x0F
        exponent = codebooks[(index >= plane_words) * 16 + code]
        value = body[index]
        word = (value & 0x7F) | ((value & 0x80) << 8) | (exponent << 7)
        raw[index * 2] = word & 0xFF
        raw[index * 2 + 1] = word >> 8
    entries = 5 + count + code_bytes
    for slot in range(escape_count):
        entry = entries + slot * 5
        position = int.from_bytes(view[entry : entry + 4], "little")
        if position >= count:
            return None
        word = raw[position * 2] | (raw[position * 2 + 1] << 8)
        word = (word & 0x807F) | (view[entry + 4] << 7)
        raw[position * 2] = word & 0xFF
        raw[position * 2 + 1] = word >> 8
    return bytes(raw)


@dataclass
class Session:
    transfer_id: str
    request_id: str
    block_ids: list[int] = field(default_factory=list)
    block_runs: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.block_ids and not self.block_runs:
            self.block_runs = _contiguous_runs(self.block_ids)

    def set_block_ids(self, block_ids: list[int]) -> None:
        self.block_ids = block_ids
        self.block_runs = _contiguous_runs(block_ids)


@dataclass
class PipelineMetadata(_ConnectorMetadata):
    requests: list[Session] = field(default_factory=list)


@dataclass(eq=False)
class _PinnedStageBuffer:
    tensor: object
    future: Future | None = None
    reserved: bool = False
    event: object | None = None
    source: object | None = None
    lease_id: int = 0


@dataclass(eq=False)
class _GpuPackBuffer:
    tensor: object
    future: Future | None = None
    reserved: bool = False
    event: object | None = None
    lease_id: int = 0


@dataclass(eq=False)
class _CodecBuffer:
    tensor: object
    future: Future | None = None
    reserved: bool = False
    event: object | None = None
    lease_id: int = 0


@dataclass
class _StagedTensor:
    tensor: object
    events: list[object]
    d2h_timing: tuple[object, object] | None = None


@dataclass
class _StagedPayload:
    view: memoryview
    events: list[object]


@dataclass
class _SplitzipCudaPayload:
    view: memoryview
    events: list[object]
    source: object | None = None
    out: object | None = None
    layer_index: int = -1
    encode_timing: tuple[object, object] | None = None
    d2h_timing: tuple[object, object] | None = None
    raw_bytes: int = 0
    encoded_bytes: int = 0
    bits: int = 0
    fallback: bool = False
    payload_start: float | None = None
    codec_buffer: _CodecBuffer | None = None
    gpu_pack_buffers: list[_GpuPackBuffer] = field(default_factory=list)
    writeback: bool = False
    writeback_segments: int = 0
    deferred_writeback: bool = False
    top16_epoch: int = -1
    top16_probe: bool = False


@dataclass
class _ReadyPayloadFuture:
    future: Future
    payload: object


@dataclass
class _LayerGroupItem:
    layer_index: int
    data: object
    block_size: int
    block_count: int


@dataclass
class _SpoolItem:
    layer_index: int
    data: bytes
    raw_block_size: int
    block_count: int
    raw_bytes: int
    codec: str


@dataclass(eq=False)
class _RequestSpool:
    transfer_id: str
    request_id: str
    reserved_bytes: int
    items: queue.Queue = field(default_factory=queue.Queue)
    stage_futures: list[Future] = field(default_factory=list)
    pending_stages: int = 0
    materialized_bytes: int = 0
    resident_bytes: int = 0
    sealed: bool = False
    sentinel_sent: bool = False
    released: bool = False
    error: BaseException | None = None


_SPOOL_END = object()


class _EarlyStageSkip(Exception):
    pass


class UnchainKVConnector(_ConnectorBase):
    """Small vLLM KVConnector probe.

    ponytail: CPU byte staging is intentional for v0; replace with direct GPU
    copy only after the layer pipeline trace proves useful.
    """

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self.connector_role = str(role).lower()
        self.kv_role = str(getattr(vllm_config.kv_transfer_config, "kv_role", role))
        self.kv_cache_config = kv_cache_config
        self.transport = transport_name()
        self.peer = _env_addr("UNCHAIN_KV_PEER", "127.0.0.1:29100")
        self.bind = _env_addr("UNCHAIN_KV_BIND", "0.0.0.0:29101")
        self.prefix_peer = (self.peer[0], self.peer[1] + 100)
        self.prefix_bind = (self.bind[0], self.bind[1] + 100)
        self.block_size_tokens = _kv_block_size_tokens(vllm_config, kv_cache_config)
        self.chunk_size = _env_int("UNCHAIN_KV_CHUNK_BYTES", 32768)
        self.max_blocks = _env_int("UNCHAIN_KV_MAX_BLOCKS", 1)
        self.grant_window = _env_int("UNCHAIN_KV_GRANT_WINDOW", 0)
        self.expected_layers = _env_int("UNCHAIN_KV_EXPECTED_LAYERS", 32)
        restore_ahead_value = _env_int("UNCHAIN_KV_RESTORE_AHEAD", 0)
        restore_ahead_requested = restore_ahead_value > 0
        self.restore_ahead = restore_ahead_requested
        self.restore_ahead_window = restore_ahead_value
        self.bulk_decode = _env_int("UNCHAIN_KV_BULK_DECODE", 0) > 0
        self.kv_major_payload = _env_int("UNCHAIN_KV_KV_MAJOR_PAYLOAD", 0) > 0
        self.native_layout_payload = _env_int("UNCHAIN_KV_NATIVE_LAYOUT_PAYLOAD", 0) > 0
        self.block_run_aggregation = _env_int("UNCHAIN_KV_BLOCK_RUNS", 1) > 0
        self.codec = os.environ.get("UNCHAIN_KV_CODEC", "")
        self.codec_min_blocks = max(0, _env_int("UNCHAIN_KV_CODEC_MIN_BLOCKS", 0))
        self.prefix_fast_wait_s = max(
            0.0, float(os.environ.get("UNCHAIN_KV_PREFIX_FAST_WAIT_S", "1.0"))
        )
        codec_bytes = os.environ.get("UNCHAIN_KV_CODEC_GPU_BYTES")
        codec_slots = os.environ.get("UNCHAIN_KV_CODEC_GPU_SLOTS")
        if codec_bytes not in {None, ""}:
            self.codec_gpu_bytes = max(0, int(codec_bytes))
            self.codec_gpu_slots = 0
        elif codec_slots not in {None, ""}:
            self.codec_gpu_bytes = 0
            self.codec_gpu_slots = max(0, int(codec_slots))
        else:
            self.codec_gpu_bytes = 1 << 30 if self.codec == "splitzip_bf16" else 0
            self.codec_gpu_slots = 0
        self.splitzip_fixed5 = _env_int("UNCHAIN_KV_SPLITZIP_FIXED5", 0) > 0
        self.splitzip_fixed5_layers = _env_int_set("UNCHAIN_KV_SPLITZIP_FIXED5_LAYERS")
        self.splitzip_top16 = _env_int("UNCHAIN_KV_SPLITZIP_TOP16", 0) > 0
        self.splitzip_top16_max_cooldown_payloads = max(
            1, _env_int("UNCHAIN_KV_SPLITZIP_TOP16_MAX_COOLDOWN", 32)
        )
        self.codec_writeback_requested = (
            _env_int(
                "UNCHAIN_KV_CODEC_WRITEBACK",
                int(self.codec == "splitzip_bf16" and self.splitzip_top16),
            )
            > 0
        )
        self.codec_writeback_strict = (
            _env_int("UNCHAIN_KV_CODEC_WRITEBACK_STRICT", 0) > 0
        )
        self.codec_writeback = False
        cache_config = getattr(vllm_config, "cache_config", None)
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        self.prefix_caching_enabled = (
            getattr(cache_config, "enable_prefix_caching", None) is True
        )
        self.chunked_prefill_enabled = (
            getattr(scheduler_config, "enable_chunked_prefill", None) is True
        )
        self.max_model_len = max(
            0,
            int(
                getattr(scheduler_config, "max_model_len", 0)
                or getattr(
                    getattr(vllm_config, "model_config", None),
                    "max_model_len",
                    0,
                )
                or 0
            ),
        )
        self.prefix_transfer_suppression = (
            "scheduler" in self.connector_role
            and self.prefix_caching_enabled
            and self.chunked_prefill_enabled
        )
        self.codec_writeback_preserve_source = (
            self.prefix_caching_enabled or self.chunked_prefill_enabled
        )
        self.codec_writeback_overwrite = False
        self.splitzip_chunks = (
            2 if _env_int("UNCHAIN_KV_SPLITZIP_CHUNKS", 1) == 2 else 1
        )
        self.splitzip_native_decode = (
            _env_int("UNCHAIN_KV_SPLITZIP_NATIVE_DECODE", 0) > 0
        )
        self.send_workers = max(1, _env_int("UNCHAIN_KV_SEND_WORKERS", 1))
        self.send_inflight = max(0, _env_int("UNCHAIN_KV_SEND_INFLIGHT", 0))
        self.layer_group_size = max(1, _env_int("UNCHAIN_KV_LAYER_GROUP_SIZE", 1))
        self.early_stage = _env_int("UNCHAIN_KV_EARLY_STAGE", 0) > 0
        self.gpu_pack_layers = max(0, _env_int("UNCHAIN_KV_GPU_PACK_LAYERS", 0))
        self.gpu_pack_bytes = max(0, _env_int("UNCHAIN_KV_GPU_PACK_BYTES", 0))
        self.gpu_pack_strict = _env_int("UNCHAIN_KV_GPU_PACK_STRICT", 0) > 0
        self.early_stage_pack_only = (
            _env_int("UNCHAIN_KV_EARLY_STAGE_PACK_ONLY", 0) > 0
        )
        extent_mode = os.environ.get("UNCHAIN_KV_EXTENT_ALLOC", "off").strip() or "off"
        self.extent_alloc_mode = extent_mode
        if extent_mode not in {"off", "normalize", "prefer"}:
            raise ValueError(
                f"UNCHAIN_KV_EXTENT_ALLOC={extent_mode!r} not supported; "
                f"expected 'off', 'normalize', or 'prefer'"
            )
        if extent_mode != "off":
            self._check_extent_eligibility(kv_cache_config, vllm_config)
        self.permute_block_ids = os.environ.get("UNCHAIN_KV_PERMUTE_BLOCK_IDS", "")
        self.trace_enabled = _env_int("UNCHAIN_KV_TRACE_ENABLED", 1) > 0
        self.trace_cuda_timing = _env_int("UNCHAIN_KV_TRACE_CUDA", 0) > 0
        self.trace_bf16_exponents = (
            _env_int("UNCHAIN_KV_TRACE_BF16_EXPONENTS", 0) > 0
        )
        self.trace_prefill_window = (
            _env_int("UNCHAIN_KV_TRACE_PREFILL_WINDOW", 0) > 0
        )
        self.pinned_staging = _env_int("UNCHAIN_KV_PINNED_STAGING", 0) > 0
        self.host_mirror_layers = max(
            0, _env_int("UNCHAIN_KV_HOST_MIRROR_LAYERS", 0)
        )
        self.host_mirror_bytes = max(
            0, _env_int("UNCHAIN_KV_HOST_MIRROR_BYTES", 0)
        )
        self.request_spool_bytes = max(
            0, _env_int("UNCHAIN_KV_REQUEST_SPOOL_BYTES", 0)
        )
        spool_auto = os.environ.get(
            "UNCHAIN_KV_REQUEST_SPOOL_AUTO", "0"
        ).strip().lower()
        spool_auto_modes = {
            "0": "off",
            "off": "off",
            "false": "off",
            "1": "auto",
            "on": "auto",
            "true": "auto",
            "auto": "auto",
            "observe": "observe",
        }
        if spool_auto not in spool_auto_modes:
            raise ValueError(
                "UNCHAIN_KV_REQUEST_SPOOL_AUTO must be 0, 1, or observe"
            )
        self.request_spool_auto_mode = spool_auto_modes[spool_auto]
        self.host_guard_bytes = max(
            0, _env_int("UNCHAIN_KV_HOST_GUARD_BYTES", 2 << 30)
        )
        self.gpu_guard_bytes = max(
            0, _env_int("UNCHAIN_KV_GPU_GUARD_BYTES", 512 << 20)
        )
        self.auto_spool_hard_bytes = max(
            0, _env_int("UNCHAIN_KV_AUTO_SPOOL_HARD_BYTES", 0)
        )
        self.spool_pressure_ratio = float(
            os.environ.get("UNCHAIN_KV_SPOOL_PRESSURE_RATIO", "1.15")
        )
        if not math.isfinite(self.spool_pressure_ratio) or self.spool_pressure_ratio <= 0:
            raise ValueError("UNCHAIN_KV_SPOOL_PRESSURE_RATIO must be positive")
        live_cap_file = os.environ.get("UNCHAIN_KV_SPOOL_LIVE_CAP_FILE", "")
        self.spool_live_cap_file = Path(live_cap_file) if live_cap_file else None
        self.request_spool_requested = bool(
            self.request_spool_bytes > 0
            or self.request_spool_auto_mode != "off"
        )
        self.request_spool_fixed = bool(
            self.request_spool_bytes > 0
            and self.request_spool_auto_mode == "off"
        )
        self.layer_delay_s = float(os.environ.get("UNCHAIN_KV_LAYER_DELAY_S", "0"))
        self.wait_timeout_s = float(os.environ.get("UNCHAIN_KV_WAIT_TIMEOUT_S", "30"))
        self.transfer_id = os.environ.get("UNCHAIN_KV_TRANSFER_ID", "")
        self.sessions: dict[str, Session] = {}
        self._requests_need_load: dict[str, Session] = {}
        self._requests_need_save: dict[str, Session] = {}
        self._producer_prompt_tokens: dict[str, int] = {}
        self._consumer_prefix_tokens: dict[str, int] = {}
        self._consumer_prefix_transfers: dict[str, str] = {}
        self._prefix_tokens_sent: set[str] = set()
        self._active_transfer_id = ""
        self._active_request_id = ""
        self._active_sessions: dict[str, Session] = {}
        self._load_sessions: dict[str, Session] = {}
        self._active_restore_ahead = False
        self.store = LayerStore()
        self.grants = GrantStore()
        trace_path = os.environ.get("UNCHAIN_KV_TRACE", "/tmp/unchain-kv.jsonl")
        self.trace = TraceWriter(Path(trace_path) if self.trace_enabled else None)
        self.codec_writeback_ineligible = self._codec_writeback_reasons(vllm_config)
        if self.codec_writeback_requested and self.kv_role == "kv_producer":
            self.codec_writeback = not self.codec_writeback_ineligible
            self.codec_writeback_overwrite = (
                self.codec_writeback and not self.codec_writeback_preserve_source
            )
            self.trace.event(
                "codec_writeback_enabled"
                if self.codec_writeback
                else "codec_writeback_ineligible",
                reasons=self.codec_writeback_ineligible,
                overwrite=self.codec_writeback_overwrite,
                preserve_source=self.codec_writeback_preserve_source,
                host_cap_bytes=self.host_mirror_bytes,
                send_inflight=self.send_inflight,
            )
            if self.codec_writeback_strict and self.codec_writeback_ineligible:
                raise RuntimeError(
                    "codec writeback is ineligible: "
                    + ",".join(self.codec_writeback_ineligible)
                )
        spool_worker = (
            self.kv_role == "kv_producer" and "worker" in self.connector_role
        )
        self.request_spool_ineligible = (
            self._request_spool_reasons()
            if self.request_spool_requested and spool_worker
            else []
        )
        self.request_spool_enabled = bool(
            self.request_spool_fixed
            and spool_worker
            and not self.request_spool_ineligible
        )
        self.request_spool_capable = bool(
            self.request_spool_requested
            and spool_worker
            and not self.request_spool_ineligible
        )
        self.payload_ready_enabled = bool(
            _env_int("UNCHAIN_KV_PAYLOAD_READY", 1) > 0
            and spool_worker
            and self.transport == "tcp"
            and self.gpu_pack_strict
            and self.gpu_pack_layers > 0
            and not self.request_spool_fixed
        )
        if self.request_spool_ineligible:
            raise RuntimeError(
                "request spool is ineligible: "
                + ",".join(self.request_spool_ineligible)
            )
        if self.request_spool_capable:
            self.trace.event(
                "request_spool_enabled",
                cap_bytes=self.request_spool_bytes,
                mode=self.request_spool_auto_mode,
            )
        self._all_ready_traced = False
        self._last_finished_state: tuple[str, str, int, int] | None = None
        self._reported_recving: set[str] = set()
        self._restored_layers: set[tuple[str, int]] = set()
        self._last_attn_done_t: float | None = None
        self._traced_grants_sent: set[tuple[str, int]] = set()
        self._traced_bf16_exponents: set[tuple[str, int]] = set()
        self.receiver = None
        self.receiver_thread: threading.Thread | None = None
        self.grant_receiver = None
        self.grant_receiver_thread: threading.Thread | None = None
        self.prefix_receiver = None
        self.prefix_receiver_thread: threading.Thread | None = None
        self.restore_thread: threading.Thread | None = None
        # ponytail: default serial sender; raise workers only when trace shows send-bound runs.
        self.send_executor = (
            ThreadPoolExecutor(max_workers=self.send_workers)
            if self.kv_role == "kv_producer" and self.transport == "tcp"
            else None
        )
        self.spool_executor = (
            ThreadPoolExecutor(max_workers=1)
            if self.request_spool_capable
            and self.request_spool_auto_mode != "observe"
            else None
        )
        self.payload_ready_executor = (
            ThreadPoolExecutor(max_workers=1) if self.payload_ready_enabled else None
        )
        self.send_futures: list[Future] = []
        self._payload_ready_futures: list[Future] = []
        self._pending_send_futures: list[Future] = []
        self._layer_groups: dict[tuple[str, str, str], list[_LayerGroupItem]] = {}
        self._pending_stage_buffers: list[_PinnedStageBuffer] = []
        self._pending_gpu_pack_buffers: list[_GpuPackBuffer] = []
        self._pending_codec_buffers: list[_CodecBuffer] = []
        self._buffer_lease_id = 0
        self._deferred_codec_writebacks: dict[
            tuple[str, int], tuple[list[int], _SplitzipCudaPayload]
        ] = {}
        self._pinned_stage_pool: dict[
            tuple[tuple[int, ...], str], list[_PinnedStageBuffer]
        ] = {}
        self._gpu_pack_pool: dict[
            tuple[tuple[int, ...], str, str], list[_GpuPackBuffer]
        ] = {}
        self._codec_pool: dict[str, list[_CodecBuffer]] = {}
        self._splitzip_fixed6_layers: set[int] = set()
        self._splitzip_top16_cooldowns: dict[int, int] = {}
        self._splitzip_top16_windows: dict[int, int] = {}
        self._splitzip_top16_probe_layers: set[int] = set()
        self._splitzip_top16_epochs: dict[int, int] = {}
        self._splitzip_top16_unavailable_layers: set[int] = set()
        # ponytail: one short global lock; use per-layer locks only if measured.
        self._splitzip_top16_lock = threading.Lock()
        self._pinned_copy_streams: dict[str, object] = {}
        self._restore_lock = threading.Lock()
        self._restore_done: dict[tuple[str, int], threading.Event] = {}
        self._restore_errors: dict[tuple[str, int], BaseException] = {}
        self._restore_ahead_limit = -1
        self._prefill_forward_request = -1
        self._prefill_kv_request = -1
        self._prefill_kv_update_events: dict[int, list[tuple[int, object]]] = {}
        self._prefill_forward_event_starts: dict[int, list[tuple[int, object]]] = {}
        self._prefill_forward_event_pairs: list[tuple[int, int, object, object]] = []
        self._prefill_kv_to_forward_event_pairs: list[
            tuple[int, int, object, object]
        ] = []
        self._pending_codec_decode_events: list[tuple[int, int, object, object]] = []
        self._pending_gpu_pack_events: list[tuple[int, int, object, object]] = []
        self._early_stage_active = False
        self._early_sent_layers: set[tuple[str, int]] = set()
        self._spool_condition = threading.Condition()
        self._request_spools: dict[str, _RequestSpool] = {}
        self._active_request_spools: list[_RequestSpool] = []
        self._spool_layer_block_bytes: list[int] = []
        self._spool_capacity_ready = False
        self._spool_short_fast_path = False
        self._spool_hard_bytes = (
            self.request_spool_bytes if self.request_spool_fixed else 0
        )
        self._spool_live_bytes = self._spool_hard_bytes
        self._spool_target_bytes = self._spool_hard_bytes
        self._spool_gpu_aux_bytes = 0
        self._spool_pack_hard_bytes = self.gpu_pack_bytes
        self._spool_auto_active = False
        self._spool_last_control = 0.0
        self._spool_pressure_windows = 0
        self._spool_clear_windows = 0
        self._spool_shrink_windows = 0
        self._spool_compression_ratio = 1.0
        self._spool_ready_samples = deque(maxlen=32)
        self._spool_network_rates = deque(maxlen=32)
        self._spool_d2h_rates = deque(maxlen=32)
        self._spool_send_waits = deque(maxlen=32)
        self._spool_sender_gaps = deque(maxlen=32)
        self._spool_reserved_bytes = 0
        self._spool_resident_bytes = 0
        self._spool_error: BaseException | None = None
        if (
            self.prefix_transfer_suppression
            and self.codec_min_blocks > 0
            and self.transport == "tcp"
            and self.kv_role == "kv_producer"
        ):
            self._ensure_prefix_receiver()

    def _check_extent_eligibility(self, kv_cache_config, vllm_config):
        """Validate normalize mode: at most one cache group, full-attention type."""
        if kv_cache_config is None:
            return
        groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
        if len(groups) > 1:
            raise RuntimeError(
                "UNCHAIN_KV_EXTENT_ALLOC requires at most "
                f"one KV cache group, got {len(groups)}"
            )
        if not groups:
            return
        spec = getattr(groups[0], "kv_cache_spec", None)
        if spec is None:
            return
        attn_type = getattr(spec, "attention_type", "full")
        if attn_type != "full":
            raise RuntimeError(
                "UNCHAIN_KV_EXTENT_ALLOC requires "
                f"full-attention, got {attn_type!r}"
            )

    def _codec_writeback_reasons(self, vllm_config) -> list[str]:
        if not self.codec_writeback_requested or self.kv_role != "kv_producer":
            return []
        cache_config = getattr(vllm_config, "cache_config", None)
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        checks = (
            (self.transport != "tcp", "transport"),
            (self.codec != "splitzip_bf16", "codec"),
            (not self.splitzip_top16, "top16"),
            (not self.splitzip_native_decode, "native_decode"),
            (self.splitzip_chunks != 1, "codec_chunks"),
            (
                not (self.pinned_staging or self.host_mirror_layers > 0),
                "host_staging",
            ),
            (self.host_mirror_bytes <= 0, "host_byte_cap"),
            (self.send_inflight <= 0, "send_inflight"),
            (
                getattr(cache_config, "enable_prefix_caching", None)
                not in {False, True},
                "prefix_cache",
            ),
            (
                getattr(scheduler_config, "enable_chunked_prefill", None)
                not in {False, True},
                "chunked_prefill",
            ),
        )
        return [reason for failed, reason in checks if failed]

    def _request_spool_reasons(self) -> list[str]:
        checks = (
            (self.transport != "tcp", "transport"),
            (not os.environ.get("UNCHAIN_KV_TCP_LIB"), "tcp_native_lib"),
            (self.codec != "splitzip_bf16", "codec"),
            (not self.splitzip_top16, "top16"),
            (not self.codec_writeback, "writeback"),
            (self.splitzip_chunks != 1, "codec_chunks"),
            (not self.native_layout_payload, "native_layout"),
            (self.send_workers != 1, "send_workers"),
            (self.host_mirror_bytes <= 0, "host_byte_cap"),
            (
                self.request_spool_auto_mode != "off"
                and self.extent_alloc_mode != "prefer",
                "extent_alloc",
            ),
            (
                self.request_spool_auto_mode != "off"
                and (not self.gpu_pack_strict or self.gpu_pack_layers != 1),
                "bounded_pack",
            ),
            (
                self.request_spool_auto_mode != "off"
                and self.send_inflight <= 0,
                "send_inflight",
            ),
        )
        return [reason for failed, reason in checks if failed]

    def _configure_auto_spool_capacity(self, caches: list[object]) -> None:
        mem_available, cgroup_max, cgroup_current, host_available = (
            _host_memory_headroom()
        )
        host_cap = max(
            0, host_available - self.host_guard_bytes - self.host_mirror_bytes
        )
        if self.request_spool_bytes > 0:
            host_cap = min(host_cap, self.request_spool_bytes)
        auto_hard = getattr(self, "auto_spool_hard_bytes", 0)
        if auto_hard > 0:
            host_cap = min(host_cap, auto_hard)
        host_cap = _align_down(host_cap, 64 << 20)
        if host_cap <= 0:
            raise RuntimeError("request spool auto host capacity is not positive")

        max_blocks = self.max_blocks
        if max_blocks <= 0:
            if self.max_model_len <= 0:
                raise RuntimeError("request spool auto requires max_model_len")
            max_blocks = math.ceil(self.max_model_len / self.block_size_tokens)
        max_layer_raw = max(self._spool_layer_block_bytes) * max_blocks
        pack_required = _align_up(max_layer_raw, 2 << 20)
        pack_cap = self.gpu_pack_bytes or pack_required
        if pack_cap < pack_required:
            raise RuntimeError(
                "request spool auto pack cap is too small: "
                f"required={pack_required} cap={pack_cap}"
            )

        import torch

        devices = []
        for cache in caches:
            device = getattr(cache, "device", None)
            if device is not None and str(device) not in {str(item) for item in devices}:
                devices.append(device)
        if not devices:
            raise RuntimeError("request spool auto cannot determine CUDA device")
        memory = [torch.cuda.mem_get_info(device) for device in devices]
        gpu_free = min(int(free) for free, _total in memory)
        gpu_total = min(int(total) for _free, total in memory)
        gpu_aux = gpu_free - self.gpu_guard_bytes
        codec_budget = gpu_aux - pack_cap
        codec_cap = (
            min(self.codec_gpu_bytes, codec_budget)
            if self.codec_gpu_bytes > 0
            else codec_budget
        )
        if codec_cap < max_layer_raw:
            raise RuntimeError(
                "request spool auto GPU minimum working set does not fit: "
                f"codec={codec_cap} pack={pack_cap} aux={gpu_aux}"
            )

        self.gpu_pack_bytes = pack_cap
        self.codec_gpu_bytes = codec_cap
        self.codec_gpu_slots = 0
        self._spool_hard_bytes = host_cap
        self._spool_live_bytes = host_cap
        self._spool_target_bytes = 0
        self._spool_gpu_aux_bytes = gpu_aux
        self._spool_pack_hard_bytes = pack_cap
        self._spool_capacity_ready = True
        self.trace.event(
            "spool_auto_configured",
            host_available_bytes=host_available,
            mem_available_bytes=mem_available,
            cgroup_max_bytes=cgroup_max,
            cgroup_current_bytes=cgroup_current,
            host_guard_bytes=self.host_guard_bytes,
            configured_hard_cap_bytes=auto_hard,
            pressure_ratio=getattr(self, "spool_pressure_ratio", 1.15),
            pinned_cap_bytes=self.host_mirror_bytes,
            hard_cap_bytes=host_cap,
            gpu_free_bytes=gpu_free,
            gpu_total_bytes=gpu_total,
            gpu_guard_bytes=self.gpu_guard_bytes,
            gpu_aux_bytes=gpu_aux,
            codec_cap_bytes=codec_cap,
            pack_cap_bytes=pack_cap,
            max_layer_raw_bytes=max_layer_raw,
        )

    def _refresh_spool_live_cap(self) -> int:
        try:
            _mem, _cg_max, _cg_current, available = _host_memory_headroom()
        except (OSError, RuntimeError):
            return 0
        pinned_remaining = max(
            0, self.host_mirror_bytes - self._pinned_stage_pool_bytes()
        )
        with self._spool_condition:
            resident = self._spool_resident_bytes
        live = resident + max(
            0, available - self.host_guard_bytes - pinned_remaining
        )
        self._spool_live_bytes = max(
            resident,
            _align_down(min(self._spool_hard_bytes, live), 64 << 20),
        )
        live_cap_file = getattr(self, "spool_live_cap_file", None)
        if live_cap_file is not None:
            try:
                configured = max(
                    0, int(live_cap_file.read_text(encoding="utf-8").strip())
                )
            except (OSError, ValueError):
                configured = 0
            self._spool_live_bytes = max(
                resident,
                _align_down(
                    min(self._spool_live_bytes, max(resident, configured)),
                    64 << 20,
                ),
            )
        return self._spool_live_bytes

    def _record_spool_ready(
        self, encoded_bytes: int, raw_bytes: int, elapsed_s: float
    ) -> None:
        if (
            getattr(self, "_spool_short_fast_path", False)
            or
            getattr(self, "request_spool_auto_mode", "off") == "off"
            or encoded_bytes <= 0
            or elapsed_s <= 0
        ):
            return
        ratio = min(1.0, encoded_bytes / max(1, raw_bytes))
        with self._spool_condition:
            self._spool_ready_samples.append(encoded_bytes / elapsed_s)
            self._spool_compression_ratio = (
                0.8 * self._spool_compression_ratio + 0.2 * ratio
            )

    def _record_spool_network(self, encoded_bytes: int, elapsed_s: float) -> None:
        if (
            not getattr(self, "_spool_short_fast_path", False)
            and
            getattr(self, "request_spool_auto_mode", "off") != "off"
            and encoded_bytes > 0
            and elapsed_s > 0
        ):
            with self._spool_condition:
                self._spool_network_rates.append(encoded_bytes / elapsed_s)

    def _record_spool_d2h(self, encoded_bytes: int, elapsed_s: float) -> None:
        if (
            not getattr(self, "_spool_short_fast_path", False)
            and
            getattr(self, "request_spool_auto_mode", "off") != "off"
            and encoded_bytes > 0
            and elapsed_s > 0
        ):
            with self._spool_condition:
                self._spool_d2h_rates.append(encoded_bytes / elapsed_s)

    def _update_auto_spool_control(self, batch_bound: int) -> None:
        if batch_bound <= 0:
            return
        now = time.monotonic()
        live_cap = self._refresh_spool_live_cap()
        with self._spool_condition:
            if now - self._spool_last_control < 1.0:
                self._spool_target_bytes = min(
                    self._spool_target_bytes, live_cap
                )
                return
            self._spool_last_control = now
            ready_rate = _sample_percentile(self._spool_ready_samples, 0.20)
            d2h_rate = _sample_percentile(self._spool_d2h_rates, 0.80)
            network_rate = _sample_percentile(self._spool_network_rates, 0.20)
            fill_rate = min(ready_rate, d2h_rate) if d2h_rate else 0.0
            send_wait = _sample_percentile(self._spool_send_waits, 0.95)
            sender_gap = _sample_percentile(self._spool_sender_gaps, 0.95)
            enough_samples = bool(
                len(self._spool_ready_samples) >= 8
                and len(self._spool_network_rates) >= 8
                and len(self._spool_d2h_rates) >= 2
            )
            pressure = bool(
                enough_samples
                and network_rate > 0
                and fill_rate
                > getattr(self, "spool_pressure_ratio", 1.15) * network_rate
            )
            blocked = send_wait > 0.002

            if not self._spool_auto_active:
                self._spool_pressure_windows = (
                    self._spool_pressure_windows + 1
                    if pressure and blocked
                    else 0
                )
                if self._spool_pressure_windows >= 3:
                    self._spool_auto_active = True
                    self._spool_clear_windows = 0
            else:
                low_resident = self._spool_resident_bytes <= max(
                    1, self._spool_target_bytes // 10
                )
                self._spool_clear_windows = (
                    self._spool_clear_windows + 1
                    if not pressure and low_resident
                    else 0
                )
                if self._spool_clear_windows >= 5:
                    self._spool_auto_active = False
                    self._spool_pressure_windows = 0

            candidate = (
                _adaptive_spool_target(
                    batch_bound,
                    live_cap,
                    self._spool_compression_ratio,
                    fill_rate,
                    network_rate,
                    sender_gap,
                )
                if self._spool_auto_active
                else 0
            )
            old_target = self._spool_target_bytes
            if candidate > old_target:
                self._spool_target_bytes = min(
                    candidate, old_target + batch_bound if old_target else batch_bound
                )
                self._spool_shrink_windows = 0
            elif candidate < 0.8 * old_target:
                self._spool_shrink_windows += 1
                if self._spool_shrink_windows >= 3 or candidate == 0:
                    self._spool_target_bytes = candidate
                    self._spool_shrink_windows = 0
            self._spool_target_bytes = min(self._spool_target_bytes, live_cap)
            target = self._spool_target_bytes
            active = self._spool_auto_active
            reserved = self._spool_reserved_bytes
            resident = self._spool_resident_bytes

        self.trace.event(
            "spool_auto_decision",
            active=active,
            applied=self.request_spool_auto_mode == "auto",
            reason=(
                "pressure" if pressure and blocked else
                "rate_only" if pressure else
                "blocking_only" if blocked else
                "clear"
            ),
            hard_cap_bytes=self._spool_hard_bytes,
            live_cap_bytes=live_cap,
            target_bytes=target,
            reserved_bytes=reserved,
            resident_bytes=resident,
            batch_bound_bytes=batch_bound,
            compression_ratio=self._spool_compression_ratio,
            fill_bytes_s=fill_rate,
            d2h_bytes_s=d2h_rate,
            ready_bytes_s=ready_rate,
            network_bytes_s=network_rate,
            sender_gap_s=sender_gap,
            send_wait_p95_s=send_wait,
            pressure_ratio=getattr(self, "spool_pressure_ratio", 1.15),
            pressure_windows=self._spool_pressure_windows,
            clear_windows=self._spool_clear_windows,
        )

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict) -> bool:
        return True

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return False

    def trace_prefill_attention_start(self, layer_name: str) -> None:
        if self.kv_role == "kv_producer" and self.trace_prefill_window:
            layer = self._layer_index(layer_name)
            if layer == 0:
                self._prefill_forward_request += 1
            self.trace.event(
                "producer_attn_forward_start",
                request=self._prefill_forward_request,
                layer=layer,
            )
            self._record_prefill_forward_start(layer)

    def trace_prefill_attention_done(self, layer_name: str) -> None:
        if self.kv_role == "kv_producer" and self.trace_prefill_window:
            layer = self._layer_index(layer_name)
            self.trace.event(
                "producer_attn_forward_done",
                request=self._prefill_forward_request,
                layer=layer,
            )
            self._record_prefill_forward_done(layer)

    def trace_prefill_kv_update_done(self, layer_name: str, kv_layer=None) -> None:
        if self.kv_role != "kv_producer":
            return
        layer = self._layer_index(layer_name)
        if self.trace_prefill_window:
            if layer == 0:
                self._prefill_kv_request += 1
            self.trace.event(
                "producer_kv_update_done",
                request=self._prefill_kv_request,
                layer=layer,
            )
            self._record_prefill_kv_update_done(layer)
        self._try_early_stage(layer_name, layer, kv_layer)

    def _try_early_stage(self, layer_name: str, layer: int, kv_layer) -> None:
        metadata = getattr(self, "metadata", None)
        if (
            not self.early_stage
            or kv_layer is None
            or self._early_stage_active
            or self.transport != "tcp"
            or self._grant_enabled()
            or not isinstance(metadata, PipelineMetadata)
            or not metadata.requests
            or (
                self.codec_min_blocks > 0
                and any(
                    len(session.block_ids) < self.codec_min_blocks
                    for session in metadata.requests
                )
            )
            or all(
                (session.transfer_id, layer) in self._early_sent_layers
                for session in metadata.requests
            )
        ):
            return
        self._early_stage_active = True
        try:
            self.save_kv_layer(layer_name, kv_layer, None)
        except _EarlyStageSkip:
            self._cancel_deferred_codec_writebacks(
                {session.transfer_id for session in metadata.requests},
                layer_index=layer,
                reason="early_stage_skip",
            )
            self.trace.event("early_stage_skip", layer=layer, reason="gpu_pack_full")
            return
        except Exception as exc:
            self._cancel_deferred_codec_writebacks(
                {session.transfer_id for session in metadata.requests},
                layer_index=layer,
                reason="early_stage_failed",
            )
            self.trace.event("early_stage_failed", layer=layer, error=str(exc)[:160])
            return
        finally:
            self._early_stage_active = False
        for session in metadata.requests:
            self._early_sent_layers.add((session.transfer_id, layer))
        self.trace.event("early_stage_done", layer=layer, requests=len(metadata.requests))

    def build_connector_meta(self, scheduler_output):
        if self.kv_role == "kv_producer" and self.chunked_prefill_enabled:
            return self._build_chunked_producer_meta(scheduler_output)
        meta = PipelineMetadata()
        scheduled = getattr(scheduler_output, "scheduled_new_reqs", [])
        for new_req in scheduled:
            req_id = getattr(new_req, "req_id", "")
            session = self._requests_need_load.pop(
                req_id, self._requests_need_save.pop(req_id, None)
            )
            if session is None:
                continue
            if not session.block_ids:
                session.set_block_ids(
                    self._block_ids(getattr(new_req, "block_ids", []))
                )
            meta.requests.append(session)
        if not meta.requests:
            pending = self._requests_need_load or self._requests_need_save
            meta.requests.extend(pending.values())
            pending.clear()
        return meta

    def _build_chunked_producer_meta(self, scheduler_output) -> PipelineMetadata:
        meta = PipelineMetadata()
        num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", {})

        def complete(req_id: str, computed: int, output_tokens: int = 0) -> bool:
            prompt_tokens = self._producer_prompt_tokens.get(req_id)
            return (
                output_tokens == 0
                and prompt_tokens is not None
                and computed + int(num_scheduled.get(req_id, 0)) >= prompt_tokens
            )

        def finish(req_id: str, session: Session) -> None:
            transfer_suffix = self._prepare_producer_prefix_session(session)
            self._requests_need_save.pop(req_id, None)
            self._producer_prompt_tokens.pop(req_id, None)
            if transfer_suffix:
                meta.requests.append(session)

        for new_req in getattr(scheduler_output, "scheduled_new_reqs", []):
            req_id = getattr(new_req, "req_id", "")
            session = self._requests_need_save.get(req_id)
            if session is None:
                continue
            prompt_token_ids = getattr(new_req, "prompt_token_ids", None)
            if prompt_token_ids is None:
                prompt_token_ids = getattr(new_req, "prefill_token_ids", None)
            if prompt_token_ids is not None:
                self._producer_prompt_tokens[req_id] = len(prompt_token_ids)
            block_ids = self._block_ids(getattr(new_req, "block_ids", []))
            if block_ids:
                session.set_block_ids(block_ids)
            if complete(req_id, int(getattr(new_req, "num_computed_tokens", 0))):
                finish(req_id, session)

        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is None:
            return meta
        req_ids = getattr(cached, "req_ids", [])
        new_block_ids = getattr(cached, "new_block_ids", [])
        computed_tokens = getattr(cached, "num_computed_tokens", [])
        output_tokens = getattr(cached, "num_output_tokens", [])
        resumed = getattr(cached, "resumed_req_ids", set())
        all_token_ids = getattr(cached, "all_token_ids", {})
        for index, req_id in enumerate(req_ids):
            session = self._requests_need_save.get(req_id)
            if session is None:
                continue
            if req_id not in self._producer_prompt_tokens and req_id in all_token_ids:
                self._producer_prompt_tokens[req_id] = len(all_token_ids[req_id])
            blocks = new_block_ids[index] if index < len(new_block_ids) else None
            added = self._block_ids(blocks) if blocks is not None else []
            if req_id in resumed:
                session.set_block_ids(added)
            elif added:
                session.set_block_ids(session.block_ids + added)
            computed = int(computed_tokens[index]) if index < len(computed_tokens) else 0
            outputs = int(output_tokens[index]) if index < len(output_tokens) else 0
            if complete(req_id, computed, outputs):
                finish(req_id, session)
        return meta

    def register_kv_caches(self, kv_caches):
        self.kv_caches = kv_caches
        keys = list(kv_caches.keys()) if isinstance(kv_caches, dict) else []
        values = list(
            kv_caches.values() if isinstance(kv_caches, dict) else kv_caches
        )
        if self.request_spool_capable:
            seen = set()
            self._spool_layer_block_bytes = []
            for cache in values:
                if id(cache) in seen:
                    continue
                seen.add(id(cache))
                shape = getattr(cache, "shape", ())
                if len(shape) < 3 or shape[0] != 2 or int(shape[1]) <= 0:
                    continue
                total = _tensor_bytes(cache)
                if total > 0:
                    self._spool_layer_block_bytes.append(total // int(shape[1]))
            if self.request_spool_auto_mode != "off":
                if len(self._spool_layer_block_bytes) != self.expected_layers:
                    raise RuntimeError(
                        "request spool cache shape mismatch: "
                        f"{len(self._spool_layer_block_bytes)}/"
                        f"{self.expected_layers} layers"
                    )
                self._configure_auto_spool_capacity(values)
            else:
                self._spool_capacity_ready = True
        self.trace.event(
            "kv_cache_registered",
            count=len(kv_caches),
            sample=str(keys[:2]),
            spool_layers=len(self._spool_layer_block_bytes),
        )

    def bind_connector_metadata(self, metadata):
        if self.request_spool_capable and isinstance(metadata, PipelineMetadata):
            self._check_send_futures(done_only=True)
            self._spool_short_fast_path = False
            short_batch = bool(
                self.codec_min_blocks > 0
                and metadata.requests
                and any(
                    len(session.block_ids) < self.codec_min_blocks
                    for session in metadata.requests
                )
            )
            if short_batch:
                self._spool_short_fast_path = True
                self._active_request_spools = []
                self.trace.event(
                    "spool_auto_bypass",
                    requests=len(metadata.requests),
                    requested_bytes=0,
                    cap_bytes=0,
                    reason="short_fast_path",
                )
            elif self.request_spool_fixed:
                self._active_request_spools = self._admit_request_spools(
                    metadata.requests
                )
            else:
                if not self._spool_capacity_ready:
                    raise RuntimeError(
                        "request spool auto capacity is not initialized"
                    )
                self._active_request_spools = []
                bounds = [
                    self._request_spool_bound(session)
                    for session in metadata.requests
                ]
                requested = sum(bounds)
                self._update_auto_spool_control(requested)
                cap = min(
                    self._spool_hard_bytes,
                    self._spool_live_bytes,
                    self._spool_target_bytes,
                )
                if (
                    self.request_spool_auto_mode == "auto"
                    and self._spool_auto_active
                    and requested <= cap
                ):
                    self._active_request_spools = self._admit_request_spools(
                        metadata.requests, cap_bytes=cap, wait=False
                    )
                if requested > 0 and not self._active_request_spools:
                    self.trace.event(
                        "spool_auto_bypass",
                        requests=len(metadata.requests),
                        requested_bytes=requested,
                        cap_bytes=cap,
                        reason=(
                            "observe"
                            if self.request_spool_auto_mode == "observe"
                            else "inactive"
                            if not self._spool_auto_active
                            else "batch_exceeds_cap"
                            if requested > cap
                            else "target_full"
                        ),
                    )
        self.metadata = metadata
        if (
            self.kv_role == "kv_producer"
            and "worker" in self.connector_role
            and isinstance(metadata, PipelineMetadata)
        ):
            for session in metadata.requests:
                runs = self._block_runs(session.block_ids)
                run_lengths = [count for _start, count in runs] if runs else []
                self.trace.event(
                    "producer_block_layout",
                    transfer=session.transfer_id,
                    request=session.request_id,
                    block_digest=_block_id_digest(session.block_ids),
                    blocks=len(session.block_ids),
                    runs=len(run_lengths),
                    longest_run=max(run_lengths) if run_lengths else 0,
                    contiguous=len(run_lengths) == 1 if run_lengths else False,
                    extent_alloc=self.extent_alloc_mode,
                )

    def has_connector_metadata(self):
        return getattr(self, "metadata", None) is not None

    def clear_connector_metadata(self):
        self._cancel_deferred_codec_writebacks(reason="metadata_clear")
        if self._active_request_spools:
            self._seal_active_request_spools(
                RuntimeError("request spool metadata cleared before wait_for_save")
            )
        self.metadata = None
        self._spool_short_fast_path = False

    def _request_spool_bound(self, session: Session) -> int:
        if len(self._spool_layer_block_bytes) != self.expected_layers:
            raise RuntimeError(
                "request spool cache shape mismatch: "
                f"{len(self._spool_layer_block_bytes)}/{self.expected_layers} layers"
            )
        block_count = len(session.block_ids)
        if self.max_blocks > 0:
            block_count = min(block_count, self.max_blocks)
        return sum(
            block_count * block_bytes + 1
            for block_bytes in self._spool_layer_block_bytes
        )

    def _admit_request_spools(
        self,
        sessions: list[Session],
        cap_bytes: int | None = None,
        wait: bool = True,
    ) -> list[_RequestSpool]:
        if not sessions:
            return []
        cap = self.request_spool_bytes if cap_bytes is None else cap_bytes
        bounds = [self._request_spool_bound(session) for session in sessions]
        requested = sum(bounds)
        if requested > cap:
            raise RuntimeError(
                "request spool batch exceeds cap: "
                f"requested={requested} cap={cap}"
            )
        started = time.perf_counter()
        deadline = time.monotonic() + self.wait_timeout_s
        with self._spool_condition:
            while self._spool_reserved_bytes + requested > cap:
                if self._spool_error is not None:
                    raise self._spool_error
                if not wait:
                    return []
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "request spool admission timed out: "
                        f"requested={requested} reserved={self._spool_reserved_bytes} "
                        f"cap={cap}"
                    )
                self._spool_condition.wait(min(0.05, remaining))
            if self._spool_error is not None:
                raise self._spool_error
            duplicate = next(
                (
                    session.transfer_id
                    for session in sessions
                    if session.transfer_id in self._request_spools
                ),
                None,
            )
            if duplicate is not None:
                raise RuntimeError(f"request spool already exists for {duplicate}")
            spools = []
            for session, reserved in zip(sessions, bounds, strict=True):
                spool = _RequestSpool(
                    session.transfer_id,
                    session.request_id,
                    reserved,
                )
                self._request_spools[session.transfer_id] = spool
                self._spool_reserved_bytes += reserved
                spools.append(spool)
        if time.perf_counter() - started > 0.001:
            self.trace.event(
                "spool_admission_wait",
                requests=len(spools),
                requested_bytes=requested,
                wait_s=time.perf_counter() - started,
            )
        submitted = set()
        try:
            if self.send_executor is None:
                raise RuntimeError("send executor is not configured")
            for spool in spools:
                future = self.send_executor.submit(
                    self._drain_request_spool, spool
                )
                submitted.add(spool)
                future._kvx_request_spool = spool
                self.send_futures.append(future)
                self.trace.event(
                    "spool_request_admitted",
                    transfer=spool.transfer_id,
                    request=spool.request_id,
                    reserved_bytes=spool.reserved_bytes,
                    total_reserved_bytes=self._spool_reserved_bytes,
                    cap_bytes=cap,
                    hard_cap_bytes=self._spool_hard_bytes,
                    target_bytes=self._spool_target_bytes,
                )
        except BaseException as exc:
            for spool in spools:
                self._fail_request_spool(spool, exc)
                if spool not in submitted:
                    self._release_request_spool(spool)
            raise
        return spools

    def _submit_spooled_layer(
        self,
        spool: _RequestSpool,
        data: object,
        layer_index: int,
        raw_block_size: int,
        block_count: int,
        raw_bytes: int,
        codec: str,
    ) -> Future:
        if self.spool_executor is None:
            raise RuntimeError("spool executor is not configured")
        rejected = None
        with self._spool_condition:
            if spool.error is not None:
                rejected = spool.error
            elif spool.sealed:
                rejected = RuntimeError(
                    f"request spool is sealed for {spool.transfer_id}"
                )
            else:
                spool.pending_stages += 1
        if rejected is not None:
            self._cleanup_failed_send_args((data,))
            raise rejected
        try:
            future = self.spool_executor.submit(
                self._materialize_spool_item,
                spool,
                data,
                layer_index,
                raw_block_size,
                block_count,
                raw_bytes,
                codec,
            )
        except BaseException as exc:
            with self._spool_condition:
                spool.pending_stages -= 1
            self._cleanup_failed_send_args((data,))
            self._fail_request_spool(spool, exc)
            raise
        with self._spool_condition:
            spool.stage_futures.append(future)
        if self._early_stage_active:
            self._early_sent_layers.add((spool.transfer_id, layer_index))
        self._bind_stage_buffers(future)
        return future

    def _materialize_spool_item(
        self,
        spool: _RequestSpool,
        data: object,
        layer_index: int,
        raw_block_size: int,
        block_count: int,
        raw_bytes: int,
        codec: str,
    ) -> None:
        started = time.perf_counter()
        try:
            owned = bytes(self._ready_payload(data))
            error = None
            with self._spool_condition:
                if spool.error is None:
                    if spool.materialized_bytes + len(owned) > spool.reserved_bytes:
                        error = RuntimeError(
                            "request spool reservation exceeded: "
                            f"transfer={spool.transfer_id} "
                            f"materialized={spool.materialized_bytes + len(owned)} "
                            f"reserved={spool.reserved_bytes}"
                        )
                    else:
                        spool.materialized_bytes += len(owned)
                        spool.resident_bytes += len(owned)
                        self._spool_resident_bytes += len(owned)
                else:
                    error = spool.error
            if error is not None:
                if spool.error is None:
                    self._fail_request_spool(spool, error)
                return
            spool.items.put(
                _SpoolItem(
                    layer_index,
                    owned,
                    raw_block_size,
                    block_count,
                    raw_bytes,
                    codec,
                )
            )
            self.trace.event(
                "spool_layer_ready",
                transfer=spool.transfer_id,
                request=spool.request_id,
                layer=layer_index,
                bytes=len(owned),
                resident_bytes=self._spool_resident_bytes,
                elapsed_s=time.perf_counter() - started,
            )
        except BaseException as exc:
            self._fail_request_spool(spool, exc)
            raise
        finally:
            send_end = False
            with self._spool_condition:
                spool.pending_stages -= 1
                send_end = self._mark_spool_end_locked(spool)
            if send_end:
                spool.items.put(_SPOOL_END)

    def _drain_request_spool(self, spool: _RequestSpool) -> None:
        started = time.perf_counter()
        self.trace.event(
            "spool_send_start",
            transfer=spool.transfer_id,
            request=spool.request_id,
        )
        try:
            while True:
                gap_started = time.perf_counter()
                item = spool.items.get()
                gap_s = time.perf_counter() - gap_started
                if gap_s >= 0.001:
                    with self._spool_condition:
                        self._spool_sender_gaps.append(gap_s)
                if item is _SPOOL_END:
                    break
                try:
                    with self._spool_condition:
                        error = spool.error
                    if error is None:
                        self._send_compressed_native_layer_blocks(
                            spool.transfer_id,
                            spool.request_id,
                            item.layer_index,
                            item.data,
                            item.raw_block_size,
                            item.block_count,
                            item.raw_bytes,
                            item.codec,
                        )
                except BaseException as exc:
                    self._fail_request_spool(spool, exc)
                finally:
                    self._release_spool_item(spool, item)
            if spool.error is not None:
                raise spool.error
            self.trace.event(
                "spool_send_done",
                transfer=spool.transfer_id,
                request=spool.request_id,
                elapsed_s=time.perf_counter() - started,
            )
        except BaseException as exc:
            self.trace.event(
                "spool_send_failed",
                transfer=spool.transfer_id,
                request=spool.request_id,
                error=str(exc)[:160],
            )
            raise
        finally:
            self._release_request_spool(spool)

    def _release_spool_item(
        self, spool: _RequestSpool, item: _SpoolItem
    ) -> None:
        with self._spool_condition:
            spool.resident_bytes -= len(item.data)
            self._spool_resident_bytes -= len(item.data)
            self._spool_condition.notify_all()

    def _trace_writeback_resource_state(self) -> None:
        if not self.codec_writeback_requested:
            return
        with self._spool_condition:
            spool_reserved = self._spool_reserved_bytes
            spool_resident = self._spool_resident_bytes
            spool_requests = len(self._request_spools)
        fields = dict(
            deferred=len(self._deferred_codec_writebacks),
            codec_reserved=sum(
                int(entry.reserved)
                for entries in self._codec_pool.values()
                for entry in entries
            ),
            pack_reserved=sum(
                int(entry.reserved)
                for entries in self._gpu_pack_pool.values()
                for entry in entries
            ),
            pending_codec=len(self._pending_codec_buffers),
            pending_pack=len(self._pending_gpu_pack_buffers),
            spool_reserved=spool_reserved,
            spool_resident=spool_resident,
            spool_requests=spool_requests,
        )
        if getattr(self, "request_spool_auto_mode", "off") != "off":
            fields.update(
                spool_hard=getattr(self, "_spool_hard_bytes", 0),
                spool_live=getattr(self, "_spool_live_bytes", 0),
                spool_target=getattr(self, "_spool_target_bytes", 0),
            )
        self.trace.event("writeback_resource_state", **fields)

    def _release_request_spool(self, spool: _RequestSpool) -> None:
        with self._spool_condition:
            if spool.released:
                return
            spool.released = True
            self._spool_reserved_bytes -= spool.reserved_bytes
            self._request_spools.pop(spool.transfer_id, None)
            self._spool_condition.notify_all()
            reserved = self._spool_reserved_bytes
            resident = self._spool_resident_bytes
        self.trace.event(
            "spool_request_released",
            transfer=spool.transfer_id,
            request=spool.request_id,
            reserved_bytes=reserved,
            resident_bytes=resident,
        )
        self._trace_writeback_resource_state()

    def _fail_request_spool(
        self, spool: _RequestSpool, error: BaseException
    ) -> None:
        send_end = False
        with self._spool_condition:
            if spool.error is None:
                spool.error = error
            if self._spool_error is None:
                self._spool_error = spool.error
            spool.sealed = True
            send_end = self._mark_spool_end_locked(spool)
            self._spool_condition.notify_all()
        if send_end:
            spool.items.put(_SPOOL_END)

    def _mark_spool_end_locked(self, spool: _RequestSpool) -> bool:
        if spool.sealed and spool.pending_stages == 0 and not spool.sentinel_sent:
            spool.sentinel_sent = True
            return True
        return False

    def _seal_request_spool(
        self, spool: _RequestSpool, error: BaseException | None = None
    ) -> None:
        if error is not None:
            self._fail_request_spool(spool, error)
            return
        with self._spool_condition:
            spool.sealed = True
            send_end = self._mark_spool_end_locked(spool)
        if send_end:
            spool.items.put(_SPOOL_END)
        self.trace.event(
            "spool_request_sealed",
            transfer=spool.transfer_id,
            request=spool.request_id,
            bytes=spool.materialized_bytes,
        )

    def _seal_active_request_spools(
        self, error: BaseException | None = None
    ) -> list[_RequestSpool]:
        spools = self._active_request_spools
        self._active_request_spools = []
        for spool in spools:
            self._seal_request_spool(spool, error)
        return spools

    def _wait_spool_stage_futures(
        self, spools: list[_RequestSpool]
    ) -> BaseException | None:
        error = None
        for spool in spools:
            for future in list(spool.stage_futures):
                try:
                    future.result()
                except BaseException as exc:
                    if error is None:
                        error = exc
                finally:
                    host_bytes = getattr(future, "_kvx_host_bytes", 0)
                    if host_bytes:
                        self.trace.event(
                            "host_stage_released",
                            bytes=host_bytes,
                            reserved_bytes=self._pinned_stage_reserved_bytes(),
                        )
                        del future._kvx_host_bytes
            spool.stage_futures.clear()
            if error is None and spool.error is not None:
                error = spool.error
        return error

    def _wait_for_spool_stage_progress(self) -> bool:
        future = None
        with self._spool_condition:
            for spool in self._request_spools.values():
                future = next(
                    (current for current in spool.stage_futures if not current.done()),
                    None,
                )
                if future is not None:
                    break
        if future is None:
            return False
        future.result()
        return True

    def _apply_prefix_transfer_plan(
        self, session: Session, prefix_tokens: int, source: str
    ) -> None:
        total_blocks = len(session.block_ids)
        skipped_blocks = min(
            total_blocks, max(0, int(prefix_tokens)) // self.block_size_tokens
        )
        if skipped_blocks:
            session.set_block_ids(session.block_ids[skipped_blocks:])
        self.trace.event(
            "prefix_transfer_plan",
            request=session.request_id,
            transfer=session.transfer_id,
            source=source,
            prefix_tokens=prefix_tokens,
            block_size_tokens=self.block_size_tokens,
            total_blocks=total_blocks,
            skipped_blocks=skipped_blocks,
            transfer_blocks=len(session.block_ids),
        )

    def _prepare_producer_prefix_session(self, session: Session) -> bool:
        if not self.prefix_transfer_suppression:
            return True
        short_fast_path = bool(
            self.codec_min_blocks
            and len(session.block_ids) < self.codec_min_blocks
        )
        try:
            prefix_tokens = self.grants.wait_value(
                self.prefix_fast_wait_s if short_fast_path else self.wait_timeout_s,
                kind="prefix_tokens",
                transfer_id=session.transfer_id,
            )
        except TimeoutError:
            if not short_fast_path:
                raise
            self.trace.event(
                "prefix_tokens_missing_skip",
                request=session.request_id,
                transfer=session.transfer_id,
                transfer_blocks=len(session.block_ids),
            )
            return False
        self.trace.event(
            "prefix_tokens_received",
            transfer=session.transfer_id,
            prefix_tokens=prefix_tokens,
        )
        self._apply_prefix_transfer_plan(session, prefix_tokens, "producer")
        return bool(session.block_ids)

    def _send_prefix_tokens_once(self, transfer_id: str, prefix_tokens: int) -> None:
        if transfer_id in self._prefix_tokens_sent:
            return
        send_grant(
            self.prefix_peer,
            prefix_tokens,
            kind="prefix_tokens",
            transfer_id=transfer_id,
        )
        self._prefix_tokens_sent.add(transfer_id)
        self.trace.event(
            "prefix_tokens_sent",
            transfer=transfer_id,
            prefix_tokens=prefix_tokens,
        )

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = getattr(request, "kv_transfer_params", None) or {}
        if self.kv_role == "kv_consumer" and params.get("do_remote_prefill"):
            transfer_id = params.get("transfer_id") or self.transfer_id
            request_id = getattr(request, "request_id", "")
            token_ids = getattr(request, "prompt_token_ids", None) or []
            remote_tokens = max(0, len(token_ids) - num_computed_tokens)
            if transfer_id and request_id and remote_tokens:
                session = self.sessions.setdefault(
                    transfer_id, Session(transfer_id, request_id, [])
                )
                self._requests_need_load.setdefault(request_id, session)
            if self.prefix_transfer_suppression and transfer_id and request_id:
                self._consumer_prefix_tokens[request_id] = num_computed_tokens
                self._consumer_prefix_transfers[request_id] = transfer_id
                if num_computed_tokens > 0:
                    self._send_prefix_tokens_once(transfer_id, num_computed_tokens)
            if transfer_id and request_id:
                self.trace.event(
                    "match_request",
                    request=request_id,
                    transfer=transfer_id,
                    prompt_tokens=len(token_ids),
                    prefix_hit_tokens=num_computed_tokens,
                    remote_tokens=remote_tokens,
                )
            return remote_tokens, True
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        params = getattr(request, "kv_transfer_params", None) or {}
        transfer_id = params.get("transfer_id") or self.transfer_id
        if not transfer_id:
            return
        block_ids = self._block_ids(blocks)
        session = Session(transfer_id, request.request_id, block_ids)
        if self.kv_role == "kv_consumer" and self.prefix_transfer_suppression:
            prefix_tokens = self._consumer_prefix_tokens.pop(
                request.request_id, None
            )
            if not prefix_tokens:
                prompt_token_ids = getattr(request, "prompt_token_ids", None)
                prefix_tokens = max(
                    0,
                    len(prompt_token_ids or []) - int(num_external_tokens),
                )
            self._consumer_prefix_transfers[request.request_id] = transfer_id
            self._send_prefix_tokens_once(transfer_id, prefix_tokens)
            self._apply_prefix_transfer_plan(
                session,
                prefix_tokens,
                "consumer",
            )
        self.trace.event(
            "alloc_state",
            request=request.request_id,
            transfer=transfer_id,
            external_tokens=num_external_tokens,
        )
        if self.kv_role == "kv_consumer" and num_external_tokens <= 0:
            self.sessions.pop(transfer_id, None)
            self._requests_need_load.pop(request.request_id, None)
            return
        self.sessions[transfer_id] = session
        if self.kv_role == "kv_consumer":
            self._requests_need_load[request.request_id] = session
        elif self.kv_role == "kv_producer" and params.get("do_remote_decode"):
            if self.prefix_transfer_suppression:
                self._ensure_prefix_receiver()
            self._requests_need_save[request.request_id] = session
            prompt_token_ids = getattr(request, "prompt_token_ids", None)
            if prompt_token_ids is not None:
                self._producer_prompt_tokens[request.request_id] = len(
                    prompt_token_ids
                )

    def start_load_kv(self, forward_context, **kwargs):
        if self.kv_role != "kv_consumer":
            return
        metadata = getattr(self, "metadata", None)
        requests = getattr(metadata, "requests", []) or []
        if requests:
            self._active_transfer_id = requests[0].transfer_id
            self._active_request_id = requests[0].request_id
            self._all_ready_traced = False
            self._last_finished_state = None
            self._last_attn_done_t = None
            for session in requests:
                self.sessions[session.transfer_id] = session
                self._active_sessions[session.transfer_id] = session
                self.trace.event(
                    "consumer_block_layout",
                    transfer=session.transfer_id,
                    request=session.request_id,
                    block_digest=_block_id_digest(session.block_ids),
                    blocks=len(session.block_ids),
                    runs=len(session.block_runs),
                )
        if self._active_sessions:
            self._active_restore_ahead = self._should_restore_ahead(
                self._active_sessions.values()
            )
            if self._active_restore_ahead:
                self._allow_restore_through(self.restore_ahead_window)
        self.trace.event(
            "start_load",
            active=bool(self._active_transfer_id),
            requests=len(requests),
        )
        if self.receiver is None:
            self.receiver = make_receiver(self.bind, self.store, trace=self.trace)
            self.receiver_thread = threading.Thread(
                target=self.receiver.serve, daemon=True
            )
            self.receiver_thread.start()
        if self._grant_enabled() and self._active_sessions:
            for session in self._active_sessions.values():
                for layer in range(min(self.grant_window, self.expected_layers)):
                    self._send_layer_grant(layer, session.transfer_id)
        self._ensure_restore_thread()

    def wait_for_layer_load(self, layer_name: str):
        self._flush_codec_decode_events()
        if self.kv_role != "kv_consumer":
            return
        layer_index = self._layer_index(layer_name)
        sessions = self._sessions_to_load()
        if not sessions:
            return
        if self._restore_ahead_for_wait(sessions):
            self._allow_restore_through(layer_index + self.restore_ahead_window)
            self._ensure_restore_thread()
        self.trace.event("wait_layer_start", layer=layer_index)
        network_wait_start = time.perf_counter()
        ready_before_wait = all(
            self.store.is_ready(session.transfer_id, layer_index)
            for session in sessions
        )
        for session in sessions:
            self.store.wait(session.transfer_id, layer_index, self.wait_timeout_s)
        ready_t = time.perf_counter()
        self.trace.event(
            "network_ready_wait",
            layer=layer_index,
            blocked=not ready_before_wait,
            wait_s=ready_t - network_wait_start,
        )
        self.trace.event(
            "wait_layer_done", t=ready_t, layer=layer_index, requests=len(sessions)
        )
        if self._grant_enabled():
            # The producer handles all requests in a forward batch before it can
            # advance to the next layer.  Grant every active transfer at the
            # consumer frontier, even when only a subset has been admitted for
            # this decode step, or the two sides can wait on different requests.
            grant_sessions = (
                self._active_sessions.values() if self._active_sessions else sessions
            )
            for session in grant_sessions:
                for layer in range(
                    layer_index + 1,
                    min(layer_index + 1 + self.grant_window, self.expected_layers),
                ):
                    self._send_layer_grant(layer, session.transfer_id)
        if self._last_attn_done_t is not None and not ready_before_wait:
            self.trace.event(
                "layer_stall",
                layer=layer_index,
                stall_s=max(0.0, ready_t - self._last_attn_done_t),
            )
        restore_wait_start = time.perf_counter()
        if self._restore_ahead_for_wait(sessions):
            for session in sessions:
                self._wait_restore_done(session.transfer_id, layer_index)
        else:
            for session in sessions:
                self._restore_layer(layer_name, session.transfer_id)
        self.trace.event(
            "restore_wait",
            layer=layer_index,
            wait_s=time.perf_counter() - restore_wait_start,
        )
        if not self._all_ready_traced and all(
            self.store.is_ready(session.transfer_id, index)
            for session in sessions
            for index in range(self.expected_layers)
        ):
            self._all_ready_traced = True
            self.trace.event("all_layers_ready")
        self.trace.event("attn_layer_start", layer=layer_index)

    def save_kv_layer(self, layer_name: str, kv_layer, attn_metadata, **kwargs):
        self._flush_codec_decode_events()
        if self.kv_role == "kv_consumer":
            done_t = time.perf_counter()
            self._last_attn_done_t = done_t
            self.trace.event(
                "attn_layer_done", t=done_t, layer=self._layer_index(layer_name)
            )
            return
        metadata = getattr(self, "metadata", None)
        if self.kv_role != "kv_producer" or not isinstance(metadata, PipelineMetadata):
            return
        layer_index = self._layer_index(layer_name)
        early_active = self._early_stage_active
        sessions = [
            session
            for session in metadata.requests
            if not self.early_stage
            or (session.transfer_id, layer_index) not in self._early_sent_layers
        ]
        if not early_active:
            self._finish_deferred_codec_writebacks(
                kv_layer, layer_index, metadata.requests
            )
        self._check_send_futures(done_only=True)
        if not sessions:
            if not early_active and metadata.requests:
                self.trace.event("layer_send_skip_early", layer=layer_index)
            return
        if self._grant_enabled():
            self._ensure_grant_receiver()
        if self._grant_enabled():
            for session in sessions:
                self._wait_for_grant(layer_index, session.transfer_id)
        source = "early" if early_active else "save"
        self.trace.event("layer_send_start", layer=layer_index, source=source)
        if self.transport == "tcp":
            for session in sessions:
                stage_start = time.perf_counter()
                if self.max_blocks > 0:
                    block_ids = session.block_ids[: self.max_blocks]
                    block_runs = self._block_runs(block_ids)
                else:
                    block_ids = session.block_ids
                    block_runs = self._block_runs(block_ids)
                contiguous_blocks = len(block_runs) == 1
                if self._try_send_splitzip_two_chunks(
                    kv_layer,
                    block_ids,
                    contiguous_blocks,
                    session,
                    layer_index,
                    source,
                ):
                    continue
                codec_native = (
                    self._codec_native_blocks(
                        kv_layer, block_ids, layer_index, session.transfer_id
                    )
                    if self._use_compressed_native_codec(block_ids)
                    and (contiguous_blocks or self.gpu_pack_layers > 0)
                    else None
                )
                if codec_native is not None:
                    (
                        native_data,
                        block_size,
                        raw_bytes,
                        encoded_bytes,
                        codec,
                        ratio,
                        codec_elapsed_s,
                    ) = codec_native
                    if codec == "splitzip_bf16":
                        self.trace.event(
                            "codec_encode_submitted",
                            layer=layer_index,
                            raw_bytes=raw_bytes,
                            encoded_bytes=encoded_bytes,
                            ratio=ratio,
                            submit_elapsed_s=codec_elapsed_s,
                            source=source,
                        )
                    else:
                        self.trace.event(
                            "codec_raw_fallback",
                            layer=layer_index,
                            raw_bytes=raw_bytes,
                            encoded_bytes=encoded_bytes,
                            ratio=ratio,
                            codec_elapsed_s=codec_elapsed_s,
                            source=source,
                        )
                    if (
                        early_active
                        and self.codec_writeback_overwrite
                        and isinstance(native_data, _SplitzipCudaPayload)
                        and native_data.bits == 4
                        and not native_data.fallback
                        and native_data.codec_buffer is not None
                    ):
                        native_data.deferred_writeback = True
                        if native_data.codec_buffer in self._pending_codec_buffers:
                            self._pending_codec_buffers.remove(
                                native_data.codec_buffer
                            )
                        self._deferred_codec_writebacks[
                            (session.transfer_id, layer_index)
                        ] = (list(block_ids), native_data)
                    stage_elapsed_s = time.perf_counter() - stage_start
                    self.trace.event(
                        "layer_stage_done",
                        layer=layer_index,
                        bytes=encoded_bytes,
                        raw_bytes=raw_bytes,
                        chunks=len(block_ids),
                        contiguous_blocks=contiguous_blocks,
                        layout="compressed_native",
                        codec=codec,
                        source=source,
                        stage_elapsed_s=stage_elapsed_s,
                    )
                    self._record_spool_ready(
                        encoded_bytes, raw_bytes, stage_elapsed_s
                    )
                    spool = self._request_spools.get(session.transfer_id)
                    if spool is not None:
                        if codec != "splitzip_bf16":
                            self._cleanup_failed_send_args((native_data,))
                            if early_active:
                                raise _EarlyStageSkip()
                            raise RuntimeError(
                                "request spool requires splitzip payload"
                            )
                        self._submit_spooled_layer(
                            spool,
                            native_data,
                            layer_index,
                            block_size,
                            len(block_ids),
                            raw_bytes,
                            codec,
                        )
                    else:
                        future = self._submit_send(
                            self._send_compressed_native_layer_blocks,
                            session.transfer_id,
                            session.request_id,
                            layer_index,
                            native_data,
                            block_size,
                            len(block_ids),
                            raw_bytes,
                            codec,
                        )
                        self._bind_stage_buffers(future)
                    continue
                native = (
                    self._native_blocks_bytes(kv_layer, block_ids)
                    if self.native_layout_payload
                    and (contiguous_blocks or self.gpu_pack_layers > 0)
                    else None
                )
                if native is not None:
                    native_data, block_size = native
                    self.trace.event(
                        "layer_stage_done",
                        layer=layer_index,
                        bytes=block_size * len(block_ids),
                        chunks=len(block_ids),
                        contiguous_blocks=contiguous_blocks,
                        layout="native",
                        source=source,
                        stage_elapsed_s=time.perf_counter() - stage_start,
                    )
                    if not self._queue_layer_group(
                        session.transfer_id,
                        session.request_id,
                        layer_index,
                        native_data,
                        block_size,
                        len(block_ids),
                        "native",
                    ):
                        future = self._submit_send(
                            self._send_native_layer_blocks,
                            session.transfer_id,
                            session.request_id,
                            layer_index,
                            native_data,
                            block_size,
                            len(block_ids),
                        )
                        self._bind_stage_buffers(future)
                    continue
                kv_major = (
                    self._kv_major_blocks_bytes(kv_layer, block_ids)
                    if self.kv_major_payload and contiguous_blocks
                    else None
                )
                if kv_major is not None:
                    key_data, value_data, part_size = kv_major
                    self.trace.event(
                        "layer_stage_done",
                        layer=layer_index,
                        bytes=part_size * len(block_ids) * 2,
                        chunks=len(block_ids),
                        contiguous_blocks=contiguous_blocks,
                        layout="kv_major",
                        source=source,
                        stage_elapsed_s=time.perf_counter() - stage_start,
                    )
                    future = self._submit_send(
                        self._send_kv_layer_blocks,
                        session.transfer_id,
                        session.request_id,
                        layer_index,
                        key_data,
                        value_data,
                        part_size,
                        len(block_ids),
                    )
                    self._bind_stage_buffers(future)
                    continue
                blocks_data, block_size = self._blocks_bytes(kv_layer, block_ids)
                self.trace.event(
                    "layer_stage_done",
                    layer=layer_index,
                    bytes=block_size * len(block_ids),
                    chunks=len(block_ids),
                    contiguous_blocks=contiguous_blocks,
                    layout="block_major",
                    source=source,
                    stage_elapsed_s=time.perf_counter() - stage_start,
                )
                if not self._queue_layer_group(
                    session.transfer_id,
                    session.request_id,
                    layer_index,
                    blocks_data,
                    block_size,
                    len(block_ids),
                    "block_major",
                ):
                    future = self._submit_send(
                        self._send_layer_blocks,
                        session.transfer_id,
                        session.request_id,
                        layer_index,
                        blocks_data,
                        block_size,
                        len(block_ids),
                    )
                    self._bind_stage_buffers(future)
            if self.layer_delay_s > 0:
                time.sleep(self.layer_delay_s)
            return

        chunks: list[Chunk] = []
        for session in sessions:
            block_ids = (
                session.block_ids[: self.max_blocks]
                if self.max_blocks > 0
                else session.block_ids
            )
            blocks_data, block_size = self._blocks_bytes(kv_layer, block_ids)
            for block_index, block_id in enumerate(block_ids):
                start = block_index * block_size
                data = blocks_data[start : start + block_size]
                pieces = list(split_block_bytes(data, self.chunk_size))
                for piece_index, (offset, payload) in enumerate(pieces):
                    payload = bytes(payload)
                    chunk_index = block_index * len(pieces) + piece_index
                    chunks.append(
                        Chunk(
                            ChunkHeader(
                                session.transfer_id,
                                session.request_id,
                                layer_index,
                                block_index,
                                chunk_index,
                                len(block_ids) * len(pieces),
                                offset,
                                len(payload),
                                crc32(payload) & 0xFFFFFFFF,
                            ),
                            payload,
                        )
                    )
        random.shuffle(chunks)
        send_chunks(self.peer, chunks)
        self.trace.event("layer_send_done", layer=layer_index, chunks=len(chunks))
        if self.layer_delay_s > 0:
            time.sleep(self.layer_delay_s)

    def wait_for_save(self):
        self._cancel_deferred_codec_writebacks(reason="wait_for_save")
        error = None
        try:
            self._flush_layer_groups()
            if (
                getattr(self, "request_spool_capable", False)
                and not getattr(self, "_spool_short_fast_path", False)
            ):
                spools = list(self._active_request_spools)
                if spools:
                    error = self._wait_spool_stage_futures(spools)
                    self._seal_active_request_spools()
                try:
                    self._check_send_futures(done_only=True)
                except BaseException as exc:
                    if error is None:
                        error = exc
            else:
                self._check_send_futures(done_only=False)
            self._flush_prefill_forward_events()
        except BaseException as exc:
            if error is None:
                error = exc
        finally:
            if self._active_request_spools:
                self._seal_active_request_spools(error)
            self._cancel_deferred_codec_writebacks(reason="wait_for_save")
            self._trace_writeback_resource_state()
        if error is not None:
            raise error
        return None

    def get_finished(self, finished_req_ids=None):
        self._flush_codec_decode_events()
        if self._request_spools:
            self._check_send_futures(done_only=True)
        if finished_req_ids:
            finished = set(finished_req_ids)
            finished_transfers = {
                transfer_id
                for sessions in (
                    self.sessions,
                    self._active_sessions,
                    self._load_sessions,
                )
                for transfer_id, session in sessions.items()
                if session.request_id in finished
            }
            self._cancel_deferred_codec_writebacks(
                finished_transfers,
                reason="request_finished",
            )
            for sessions in (
                self.sessions,
                self._active_sessions,
                self._load_sessions,
            ):
                for transfer_id, session in list(sessions.items()):
                    if session.request_id in finished:
                        del sessions[transfer_id]
            for request_id in finished:
                self._requests_need_load.pop(request_id, None)
                self._requests_need_save.pop(request_id, None)
                self._producer_prompt_tokens.pop(request_id, None)
                self._consumer_prefix_tokens.pop(request_id, None)
                transfer_id = self._consumer_prefix_transfers.pop(request_id, None)
                if transfer_id is not None:
                    finished_transfers.add(transfer_id)
            self._prefix_tokens_sent.difference_update(finished_transfers)
            if self.kv_role == "kv_consumer":
                for transfer_id in finished_transfers:
                    released = self.store.discard_transfer(transfer_id)
                    self._discard_restore_state(transfer_id)
                    self.trace.event(
                        "load_transfer_released",
                        transfer=transfer_id,
                        bytes=released,
                        retained_bytes=self.store.payload_bytes(),
                    )
            if self._active_sessions:
                session = next(iter(self._active_sessions.values()))
                self._active_transfer_id = session.transfer_id
                self._active_request_id = session.request_id
            else:
                self._active_transfer_id = ""
                self._active_request_id = ""
            self.trace.event(
                "sessions_finished",
                requests=len(finished),
                active=len(self._active_sessions),
            )
        if self.kv_role == "kv_consumer":
            if not self._active_sessions and self.sessions:
                self._active_sessions.update(self.sessions)
                session = next(iter(self._active_sessions.values()))
                self._active_transfer_id = session.transfer_id
                self._active_request_id = session.request_id
            if self._active_sessions:
                ready = sum(
                    self.store.is_ready(session.transfer_id, index)
                    for session in self._active_sessions.values()
                    for index in range(self.expected_layers)
                )
                state = (
                    self._active_transfer_id,
                    self._active_request_id,
                    ready,
                    self.expected_layers,
                )
                if state != self._last_finished_state:
                    self._last_finished_state = state
                    self.trace.event(
                        "load_progress",
                        ready_layers=ready,
                        expected_layers=self.expected_layers,
                    )
            if self._active_sessions and ready == (
                self.expected_layers * len(self._active_sessions)
            ):
                if not self._all_ready_traced:
                    self._all_ready_traced = True
                    self.trace.event("all_layers_ready")
            if self.bulk_decode:
                ready_sessions = [
                    session
                    for session in self._active_sessions.values()
                    if all(
                        self.store.is_ready(session.transfer_id, index)
                        for index in range(self.expected_layers)
                    )
                ]
            else:
                ready_sessions = [
                    session
                    for session in self._active_sessions.values()
                    if self.store.is_ready(session.transfer_id, 0)
                ]
            recving = {
                session.request_id
                for session in ready_sessions
                if session.request_id not in self._reported_recving
            }
            if recving:
                for session in ready_sessions:
                    if session.request_id in recving:
                        self._load_sessions[session.transfer_id] = session
                self._reported_recving.update(recving)
                return set(), recving
        return None, None

    def get_block_ids_with_load_errors(self):
        return set()

    def _block_ids(self, blocks) -> list[int]:
        if hasattr(blocks, "get_block_ids"):
            groups = blocks.get_block_ids()
            return self._maybe_permute_block_ids(list(groups[0]) if groups else [])
        if hasattr(blocks, "get_unhashed_block_ids_all_groups"):
            groups = blocks.get_unhashed_block_ids_all_groups()
            return self._maybe_permute_block_ids(list(groups[0]) if groups else [])
        if hasattr(blocks, "get_unhashed_block_ids"):
            return self._maybe_permute_block_ids(list(blocks.get_unhashed_block_ids()))
        values = list(blocks or [])
        if values and isinstance(values[0], (list, tuple)):
            return self._maybe_permute_block_ids(list(values[0]))
        return self._maybe_permute_block_ids(values)

    def _maybe_permute_block_ids(self, values: list[int]) -> list[int]:
        if self.permute_block_ids == "odd_even":
            return values[1::2] + values[::2]
        return values

    def _block_bytes(self, kv_layer, block_id: int) -> bytes:
        shape = getattr(kv_layer, "shape", ())
        dim = kv_layer.dim() if hasattr(kv_layer, "dim") else len(shape)
        block = kv_layer[:, block_id] if dim >= 3 and shape[0] == 2 else kv_layer[block_id]
        block = block.detach().contiguous().cpu()
        if str(block.dtype) == "torch.bfloat16":
            import torch

            block = block.view(torch.uint16)
        return block.numpy().tobytes()

    def _block_runs(self, block_ids: list[int]) -> tuple[tuple[int, int], ...]:
        if self.block_run_aggregation:
            return _contiguous_runs(block_ids)
        return tuple((block_id, 1) for block_id in block_ids)

    def _blocks_bytes(self, kv_layer, block_ids: list[int]) -> tuple[bytes | memoryview, int]:
        if not block_ids:
            return b"", 0
        if not self.block_run_aggregation:
            blocks = [self._block_bytes(kv_layer, block_id) for block_id in block_ids]
            return b"".join(blocks), len(blocks[0]) if blocks else 0
        if not hasattr(kv_layer, "index_select") or not hasattr(kv_layer, "device"):
            blocks = [self._block_bytes(kv_layer, block_id) for block_id in block_ids]
            return b"".join(blocks), len(blocks[0]) if blocks else 0

        shape = getattr(kv_layer, "shape", ())
        dim = kv_layer.dim() if hasattr(kv_layer, "dim") else len(shape)
        span = _contiguous_span(block_ids)
        if dim >= 3 and shape[0] == 2 and len(block_ids) == 1:
            blocks = kv_layer[:, block_ids[0]]
        elif dim >= 3 and shape[0] == 2 and span is not None:
            start, count = span
            blocks = kv_layer[:, start : start + count].transpose(0, 1)
        elif dim >= 3 and shape[0] == 2:
            import torch

            ids = torch.as_tensor(block_ids, dtype=torch.long, device=kv_layer.device)
            blocks = kv_layer.index_select(1, ids).transpose(0, 1)
        elif span is not None:
            start, count = span
            blocks = kv_layer[start : start + count]
        else:
            import torch

            ids = torch.as_tensor(block_ids, dtype=torch.long, device=kv_layer.device)
            blocks = kv_layer.index_select(0, ids)
        staged = self._stage_cpu(blocks.detach())
        blocks, events = self._staged_tensor_parts(staged)
        if str(blocks.dtype) == "torch.bfloat16":
            import torch

            blocks = blocks.view(torch.uint16)
        data = memoryview(blocks.numpy()).cast("B")
        return self._staged_payload(data, events), data.nbytes // len(block_ids)

    def _kv_major_blocks_bytes(
        self, kv_layer, block_ids: list[int]
    ) -> tuple[bytes | memoryview, bytes | memoryview, int] | None:
        if not block_ids:
            return b"", b"", 0
        if not hasattr(kv_layer, "index_select") or not hasattr(kv_layer, "device"):
            return None
        shape = getattr(kv_layer, "shape", ())
        dim = kv_layer.dim() if hasattr(kv_layer, "dim") else len(shape)
        if dim < 3 or shape[0] != 2:
            return None
        span = _contiguous_span(block_ids)
        if span is None:
            return None
        start, count = span
        key = kv_layer[0, start : start + count]
        value = kv_layer[1, start : start + count]
        key_staged = self._stage_cpu(key.detach())
        value_staged = self._stage_cpu(value.detach())
        key, key_events = self._staged_tensor_parts(key_staged)
        value, value_events = self._staged_tensor_parts(value_staged)
        if str(key.dtype) == "torch.bfloat16":
            import torch

            key = key.view(torch.uint16)
            value = value.view(torch.uint16)
        key_data = memoryview(key.numpy()).cast("B")
        value_data = memoryview(value.numpy()).cast("B")
        return (
            self._staged_payload(key_data, key_events),
            self._staged_payload(value_data, value_events),
            key_data.nbytes // len(block_ids),
        )

    def _native_blocks_bytes(
        self, kv_layer, block_ids: list[int]
    ) -> tuple[bytes | memoryview, int] | None:
        if not block_ids:
            return b"", 0
        if not hasattr(kv_layer, "device"):
            return None
        shape = getattr(kv_layer, "shape", ())
        dim = kv_layer.dim() if hasattr(kv_layer, "dim") else len(shape)
        span = _contiguous_span(block_ids)
        if dim < 3 or shape[0] != 2:
            return None
        if span is None:
            return self._gpu_pack_native_blocks_bytes(kv_layer, block_ids)
        start, count = span
        blocks = kv_layer[:, start : start + count]
        staged = self._stage_cpu(blocks.detach())
        blocks, events = self._staged_tensor_parts(staged)
        if str(blocks.dtype) == "torch.bfloat16":
            import torch

            blocks = blocks.view(torch.uint16)
        data = memoryview(blocks.numpy()).cast("B")
        return self._staged_payload(data, events), data.nbytes // len(block_ids)

    def _gpu_pack_native_blocks_bytes(
        self, kv_layer, block_ids: list[int]
    ) -> tuple[bytes | memoryview, int] | None:
        if self.gpu_pack_layers <= 0 or not hasattr(kv_layer, "index_select"):
            if self.gpu_pack_strict:
                raise RuntimeError("strict gpu pack requires a configured pack pool")
            return None
        import torch

        shape = (2, len(block_ids)) + tuple(kv_layer.shape[2:])
        entry = self._wait_for_gpu_pack_buffer(
            shape, kv_layer.dtype, kv_layer.device, torch
        )
        if entry is None:
            if self._early_stage_active and (
                self.early_stage_pack_only or self.gpu_pack_strict
            ):
                raise _EarlyStageSkip()
            self.trace.event("gpu_pack_full", blocks=len(block_ids))
            if self.gpu_pack_strict:
                raise RuntimeError("gpu pack capacity exceeded")
            return None
        ids = torch.as_tensor(block_ids, dtype=torch.long, device=kv_layer.device)
        torch.index_select(kv_layer, 1, ids, out=entry.tensor)
        try:
            staged = self._stage_cpu(entry.tensor.detach())
        except BaseException:
            self._release_pending_gpu_pack_buffers()
            raise
        blocks, events = self._staged_tensor_parts(staged)
        self._release_gpu_pack_buffers(
            self._take_pending_gpu_pack_buffers(),
            events[-1] if events else None,
        )
        if str(blocks.dtype) == "torch.bfloat16":
            blocks = blocks.view(torch.uint16)
        data = memoryview(blocks.numpy()).cast("B")
        return self._staged_payload(data, events), data.nbytes // len(block_ids)

    def _codec_native_blocks(
        self,
        kv_layer,
        block_ids: list[int],
        layer_index: int,
        transfer_id: str = "",
    ) -> tuple[object, int, int, int, str, float, float] | None:
        codec_start = time.perf_counter()
        if self.codec == "splitzip_bf16":
            encoded = self._splitzip_bf16_native_blocks(
                kv_layer, block_ids, layer_index, transfer_id
            )
            if encoded is not None:
                data, raw_block_size, raw_bytes, encoded_bytes, ratio = encoded
                return (
                    data,
                    raw_block_size,
                    raw_bytes,
                    encoded_bytes,
                    "splitzip_bf16",
                    ratio,
                    time.perf_counter() - codec_start,
                )
            self._release_pending_codec_buffers()
            self._release_pending_gpu_pack_buffers()
        native = self._native_blocks_bytes(kv_layer, block_ids)
        if native is None:
            return None
        data, raw_block_size = native
        raw_bytes = raw_block_size * len(block_ids)
        return (
            data,
            raw_block_size,
            raw_bytes,
            raw_bytes,
            "raw_passthrough",
            1.0,
            time.perf_counter() - codec_start,
        )

    def _try_send_splitzip_two_chunks(
        self,
        kv_layer,
        block_ids: list[int],
        contiguous_blocks: bool,
        session: Session,
        layer_index: int,
        source: str,
    ) -> bool:
        if (
            self.splitzip_chunks != 2
            or self.codec != "splitzip_bf16"
            or self.splitzip_top16
            or self.splitzip_fixed5
            or not contiguous_blocks
            or len(block_ids) < 2
            or len(block_ids) % 2
        ):
            return False
        half = len(block_ids) // 2
        for chunk_index in range(2):
            chunk_ids = block_ids[chunk_index * half : (chunk_index + 1) * half]
            stage_start = time.perf_counter()
            self._wait_for_codec_slot()
            encoded = self._codec_native_blocks(
                kv_layer, chunk_ids, layer_index, session.transfer_id
            )
            if encoded is None or encoded[4] != "splitzip_bf16":
                if chunk_index == 0:
                    return False
                raise RuntimeError("splitzip second chunk encode failed")
            (
                data,
                raw_block_size,
                raw_bytes,
                encoded_bytes,
                codec,
                ratio,
                codec_elapsed_s,
            ) = encoded
            self.trace.event(
                "codec_encode_submitted",
                layer=layer_index,
                chunk=chunk_index,
                frame_chunks=2,
                raw_bytes=raw_bytes,
                encoded_bytes=encoded_bytes,
                ratio=ratio,
                submit_elapsed_s=codec_elapsed_s,
                source=source,
            )
            self.trace.event(
                "layer_stage_done",
                layer=layer_index,
                chunk=chunk_index,
                frame_chunks=2,
                bytes=encoded_bytes,
                raw_bytes=raw_bytes,
                chunks=len(chunk_ids),
                contiguous_blocks=True,
                layout="compressed_native",
                codec=codec,
                source=source,
                stage_elapsed_s=time.perf_counter() - stage_start,
            )
            future = self._submit_send(
                self._send_compressed_native_layer_blocks,
                session.transfer_id,
                session.request_id,
                layer_index,
                data,
                raw_block_size,
                len(chunk_ids),
                raw_bytes,
                codec,
                chunk_index,
                2,
                len(block_ids),
                raw_block_size * len(block_ids),
            )
            self._bind_stage_buffers(future)
        return True

    def _wait_for_codec_slot(self) -> None:
        wait_start = None
        while self.codec_gpu_slots > 0:
            self._check_send_futures(done_only=True)
            entries = [
                entry
                for pool in self._codec_pool.values()
                for entry in pool
            ]
            if len(entries) < self.codec_gpu_slots or any(
                not entry.reserved
                and (entry.future is None or entry.future.done())
                for entry in entries
            ):
                if wait_start is not None:
                    self.trace.event(
                        "codec_slot_wait", wait_s=time.perf_counter() - wait_start
                    )
                return
            if not self.send_futures:
                return
            if wait_start is None:
                wait_start = time.perf_counter()
            future = self.send_futures.pop(0)
            self._finish_send_future(future, done_only=False)

    def _native_tensor_for_codec(self, kv_layer, block_ids: list[int], torch):
        if not block_ids or not hasattr(kv_layer, "device"):
            return None
        shape = getattr(kv_layer, "shape", ())
        dim = kv_layer.dim() if hasattr(kv_layer, "dim") else len(shape)
        if dim < 3 or shape[0] != 2:
            return None
        span = _contiguous_span(block_ids)
        if span is not None:
            start, count = span
            return kv_layer[:, start : start + count].contiguous()
        if self.gpu_pack_layers <= 0 or not hasattr(kv_layer, "index_select"):
            if self.gpu_pack_strict:
                raise RuntimeError("strict gpu pack requires a configured pack pool")
            return None
        pack_shape = (2, len(block_ids)) + tuple(kv_layer.shape[2:])
        entry = self._wait_for_gpu_pack_buffer(
            pack_shape, kv_layer.dtype, kv_layer.device, torch
        )
        if entry is None:
            if self._early_stage_active and (
                self.early_stage_pack_only or self.gpu_pack_strict
            ):
                raise _EarlyStageSkip()
            self.trace.event("gpu_pack_full", blocks=len(block_ids))
            if self.gpu_pack_strict:
                raise RuntimeError("gpu pack capacity exceeded")
            return None
        ids = torch.as_tensor(block_ids, dtype=torch.long, device=kv_layer.device)
        pack_start = None
        stream = None
        if getattr(self, "trace_cuda_timing", False):
            stream = torch.cuda.current_stream(kv_layer.device)
            pack_start = torch.cuda.Event(enable_timing=True)
            pack_start.record(stream)
        torch.index_select(kv_layer, 1, ids, out=entry.tensor)
        if pack_start is not None:
            pack_end = torch.cuda.Event(enable_timing=True)
            pack_end.record(stream)
            self._pending_gpu_pack_events.append(
                (-1, _tensor_bytes(entry.tensor), pack_start, pack_end)
            )
        return entry.tensor

    def _trace_bf16_exponents(
        self, source, transfer_id: str, layer_index: int, torch
    ) -> None:
        key = (transfer_id, layer_index)
        if key in self._traced_bf16_exponents:
            return
        self._traced_bf16_exponents.add(key)
        try:
            counts = []
            for plane in (source[0], source[1]):
                words = plane.detach().view(torch.uint16).reshape(-1).to(torch.int32)
                exponents = torch.bitwise_and(
                    torch.bitwise_right_shift(words, 7), 0xFF
                )
                counts.append(torch.bincount(exponents, minlength=256))
            k_counts, v_counts = torch.stack(counts).cpu().tolist()
            self.trace.event(
                "bf16_exponent_stats",
                transfer=transfer_id,
                layer=layer_index,
                k=summarize_symbol_counts(k_counts),
                v=summarize_symbol_counts(v_counts),
            )
        except Exception as exc:
            self._traced_bf16_exponents.discard(key)
            try:
                self.trace.event(
                    "bf16_exponent_stats_error",
                    transfer=transfer_id,
                    layer=layer_index,
                    error=str(exc)[:160],
                )
            except Exception:
                pass

    def _should_try_splitzip_top16(
        self, layer_index: int, raw_bytes: int
    ) -> tuple[bool, int, bool]:
        if not self.splitzip_top16 or raw_bytes < 1 << 20:
            return False, -1, False
        event = None
        fields = {}
        with self._splitzip_top16_lock:
            epoch = self._splitzip_top16_epochs.get(layer_index, 0)
            if layer_index in self._splitzip_top16_unavailable_layers:
                return False, epoch, False
            remaining = self._splitzip_top16_cooldowns.get(layer_index)
            if remaining is None:
                return True, epoch, False
            if remaining > 0:
                remaining -= 1
                self._splitzip_top16_cooldowns[layer_index] = remaining
                event = "splitzip_top16_cooldown_skip"
                fields = {
                    "layer": layer_index,
                    "remaining": remaining,
                    "window": self._splitzip_top16_windows[layer_index],
                }
            elif layer_index in self._splitzip_top16_probe_layers:
                event = "splitzip_top16_probe_wait_skip"
                fields = {"layer": layer_index}
            else:
                epoch += 1
                self._splitzip_top16_epochs[layer_index] = epoch
                self._splitzip_top16_probe_layers.add(layer_index)
                event = "splitzip_top16_probe"
                fields = {
                    "layer": layer_index,
                    "window": self._splitzip_top16_windows[layer_index],
                }
                probe = True
        self.trace.event(event, **fields)
        return (True, epoch, probe) if event == "splitzip_top16_probe" else (
            False,
            epoch,
            False,
        )

    def _finish_splitzip_top16_attempt(
        self,
        layer_index: int,
        epoch: int,
        probe: bool,
        overflow: bool,
    ) -> None:
        if layer_index < 0 or epoch < 0:
            return
        event = None
        fields = {"layer": layer_index}
        with self._splitzip_top16_lock:
            if self._splitzip_top16_epochs.get(layer_index, 0) != epoch:
                event = "splitzip_top16_stale_result"
            elif overflow:
                previous = self._splitzip_top16_windows.get(layer_index, 0)
                window = (
                    min(
                        max(1, previous * 2),
                        self.splitzip_top16_max_cooldown_payloads,
                    )
                    if probe
                    else 1
                )
                self._splitzip_top16_windows[layer_index] = window
                self._splitzip_top16_cooldowns[layer_index] = window
                self._splitzip_top16_probe_layers.discard(layer_index)
                self._splitzip_top16_epochs[layer_index] = epoch + 1
                event = "splitzip_top16_overflow"
                fields.update(window=window, probe=probe)
            elif probe:
                self._splitzip_top16_windows.pop(layer_index, None)
                self._splitzip_top16_cooldowns.pop(layer_index, None)
                self._splitzip_top16_probe_layers.discard(layer_index)
                self._splitzip_top16_epochs[layer_index] = epoch + 1
                event = "splitzip_top16_probe_success"
        if event is not None:
            self.trace.event(
                event,
                **fields,
            )

    def _disable_splitzip_top16(
        self, layer_index: int, epoch: int
    ) -> None:
        if layer_index < 0 or epoch < 0:
            return
        disabled = False
        with self._splitzip_top16_lock:
            if self._splitzip_top16_epochs.get(layer_index, 0) == epoch:
                self._splitzip_top16_unavailable_layers.add(layer_index)
                self._splitzip_top16_cooldowns.pop(layer_index, None)
                self._splitzip_top16_windows.pop(layer_index, None)
                self._splitzip_top16_probe_layers.discard(layer_index)
                self._splitzip_top16_epochs[layer_index] = epoch + 1
                disabled = True
        self.trace.event(
            "splitzip_top16_unavailable"
            if disabled
            else "splitzip_top16_stale_result",
            layer=layer_index,
        )

    def _cancel_splitzip_top16_probe(
        self, layer_index: int, epoch: int, probe: bool
    ) -> None:
        if not probe:
            return
        with self._splitzip_top16_lock:
            if self._splitzip_top16_epochs.get(layer_index, 0) != epoch:
                return
            self._splitzip_top16_probe_layers.discard(layer_index)
        self.trace.event("splitzip_top16_probe_error", layer=layer_index)

    def _splitzip_bf16_native_blocks(
        self,
        kv_layer,
        block_ids: list[int],
        layer_index: int = -1,
        transfer_id: str = "",
    ) -> tuple[object, int, int, int, float] | None:
        import torch

        source = self._native_tensor_for_codec(kv_layer, block_ids, torch)
        if source is None or str(source.dtype) != "torch.bfloat16":
            return None
        if (
            self.trace_bf16_exponents
            and (transfer_id, layer_index) not in self._traced_bf16_exponents
        ):
            self._trace_bf16_exponents(source, transfer_id, layer_index, torch)
        raw_bytes = source.numel() * source.element_size()
        count = source.numel()
        prefer_native = raw_bytes >= 1 << 20 or (
            self.splitzip_top16 and self.splitzip_native_decode
        )
        entry = (
            self._wait_for_codec_buffer(raw_bytes, source.device, torch)
            if prefer_native
            else None
        )
        if prefer_native and entry is None:
            if self._early_stage_active and self._active_request_spools:
                raise _EarlyStageSkip()
            self.trace.event(
                "codec_buffer_bypass",
                requested_bytes=raw_bytes,
                committed_bytes=self._codec_pool_bytes(),
                cap_bytes=self.codec_gpu_bytes,
            )
            return None
        if entry is not None:
            requested_bits = 0
            top16_epoch = -1
            top16_probe = False
            try:
                from . import splitzip_cuda

                top16_requested = self.splitzip_top16 and raw_bytes >= 1 << 20
                try_top16, top16_epoch, top16_probe = (
                    self._should_try_splitzip_top16(layer_index, raw_bytes)
                )
                bits = (
                    4
                    if try_top16
                    else 6
                    if top16_requested
                    else 5
                    if self.splitzip_fixed5
                    and layer_index not in self._splitzip_fixed6_layers
                    and (
                        not self.splitzip_fixed5_layers
                        or layer_index in self.splitzip_fixed5_layers
                    )
                    else 6
                )
                requested_bits = bits
                fallback = False
                measure_timing = bool(
                    self.trace_cuda_timing
                    or (
                        getattr(self, "request_spool_auto_mode", "off") != "off"
                        and layer_index >= 0
                        and layer_index % 8 == 0
                    )
                )
                payload_start = (
                    time.perf_counter() if measure_timing else None
                )
                encode_start = None
                encode_stream = None
                if measure_timing:
                    encode_stream = torch.cuda.current_stream(source.device)
                    encode_start = torch.cuda.Event(enable_timing=True)
                    encode_start.record(encode_stream)
                encoded_bytes = (
                    splitzip_cuda.encode_top16(source, entry.tensor, layer_index)
                    if bits == 4
                    else splitzip_cuda.encode_bf16(source, entry.tensor, bits=5)
                    if bits == 5
                    else splitzip_cuda.encode_bf16(source, entry.tensor)
                )
                if bits in {4, 5} and not encoded_bytes:
                    bits = 6
                    fallback = True
                    if requested_bits == 4 and layer_index >= 0:
                        self._disable_splitzip_top16(layer_index, top16_epoch)
                    elif layer_index >= 0:
                        self._splitzip_fixed6_layers.add(layer_index)
                    encoded_bytes = splitzip_cuda.encode_bf16(source, entry.tensor)
                encode_timing = None
                if encode_start is not None:
                    encode_end = torch.cuda.Event(enable_timing=True)
                    encode_end.record(encode_stream)
                    encode_timing = (encode_start, encode_end)
                if fallback:
                    self.trace.event(
                        "splitzip_top16_fallback"
                        if requested_bits == 4
                        else "splitzip_fixed5_overflow",
                        layer=layer_index,
                    )
                if encoded_bytes is not None and 0 < encoded_bytes < raw_bytes:
                    writeback = (
                        self._try_splitzip_writeback(
                            kv_layer,
                            block_ids,
                            source,
                            entry,
                            encoded_bytes,
                            raw_bytes,
                            layer_index,
                            torch,
                            top16_epoch,
                            top16_probe,
                        )
                        if self.codec_writeback_overwrite
                        and bits == 4
                        and not fallback
                        and not self._early_stage_active
                        else None
                    )
                    if writeback is not None:
                        return (
                            writeback,
                            raw_bytes // len(block_ids),
                            raw_bytes,
                            encoded_bytes,
                            raw_bytes / encoded_bytes,
                        )
                    out = entry.tensor[:encoded_bytes]
                    staged = (
                        self._stage_cpu(out.detach(), measure_timing=True)
                        if measure_timing and not self.trace_cuda_timing
                        else self._stage_cpu(out.detach())
                    )
                    staged_tensor, events = self._staged_tensor_parts(staged)
                    d2h_timing = (
                        staged.d2h_timing
                        if isinstance(staged, _StagedTensor)
                        else None
                    )
                    data = memoryview(staged_tensor.numpy()).cast("B")
                    pack_buffers = self._take_pending_gpu_pack_buffers()
                    return (
                        _SplitzipCudaPayload(
                            data,
                            events,
                            source,
                            entry.tensor,
                            layer_index,
                            encode_timing,
                            d2h_timing,
                            raw_bytes,
                            encoded_bytes,
                            bits,
                            fallback,
                            payload_start,
                            codec_buffer=entry,
                            gpu_pack_buffers=pack_buffers,
                            top16_epoch=top16_epoch,
                            top16_probe=top16_probe,
                        ),
                        raw_bytes // len(block_ids),
                        raw_bytes,
                        encoded_bytes,
                        raw_bytes / encoded_bytes,
                    )
            except Exception as exc:
                if requested_bits == 4:
                    self._cancel_splitzip_top16_probe(
                        layer_index, top16_epoch, top16_probe
                    )
                self.trace.event("splitzip_cuda_fallback", error=str(exc)[:160])
            self._release_pending_codec_buffers()
        words = source.detach().view(torch.uint16).reshape(-1).to(torch.int32)
        high = torch.bitwise_right_shift(words, 8).to(torch.uint8)
        low = torch.bitwise_and(words, 0xFF).to(torch.uint8)
        high_i = high.to(torch.int16)
        sign = torch.bitwise_right_shift(high_i, 7).to(torch.uint8)
        mag = torch.bitwise_and(high_i, 0x7F).to(torch.uint8)
        mag_min = int(mag.min().cpu().item())
        mag_max = int(mag.max().cpu().item())
        mag_width = max(1, (mag_max - mag_min).bit_length())
        if mag_width < 7:
            sign_bytes = (count + 7) // 8
            mag_bytes = (count * mag_width + 7) // 8
            encoded_bytes = 3 + count + sign_bytes + mag_bytes
            if encoded_bytes < raw_bytes:
                entry = self._acquire_codec_buffer(encoded_bytes, source.device, torch)
                if entry is None:
                    self.trace.event("codec_slot_skip", blocks=len(block_ids))
                    if self._early_stage_active:
                        raise _EarlyStageSkip()
                    return None
                out = entry.tensor[:encoded_bytes]
                out[0].fill_(2)
                out[1].fill_(mag_width)
                out[2].fill_(mag_min)
                offset = 3
                out[offset : offset + count].copy_(low)
                offset += count
                self._pack_splitzip_bits(sign, 1, out[offset : offset + sign_bytes], torch)
                offset += sign_bytes
                self._pack_splitzip_bits(
                    (mag.to(torch.int16) - mag_min).to(torch.uint8),
                    mag_width,
                    out[offset:],
                    torch,
                )
                staged = self._stage_cpu(out.detach())
                staged_tensor, events = self._staged_tensor_parts(staged)
                data = memoryview(staged_tensor.numpy()).cast("B")
                return (
                    self._staged_payload(data, events),
                    raw_bytes // len(block_ids),
                    raw_bytes,
                    encoded_bytes,
                    raw_bytes / encoded_bytes,
                )

        palette, codes = torch.unique(high, sorted=True, return_inverse=True)
        width = max(1, (palette.numel() - 1).bit_length())
        if width < 8:
            code_bytes = (count * width + 7) // 8
            encoded_bytes = 258 + count + code_bytes
            if encoded_bytes < raw_bytes:
                entry = self._acquire_codec_buffer(encoded_bytes, source.device, torch)
                if entry is None:
                    self.trace.event("codec_slot_skip", blocks=len(block_ids))
                    if self._early_stage_active:
                        raise _EarlyStageSkip()
                    return None
                out = entry.tensor[:encoded_bytes]
                out[:258].zero_()
                out[1].fill_(width)
                out[2 : 2 + palette.numel()].copy_(palette)
                offset = 258
                out[offset : offset + count].copy_(low)
                offset += count
                self._pack_splitzip_bits(codes.to(torch.uint8), width, out[offset:], torch)
                staged = self._stage_cpu(out.detach())
                staged_tensor, events = self._staged_tensor_parts(staged)
                data = memoryview(staged_tensor.numpy()).cast("B")
                return (
                    self._staged_payload(data, events),
                    raw_bytes // len(block_ids),
                    raw_bytes,
                    encoded_bytes,
                    raw_bytes / encoded_bytes,
                )

        header_bytes = 69
        planes = []
        encoded_bytes = header_bytes
        for shift in (0, 4, 8, 12):
            nibble = torch.bitwise_and(
                torch.bitwise_right_shift(words, shift), 0xF
            ).to(torch.uint8)
            palette, codes = torch.unique(nibble, sorted=True, return_inverse=True)
            width = 1 if palette.numel() <= 2 else 2 if palette.numel() <= 4 else 4
            size = (count * width + 7) // 8
            planes.append((palette, codes.to(torch.uint8), width, size))
            encoded_bytes += size
        if encoded_bytes >= raw_bytes:
            return None
        entry = self._acquire_codec_buffer(encoded_bytes, source.device, torch)
        if entry is None:
            self.trace.event("codec_slot_skip", blocks=len(block_ids))
            if self._early_stage_active:
                raise _EarlyStageSkip()
            return None
        out = entry.tensor[:encoded_bytes]
        out[:header_bytes].zero_()
        out[0].fill_(1)
        out[1:5].copy_(
            torch.tensor(
                [plane[2] for plane in planes], dtype=torch.uint8, device=source.device
            )
        )
        for index, (palette, _codes, _width, _size) in enumerate(planes):
            start = 5 + index * 16
            out[start : start + palette.numel()].copy_(palette)
        offset = header_bytes
        for _palette, codes, width, size in planes:
            packed = out[offset : offset + size]
            packed.zero_()
            if width == 4:
                packed.copy_(codes[0::2])
                hi_codes = torch.bitwise_left_shift(codes[1::2].to(torch.int16), 4).to(
                    torch.uint8
                )
                if hi_codes.numel():
                    packed[: hi_codes.numel()].add_(hi_codes)
            else:
                per_byte = 8 // width
                for index in range(per_byte):
                    part = codes[index::per_byte]
                    if part.numel():
                        packed[: part.numel()].add_(
                            torch.bitwise_left_shift(
                                part.to(torch.int16), width * index
                            ).to(torch.uint8)
                        )
            offset += size
        staged = self._stage_cpu(out.detach())
        staged_tensor, events = self._staged_tensor_parts(staged)
        data = memoryview(staged_tensor.numpy()).cast("B")
        return (
            self._staged_payload(data, events),
            raw_bytes // len(block_ids),
            raw_bytes,
            encoded_bytes,
            raw_bytes / encoded_bytes,
        )

    def _pack_splitzip_bits(self, codes, width: int, out, torch) -> None:
        out.zero_()
        if not codes.numel():
            return
        if width in {1, 2, 4}:
            per_byte = 8 // width
            for index in range(per_byte):
                part = codes[index::per_byte]
                if part.numel():
                    out[: part.numel()].add_(
                        torch.bitwise_left_shift(
                            part.to(torch.int16), width * index
                        ).to(torch.uint8)
                    )
            return
        bit_offsets = torch.arange(
            codes.numel(), dtype=torch.int64, device=codes.device
        ) * int(width)
        byte_indices = torch.div(bit_offsets, 8, rounding_mode="floor")
        shifts = torch.remainder(bit_offsets, 8).to(torch.int16)
        values = torch.bitwise_left_shift(codes.to(torch.int16), shifts)
        tmp = torch.zeros((out.numel() + 1,), dtype=torch.int16, device=out.device)
        tmp.scatter_add_(0, byte_indices, torch.bitwise_and(values, 0xFF))
        tmp.scatter_add_(0, byte_indices + 1, torch.bitwise_right_shift(values, 8))
        out.copy_(tmp[: out.numel()].to(torch.uint8))

    def _unpack_splitzip_bits(self, packed, count: int, width: int, torch):
        if count <= 0:
            return torch.empty((0,), dtype=torch.uint8, device=packed.device)
        if width in {1, 2, 4}:
            codes = torch.empty((count,), dtype=torch.uint8, device=packed.device)
            mask = (1 << width) - 1
            per_byte = 8 // width
            for index in range(per_byte):
                target = codes[index::per_byte]
                if target.numel():
                    target.copy_(
                        torch.bitwise_and(
                            torch.bitwise_right_shift(
                                packed[: target.numel()].to(torch.int16),
                                width * index,
                            ),
                            mask,
                        ).to(torch.uint8)
                    )
            return codes
        bit_offsets = torch.arange(count, dtype=torch.int64, device=packed.device) * int(
            width
        )
        byte_indices = torch.div(bit_offsets, 8, rounding_mode="floor")
        shifts = torch.remainder(bit_offsets, 8).to(torch.int16)
        tmp = torch.zeros((packed.numel() + 1,), dtype=torch.int32, device=packed.device)
        tmp[: packed.numel()].copy_(packed.to(torch.int32))
        values = torch.bitwise_or(
            tmp[byte_indices],
            torch.bitwise_left_shift(tmp[byte_indices + 1], 8),
        )
        return torch.bitwise_and(
            torch.bitwise_right_shift(values, shifts), (1 << width) - 1
        ).to(torch.uint8)

    def _stage_cpu(self, tensor, measure_timing: bool = False):
        measure_timing = bool(measure_timing or self.trace_cuda_timing)
        if (
            not (self.pinned_staging or self.host_mirror_layers > 0)
            or self.transport != "tcp"
            or getattr(tensor.device, "type", "") != "cuda"
        ):
            return tensor.contiguous().cpu()
        import torch

        try:
            source = tensor.contiguous()
            entry = self._acquire_pinned_stage_buffer(source, torch)
            while (
                entry is None
                and self._active_request_spools
                and self._wait_for_spool_stage_progress()
            ):
                entry = self._acquire_pinned_stage_buffer(source, torch)
            if entry is None:
                self.trace.event("host_mirror_full")
                if self._active_request_spools:
                    if self._early_stage_active:
                        raise _EarlyStageSkip()
                    raise RuntimeError("request spool pinned ring is full")
                return source.cpu()
            stream = self._pinned_copy_stream(source.device, torch)
            current = torch.cuda.current_stream(source.device)
            with torch.cuda.stream(stream):
                stream.wait_stream(current)
                d2h_start = None
                if measure_timing:
                    d2h_start = torch.cuda.Event(enable_timing=True)
                    d2h_start.record(stream)
                entry.tensor.copy_(source, non_blocking=True)
                event = (
                    torch.cuda.Event(enable_timing=True)
                    if d2h_start is not None
                    else torch.cuda.Event()
                )
                event.record(stream)
            entry.event = event
            entry.source = source
            return _StagedTensor(
                entry.tensor,
                [event],
                (d2h_start, event) if d2h_start is not None else None,
            )
        except RuntimeError as exc:
            if self._active_request_spools:
                raise
            self.trace.event("pinned_staging_fallback", error=str(exc)[:160])
            return tensor.contiguous().cpu()

    def _try_splitzip_writeback(
        self,
        kv_layer,
        block_ids: list[int],
        source,
        codec_entry: _CodecBuffer,
        encoded_bytes: int,
        raw_bytes: int,
        layer_index: int,
        torch,
        top16_epoch: int = -1,
        top16_probe: bool = False,
    ) -> _StagedPayload | None:
        if (
            self.transport != "tcp"
            or not (self.pinned_staging or self.host_mirror_layers > 0)
            or getattr(getattr(kv_layer, "device", None), "type", "") != "cuda"
            or str(getattr(kv_layer, "dtype", "")) != "torch.bfloat16"
        ):
            return None
        block_runs = _contiguous_runs(block_ids)
        if len(block_runs) != 1:
            self.trace.event(
                "codec_writeback_skip",
                layer=layer_index,
                reason="fragmented",
                segments=len(block_runs) * 2,
            )
            return None
        segments = self._kv_writeback_segments(kv_layer, block_ids, torch)
        if not segments or sum(int(segment.numel()) for segment in segments) < raw_bytes:
            self.trace.event("codec_writeback_skip", layer=layer_index, reason="layout")
            return None
        source_start = int(source.data_ptr())
        source_end = source_start + int(source.numel()) * int(source.element_size())
        if any(
            source_start < int(segment.data_ptr()) + int(segment.numel())
            and int(segment.data_ptr()) < source_end
            for segment in segments
        ):
            self.trace.event("codec_writeback_skip", layer=layer_index, reason="alias")
            return None

        host_entry = self._wait_for_pinned_stage_buffer(
            codec_entry.tensor[:encoded_bytes], torch
        )
        if host_entry is None:
            self.trace.event("codec_writeback_skip", layer=layer_index, reason="host")
            return None

        started = time.perf_counter()
        try:
            offset = 0
            for segment in segments:
                count = min(int(segment.numel()), encoded_bytes - offset)
                if count <= 0:
                    break
                segment[:count].copy_(
                    codec_entry.tensor[offset : offset + count], non_blocking=True
                )
                offset += count
            if offset != encoded_bytes:
                raise RuntimeError("codec writeback capacity changed after preflight")
            writeback_event = torch.cuda.Event()
            writeback_event.record(torch.cuda.current_stream(source.device))
            staged = self._stage_cpu_segments(
                host_entry,
                segments,
                encoded_bytes,
                codec_entry.tensor,
                torch,
                writeback_event,
            )
        except BaseException:
            self._release_pinned_stage_buffer(host_entry)
            raise
        staged_tensor, events = self._staged_tensor_parts(staged)
        data = memoryview(staged_tensor.numpy()).cast("B")
        pack_buffers = self._take_pending_gpu_pack_buffers()
        self._release_codec_entry(codec_entry, writeback_event)
        self.trace.event(
            "codec_writeback_submitted",
            layer=layer_index,
            bytes=encoded_bytes,
            raw_bytes=raw_bytes,
            segments=len(segments),
            elapsed_s=time.perf_counter() - started,
        )
        return _SplitzipCudaPayload(
            data,
            events,
            source,
            None,
            layer_index,
            raw_bytes=raw_bytes,
            encoded_bytes=encoded_bytes,
            bits=4,
            payload_start=started,
            gpu_pack_buffers=pack_buffers,
            writeback=True,
            writeback_segments=len(segments),
            top16_epoch=top16_epoch,
            top16_probe=top16_probe,
        )

    def _finish_deferred_codec_writebacks(
        self, kv_layer, layer_index: int, sessions: list[Session]
    ) -> None:
        if not self._deferred_codec_writebacks:
            return
        import torch

        for session in sessions:
            deferred = self._deferred_codec_writebacks.pop(
                (session.transfer_id, layer_index), None
            )
            if deferred is None:
                continue
            block_ids, payload = deferred
            entry = payload.codec_buffer
            if entry is None:
                payload.deferred_writeback = False
                continue
            if not all(self._cuda_event_ready(event) for event in payload.events):
                payload.deferred_writeback = False
                self.trace.event(
                    "codec_writeback_skip",
                    layer=layer_index,
                    reason="d2h_pending",
                    source="early",
                )
                continue
            if payload.view and payload.view[0] == 255:
                payload.deferred_writeback = False
                self.trace.event(
                    "codec_writeback_skip",
                    layer=layer_index,
                    reason="overflow",
                    source="early",
                )
                continue

            runs = _contiguous_runs(block_ids)
            release_event = payload.events[-1] if payload.events else None
            stream = None
            try:
                stream = torch.cuda.current_stream(entry.tensor.device)
                for event in payload.events:
                    stream.wait_event(event)
                segments = (
                    self._kv_writeback_segments(kv_layer, block_ids, torch)
                    if len(runs) == 1
                    else None
                )
                aliases = False
                if segments and payload.source is not None:
                    source_start = int(payload.source.data_ptr())
                    source_end = source_start + int(payload.source.numel()) * int(
                        payload.source.element_size()
                    )
                    aliases = any(
                        source_start
                        < int(segment.data_ptr()) + int(segment.numel())
                        and int(segment.data_ptr()) < source_end
                        for segment in segments
                    )
                if (
                    segments
                    and not aliases
                    and sum(int(segment.numel()) for segment in segments)
                    >= payload.raw_bytes
                ):
                    offset = 0
                    for segment in segments:
                        count = min(
                            int(segment.numel()), payload.encoded_bytes - offset
                        )
                        if count <= 0:
                            break
                        segment[:count].copy_(
                            entry.tensor[offset : offset + count], non_blocking=True
                        )
                        offset += count
                    if offset != payload.encoded_bytes:
                        raise RuntimeError(
                            "deferred codec writeback capacity changed"
                        )
                    release_event = torch.cuda.Event()
                    release_event.record(stream)
                    self.trace.event(
                        "codec_writeback_submitted",
                        layer=layer_index,
                        bytes=payload.encoded_bytes,
                        raw_bytes=payload.raw_bytes,
                        segments=len(segments),
                        source="early",
                    )
                else:
                    self.trace.event(
                        "codec_writeback_skip",
                        layer=layer_index,
                        reason=(
                            "fragmented"
                            if len(runs) != 1
                            else "alias"
                            if aliases
                            else "layout"
                        ),
                        segments=len(runs) * 2,
                        source="early",
                    )
            except Exception as exc:
                if stream is not None:
                    try:
                        release_event = torch.cuda.Event()
                        release_event.record(stream)
                    except Exception:
                        pass
                self.trace.event(
                    "codec_writeback_error",
                    layer=layer_index,
                    error=str(exc)[:160],
                    source="early",
                )
            finally:
                self._release_splitzip_codec_buffer(payload, release_event)
                payload.deferred_writeback = False
                payload.source = None
                payload.out = None

    def _cancel_deferred_codec_writebacks(
        self,
        transfer_ids: set[str] | None = None,
        layer_index: int | None = None,
        reason: str = "cancelled",
    ) -> None:
        for key, (_block_ids, payload) in list(
            self._deferred_codec_writebacks.items()
        ):
            if transfer_ids is not None and key[0] not in transfer_ids:
                continue
            if layer_index is not None and key[1] != layer_index:
                continue
            del self._deferred_codec_writebacks[key]
            payload.deferred_writeback = False
            ready = all(self._cuda_event_ready(event) for event in payload.events)
            overflow = ready and bool(payload.view and payload.view[0] == 255)
            if ready and not overflow:
                event = payload.events[-1] if payload.events else None
                self._release_splitzip_codec_buffer(payload, event)
                self._release_splitzip_pack_buffers(payload, event)
                payload.source = None
                payload.out = None
            self.trace.event(
                "codec_writeback_skip",
                layer=payload.layer_index,
                reason=reason,
                source="early",
            )

    def _kv_writeback_segments(self, kv_layer, block_ids: list[int], torch):
        shape = getattr(kv_layer, "shape", ())
        dim = kv_layer.dim() if hasattr(kv_layer, "dim") else len(shape)
        if (
            not block_ids
            or dim < 3
            or shape[0] != 2
            or len(set(block_ids)) != len(block_ids)
            or any(block < 0 or block >= shape[1] for block in block_ids)
        ):
            return None
        segments = []
        for plane in range(2):
            for start, count in _contiguous_runs(block_ids):
                segment = kv_layer[plane, start : start + count]
                if not segment.is_contiguous():
                    return None
                segments.append(segment.view(torch.uint8).reshape(-1))
        return segments

    def _stage_cpu_segments(
        self,
        entry: _PinnedStageBuffer,
        segments,
        encoded_bytes: int,
        reference,
        torch,
        dependency,
    ):
        try:
            stream = self._pinned_copy_stream(reference.device, torch)
            with torch.cuda.stream(stream):
                stream.wait_event(dependency)
                d2h_start = None
                if self.trace_cuda_timing:
                    d2h_start = torch.cuda.Event(enable_timing=True)
                    d2h_start.record(stream)
                offset = 0
                for segment in segments:
                    count = min(int(segment.numel()), encoded_bytes - offset)
                    if count <= 0:
                        break
                    entry.tensor[offset : offset + count].copy_(
                        segment[:count], non_blocking=True
                    )
                    offset += count
                event = (
                    torch.cuda.Event(enable_timing=True)
                    if d2h_start is not None
                    else torch.cuda.Event()
                )
                event.record(stream)
            entry.event = event
            entry.source = segments
            return _StagedTensor(
                entry.tensor,
                [event],
                (d2h_start, event) if d2h_start is not None else None,
            )
        except RuntimeError as exc:
            self._release_pinned_stage_buffer(entry)
            self.trace.event("codec_writeback_host_error", error=str(exc)[:160])
            raise

    def _pinned_copy_stream(self, device, torch):
        key = str(device)
        stream = self._pinned_copy_streams.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._pinned_copy_streams[key] = stream
        return stream

    def _record_prefill_forward_start(self, layer: int) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                return
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream())
            self._prefill_forward_event_starts.setdefault(layer, []).append(
                (self._prefill_forward_request, event)
            )
        except Exception as exc:
            self.trace.event("producer_attn_forward_event_error", error=str(exc)[:160])

    def _record_prefill_forward_done(self, layer: int) -> None:
        starts = self._prefill_forward_event_starts.get(layer)
        if not starts:
            return
        request, start = starts.pop(0)
        try:
            import torch

            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream())
            self._prefill_forward_event_pairs.append((request, layer, start, event))
            kv_starts = self._prefill_kv_update_events.get(layer)
            if kv_starts:
                kv_request, kv_start = kv_starts.pop(0)
                self._prefill_kv_to_forward_event_pairs.append(
                    (kv_request, layer, kv_start, event)
                )
        except Exception as exc:
            self.trace.event("producer_attn_forward_event_error", error=str(exc)[:160])

    def _record_prefill_kv_update_done(self, layer: int) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                return
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream())
            self._prefill_kv_update_events.setdefault(layer, []).append(
                (self._prefill_kv_request, event)
            )
        except Exception as exc:
            self.trace.event("producer_kv_update_event_error", error=str(exc)[:160])

    def _flush_prefill_forward_events(self) -> None:
        if not self.trace_prefill_window:
            return
        pending = []
        for request, layer, start, end in self._prefill_forward_event_pairs:
            if not end.query():
                pending.append((request, layer, start, end))
                continue
            self.trace.event(
                "producer_attn_forward_gpu_done",
                request=request,
                layer=layer,
                elapsed_s=start.elapsed_time(end) / 1000.0,
            )
        self._prefill_forward_event_pairs = pending
        pending = []
        for request, layer, start, end in self._prefill_kv_to_forward_event_pairs:
            if not end.query():
                pending.append((request, layer, start, end))
                continue
            self.trace.event(
                "producer_kv_to_forward_gpu_done",
                request=request,
                layer=layer,
                elapsed_s=start.elapsed_time(end) / 1000.0,
            )
        self._prefill_kv_to_forward_event_pairs = pending

    def _flush_codec_decode_events(self) -> None:
        pending = []
        for layer, raw_bytes, start, end in self._pending_codec_decode_events:
            if not end.query():
                pending.append((layer, raw_bytes, start, end))
                continue
            self.trace.event(
                "splitzip_decode_gpu_done",
                layer=layer,
                bytes=raw_bytes,
                decode_gpu_s=start.elapsed_time(end) / 1000.0,
            )
        self._pending_codec_decode_events = pending
        pending = []
        for layer, packed_bytes, start, end in getattr(
            self, "_pending_gpu_pack_events", []
        ):
            if not end.query():
                pending.append((layer, packed_bytes, start, end))
                continue
            self.trace.event(
                "gpu_pack_done",
                layer=layer,
                bytes=packed_bytes,
                pack_gpu_s=start.elapsed_time(end) / 1000.0,
            )
        self._pending_gpu_pack_events = pending

    def _staged_tensor_parts(self, staged) -> tuple[object, list[object]]:
        if isinstance(staged, _StagedTensor):
            return staged.tensor, staged.events
        return staged, []

    def _staged_payload(self, view: memoryview, events: list[object]):
        return _StagedPayload(view, events) if events else view

    def _pinned_stage_pool_bytes(self) -> int:
        return sum(
            _tensor_bytes(entry.tensor)
            for entries in self._pinned_stage_pool.values()
            for entry in entries
        )

    def _pinned_stage_reserved_bytes(self) -> int:
        return sum(
            _tensor_bytes(entry.tensor)
            for entries in self._pinned_stage_pool.values()
            for entry in entries
            if entry.reserved
            or (entry.future is not None and not entry.future.done())
        )

    def _buffer_entry_available(self, entry) -> bool:
        if entry.reserved:
            return False
        waited_network_future = entry.future is not None
        if entry.future is not None and not entry.future.done():
            return False
        if entry.event is not None and not self._cuda_event_ready(entry.event):
            return False
        entry.future = None
        entry.event = None
        self._end_buffer_lease(
            entry, waited_network_future=waited_network_future
        )
        return True

    def _buffer_kind(self, entry) -> str:
        if isinstance(entry, _PinnedStageBuffer):
            return "pinned"
        if isinstance(entry, _GpuPackBuffer):
            return "pack"
        return "codec"

    def _begin_buffer_lease(self, entry) -> None:
        if getattr(entry, "lease_id", 0):
            raise RuntimeError(f"{self._buffer_kind(entry)} buffer already leased")
        self._buffer_lease_id = getattr(self, "_buffer_lease_id", 0) + 1
        entry.lease_id = self._buffer_lease_id
        entry.reserved = True
        self.trace.event(
            "buffer_acquire",
            kind=self._buffer_kind(entry),
            lease=entry.lease_id,
            bytes=_tensor_bytes(entry.tensor),
        )

    def _end_buffer_lease(
        self, entry, *, waited_network_future: bool = False
    ) -> None:
        lease_id = getattr(entry, "lease_id", 0)
        if not lease_id:
            return
        self.trace.event(
            "buffer_release",
            kind=self._buffer_kind(entry),
            lease=lease_id,
            bytes=_tensor_bytes(entry.tensor),
            waited_network_future=waited_network_future,
        )
        entry.lease_id = 0

    def _acquire_pinned_stage_buffer(self, tensor, torch) -> _PinnedStageBuffer | None:
        key = (tuple(tensor.shape), str(tensor.dtype))
        pool = self._pinned_stage_pool.setdefault(key, [])
        for entry in pool:
            if not self._buffer_entry_available(entry):
                continue
            entry.source = None
            self._begin_buffer_lease(entry)
            self._pending_stage_buffers.append(entry)
            return entry
        requested_bytes = _tensor_bytes(tensor)
        if self.host_mirror_bytes > 0 and requested_bytes > self.host_mirror_bytes:
            return None

        def over_cap() -> bool:
            buffers = sum(len(entries) for entries in self._pinned_stage_pool.values())
            return (
                self.host_mirror_layers > 0 and buffers >= self.host_mirror_layers
            ) or (
                self.host_mirror_bytes > 0
                and self._pinned_stage_pool_bytes() + requested_bytes
                > self.host_mirror_bytes
            )

        while over_cap():
            victims = [
                (victim_key, entry)
                for victim_key, entries in self._pinned_stage_pool.items()
                for entry in entries
                if self._buffer_entry_available(entry)
            ]
            if not victims:
                return None
            victim_key, victim_entry = max(
                victims, key=lambda item: _tensor_bytes(item[1].tensor)
            )
            self._pinned_stage_pool[victim_key].remove(victim_entry)
            if not self._pinned_stage_pool[victim_key]:
                del self._pinned_stage_pool[victim_key]
            if self.trace_enabled:
                self.trace.event(
                    "host_mirror_evict",
                    buffers=sum(
                        len(entries) for entries in self._pinned_stage_pool.values()
                    ),
                    committed_bytes=self._pinned_stage_pool_bytes(),
                )
        entry = _PinnedStageBuffer(
            torch.empty(
                tuple(tensor.shape),
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=True,
            ),
        )
        self._begin_buffer_lease(entry)
        pool.append(entry)
        self._pending_stage_buffers.append(entry)
        if self.trace_enabled:
            self.trace.event(
                "host_mirror_alloc",
                buffers=sum(
                    len(entries) for entries in self._pinned_stage_pool.values()
                ),
                committed_bytes=self._pinned_stage_pool_bytes(),
                cap_buffers=self.host_mirror_layers,
                cap_bytes=self.host_mirror_bytes,
                reserved_bytes=self._pinned_stage_reserved_bytes(),
            )
        return entry

    def _wait_for_pinned_stage_buffer(self, tensor, torch):
        requested_bytes = _tensor_bytes(tensor)
        if self.host_mirror_bytes > 0 and requested_bytes > self.host_mirror_bytes:
            self.trace.event(
                "host_credit_oversize",
                requested_bytes=requested_bytes,
                cap_bytes=self.host_mirror_bytes,
            )
            return None
        wait_start = None
        while True:
            entry = self._acquire_pinned_stage_buffer(tensor, torch)
            if entry is not None:
                if wait_start is not None:
                    self.trace.event(
                        "host_credit_wait",
                        requested_bytes=requested_bytes,
                        wait_s=time.perf_counter() - wait_start,
                    )
                return entry
            self._check_send_futures(done_only=True)
            entry = self._acquire_pinned_stage_buffer(tensor, torch)
            if entry is not None:
                if wait_start is not None:
                    self.trace.event(
                        "host_credit_wait",
                        requested_bytes=requested_bytes,
                        wait_s=time.perf_counter() - wait_start,
                    )
                return entry
            if not self.send_futures:
                return None
            if wait_start is None:
                wait_start = time.perf_counter()
            future = self.send_futures.pop(0)
            self._finish_send_future(future, done_only=False)

    def _release_pinned_stage_buffer(self, entry: _PinnedStageBuffer) -> None:
        if entry in self._pending_stage_buffers:
            self._pending_stage_buffers.remove(entry)
        entry.future = None
        entry.event = None
        entry.source = None
        self._end_buffer_lease(entry)
        entry.reserved = False

    def _acquire_gpu_pack_buffer(
        self, shape: tuple[int, ...], dtype, device, torch
    ) -> _GpuPackBuffer | None:
        key = (tuple(shape), str(dtype), str(device))
        pool = self._gpu_pack_pool.setdefault(key, [])
        for entry in pool:
            if not self._buffer_entry_available(entry):
                continue
            self._begin_buffer_lease(entry)
            self._pending_gpu_pack_buffers.append(entry)
            return entry
        pack_cap = getattr(self, "gpu_pack_bytes", 0)
        requested_bytes = 0
        if pack_cap > 0:
            itemsize = getattr(dtype, "itemsize", None)
            if not isinstance(itemsize, int):
                itemsize = torch.empty((), dtype=dtype).element_size()
            requested_bytes = math.prod(shape) * itemsize
            if requested_bytes > pack_cap:
                self.trace.event(
                    "gpu_pack_oversize",
                    requested_bytes=requested_bytes,
                    cap_bytes=pack_cap,
                    shape=list(shape),
                )
                return None

        def over_cap() -> bool:
            buffers = sum(len(entries) for entries in self._gpu_pack_pool.values())
            return (
                self.gpu_pack_layers > 0 and buffers >= self.gpu_pack_layers
            ) or (
                pack_cap > 0
                and self._gpu_pack_pool_bytes() + requested_bytes > pack_cap
            )

        victims = [
            (victim_key, entry)
            for victim_key, entries in self._gpu_pack_pool.items()
            for entry in entries
            if self._buffer_entry_available(entry)
        ]
        victims.sort(key=lambda item: _tensor_bytes(item[1].tensor), reverse=True)
        while over_cap() and victims:
            victim_key, victim = victims.pop(0)
            victim_bytes = _tensor_bytes(victim.tensor)
            self._gpu_pack_pool[victim_key].remove(victim)
            if not self._gpu_pack_pool[victim_key]:
                del self._gpu_pack_pool[victim_key]
            self.trace.event(
                "gpu_pack_evict",
                bytes=victim_bytes,
                committed_bytes=self._gpu_pack_pool_bytes(),
            )
        if over_cap():
            return None
        pool = self._gpu_pack_pool.setdefault(key, [])
        entry = _GpuPackBuffer(
            torch.empty(tuple(shape), dtype=dtype, device=device),
        )
        self._begin_buffer_lease(entry)
        pool.append(entry)
        self._pending_gpu_pack_buffers.append(entry)
        self.trace.event(
            "gpu_pack_alloc",
            shape=list(shape),
            dtype=str(dtype),
            buffers=sum(len(entries) for entries in self._gpu_pack_pool.values()),
            committed_bytes=sum(
                _tensor_bytes(current.tensor)
                for entries in self._gpu_pack_pool.values()
                for current in entries
            ),
            cap_buffers=self.gpu_pack_layers,
            cap_bytes=pack_cap,
        )
        return entry

    def _gpu_pack_pool_bytes(self) -> int:
        return sum(
            _tensor_bytes(entry.tensor)
            for entries in self._gpu_pack_pool.values()
            for entry in entries
        )

    def _wait_for_gpu_pack_buffer(self, shape, dtype, device, torch):
        entry = self._acquire_gpu_pack_buffer(shape, dtype, device, torch)
        while (
            entry is None
            and (
                getattr(self, "_active_request_spools", None)
                or getattr(self, "request_spool_enabled", False)
            )
            and not self._early_stage_active
            and self._wait_for_spool_stage_progress()
        ):
            entry = self._acquire_gpu_pack_buffer(shape, dtype, device, torch)
        if entry is not None or self._early_stage_active:
            return entry

        if getattr(self, "_payload_ready_futures", None):
            started = time.perf_counter()
            waited = 0
            while self._payload_ready_futures:
                future = self._payload_ready_futures.pop(0)
                future.result()
                waited += 1
                entry = self._acquire_gpu_pack_buffer(shape, dtype, device, torch)
                if entry is not None:
                    break
            self.trace.event(
                "gpu_pack_credit_wait",
                requested_bytes=(math.prod(shape) * getattr(dtype, "itemsize", 0)),
                wait_s=time.perf_counter() - started,
                source="payload_ready",
                futures=waited,
            )
            if entry is not None:
                return entry

        blocked = [
            current
            for entries in self._gpu_pack_pool.values()
            for current in entries
            if not current.reserved
            and current.future is None
            and current.event is not None
        ]
        if not blocked:
            if not self.send_futures:
                return None
            started = time.perf_counter()
            drained = 0
            while self.send_futures:
                future = self.send_futures.pop(0)
                self._finish_send_future(future, done_only=False)
                drained += 1
                entry = self._acquire_gpu_pack_buffer(shape, dtype, device, torch)
                if entry is not None:
                    break
            self.trace.event(
                "gpu_pack_credit_wait",
                requested_bytes=(math.prod(shape) * getattr(dtype, "itemsize", 0)),
                wait_s=time.perf_counter() - started,
                source="send_future",
                futures=drained,
            )
            return entry
        waiting = min(blocked, key=lambda current: _tensor_bytes(current.tensor))
        started = time.perf_counter()
        torch.cuda.current_stream(device).wait_event(waiting.event)
        waiting.event = None
        self.trace.event(
            "gpu_pack_credit_wait",
            requested_bytes=(math.prod(shape) * getattr(dtype, "itemsize", 0)),
            wait_s=time.perf_counter() - started,
            source="cuda_event",
        )
        return self._acquire_gpu_pack_buffer(shape, dtype, device, torch)

    def _codec_pool_bytes(self) -> int:
        return sum(
            int(entry.tensor.numel())
            for entries in self._codec_pool.values()
            for entry in entries
        )

    def _acquire_codec_buffer(self, size: int, device, torch) -> _CodecBuffer | None:
        size = int(size)
        if size <= 0 or (self.codec_gpu_bytes <= 0 and self.codec_gpu_slots <= 0):
            return None
        device_key = str(device)
        pool = self._codec_pool.setdefault(device_key, [])
        available = [
            entry
            for entry in pool
            if self._buffer_entry_available(entry)
        ]
        fitting = [entry for entry in available if int(entry.tensor.numel()) >= size]
        if fitting:
            entry = min(fitting, key=lambda current: int(current.tensor.numel()))
            entry.future = None
            self._begin_buffer_lease(entry)
            self._pending_codec_buffers.append(entry)
            return entry

        reclaimed_bytes = 0
        if self.codec_gpu_bytes > 0:
            committed = self._codec_pool_bytes()
            victims = [
                (key, entry)
                for key, entries in self._codec_pool.items()
                for entry in entries
                if self._buffer_entry_available(entry)
            ]
            victims.sort(key=lambda item: int(item[1].tensor.numel()), reverse=True)
            while committed + size > self.codec_gpu_bytes and victims:
                key, victim = victims.pop(0)
                self._codec_pool[key].remove(victim)
                victim_bytes = int(victim.tensor.numel())
                committed -= victim_bytes
                reclaimed_bytes += victim_bytes
                self.trace.event(
                    "codec_buffer_evict",
                    bytes=victim_bytes,
                    committed_bytes=committed,
                )
            available.clear()
            victim = None
            if committed + size > self.codec_gpu_bytes:
                return None
        elif (
            sum(len(entries) for entries in self._codec_pool.values())
            >= self.codec_gpu_slots
        ):
            if not available:
                return None
            victim = max(available, key=lambda current: int(current.tensor.numel()))
            pool.remove(victim)
            reclaimed_bytes = int(victim.tensor.numel())
            available.clear()
            victim = None

        try:
            tensor = torch.empty((size,), dtype=torch.uint8, device=device)
        except RuntimeError as exc:
            self.trace.event(
                "codec_buffer_alloc_failed",
                bytes=size,
                error=str(exc)[:160],
            )
            return None
        entry = _CodecBuffer(tensor)
        self._begin_buffer_lease(entry)
        pool.append(entry)
        self._pending_codec_buffers.append(entry)
        self.trace.event(
            "codec_buffer_resize" if reclaimed_bytes else "codec_buffer_alloc",
            bytes=size,
            old_bytes=reclaimed_bytes,
            committed_bytes=self._codec_pool_bytes(),
            cap_bytes=self.codec_gpu_bytes,
            device=device_key,
        )
        return entry

    def _wait_for_codec_buffer(self, size: int, device, torch):
        if (
            getattr(self, "_active_request_spools", None)
            or getattr(self, "request_spool_enabled", False)
        ):
            while True:
                entry = self._acquire_codec_buffer(size, device, torch)
                if entry is not None or not self._wait_for_spool_stage_progress():
                    return entry
        if not self.codec_writeback:
            return self._acquire_codec_buffer(size, device, torch)
        device_key = str(device)
        pool = self._codec_pool.get(device_key, [])
        if any(
            int(entry.tensor.numel()) >= size
            and self._buffer_entry_available(entry)
            for entry in pool
        ):
            return self._acquire_codec_buffer(size, device, torch)
        blocked = [
            entry
            for entry in pool
            if int(entry.tensor.numel()) >= size
            if not entry.reserved
            and entry.future is None
            and entry.event is not None
        ]
        if not blocked:
            return self._acquire_codec_buffer(size, device, torch)
        entry = min(blocked, key=lambda current: int(current.tensor.numel()))
        started = time.perf_counter()
        stream = torch.cuda.current_stream(device)
        stream.wait_event(entry.event)
        entry.event = None
        self.trace.event(
            "codec_credit_wait",
            requested_bytes=size,
            wait_s=time.perf_counter() - started,
        )
        return self._acquire_codec_buffer(size, device, torch)

    def _release_pending_gpu_pack_buffers(self, event=None) -> None:
        for entry in self._pending_gpu_pack_buffers:
            entry.future = None
            entry.event = event
            if event is None:
                self._end_buffer_lease(entry)
            entry.reserved = False
        self._pending_gpu_pack_buffers.clear()

    def _take_pending_gpu_pack_buffers(self) -> list[_GpuPackBuffer]:
        entries = self._pending_gpu_pack_buffers
        self._pending_gpu_pack_buffers = []
        return entries

    def _release_gpu_pack_buffers(self, entries, event=None) -> None:
        for entry in entries:
            entry.future = None
            entry.event = event
            if event is None:
                self._end_buffer_lease(entry)
            entry.reserved = False

    def _release_pending_codec_buffers(self, event=None) -> None:
        for entry in self._pending_codec_buffers:
            entry.future = None
            entry.event = event
            if event is None:
                self._end_buffer_lease(entry)
            entry.reserved = False
        self._pending_codec_buffers.clear()

    def _release_codec_entry(self, entry: _CodecBuffer, event=None) -> None:
        if entry in self._pending_codec_buffers:
            self._pending_codec_buffers.remove(entry)
        entry.future = None
        entry.event = event
        if event is None:
            self._end_buffer_lease(entry)
        entry.reserved = False

    def _bind_stage_buffers(self, future: Future) -> None:
        host_bytes = 0
        for entry in self._pending_stage_buffers:
            entry.future = future
            entry.reserved = False
            host_bytes += _tensor_bytes(entry.tensor)
        self._pending_stage_buffers.clear()
        for entry in self._pending_gpu_pack_buffers:
            entry.future = future
            entry.event = None
            entry.reserved = False
        self._pending_gpu_pack_buffers.clear()
        for entry in self._pending_codec_buffers:
            entry.future = future
            entry.event = None
            entry.reserved = False
        self._pending_codec_buffers.clear()
        if host_bytes:
            future._kvx_host_bytes = host_bytes
            self.trace.event(
                "host_send_bound",
                bytes=host_bytes,
                reserved_bytes=self._pinned_stage_reserved_bytes(),
            )

    def _abandon_pending_stage_buffers(self) -> None:
        for entry in self._pending_stage_buffers:
            entry.future = None
            self._end_buffer_lease(entry)
            entry.reserved = False
        self._pending_stage_buffers.clear()

    def _payload_completion_event(self, data: object):
        if isinstance(data, _ReadyPayloadFuture):
            return self._payload_completion_event(data.payload)
        if isinstance(data, (_SplitzipCudaPayload, _StagedPayload)):
            return data.events[-1] if data.events else None
        if isinstance(data, (list, tuple)):
            for item in reversed(data):
                event = self._payload_completion_event(item)
                if event is not None:
                    return event
        return None

    def _release_failed_send_payload(self, data: object) -> None:
        if isinstance(data, _ReadyPayloadFuture):
            if not data.future.cancel():
                try:
                    data.future.result()
                except BaseException:
                    pass
            self._release_failed_send_payload(data.payload)
            return
        if isinstance(data, _SplitzipCudaPayload):
            data.deferred_writeback = False
            self._cancel_splitzip_top16_probe(
                data.layer_index, data.top16_epoch, data.top16_probe
            )
            event = data.events[-1] if data.events else None
            self._release_codec_payload_buffer(data, event)
            return
        if isinstance(data, (list, tuple)):
            for item in data:
                self._release_failed_send_payload(item)

    def _cleanup_failed_send_args(self, args: tuple[object, ...]) -> None:
        event = self._payload_completion_event(args)
        self._release_failed_send_payload(args)
        self._release_pending_gpu_pack_buffers(event)
        self._release_pending_codec_buffers(event)
        self._abandon_pending_stage_buffers()

    def _release_splitzip_pack_buffers(
        self, data: _SplitzipCudaPayload, event=None
    ) -> None:
        self._release_gpu_pack_buffers(data.gpu_pack_buffers, event)
        data.gpu_pack_buffers = []

    def _release_splitzip_codec_buffer(
        self, data: _SplitzipCudaPayload, event=None
    ) -> None:
        entry = data.codec_buffer
        if entry is not None:
            self._release_codec_entry(entry, event)
        data.codec_buffer = None

    def _release_codec_payload_buffer(
        self, data: _SplitzipCudaPayload, event=None
    ) -> None:
        self._release_splitzip_codec_buffer(data, event)
        self._release_splitzip_pack_buffers(data, event)
        data.source = None
        data.out = None

    def _ready_payload(self, data: object) -> object:
        if isinstance(data, _ReadyPayloadFuture):
            return data.future.result()
        if isinstance(data, _SplitzipCudaPayload):
            try:
                return self._ready_splitzip_payload(data)
            finally:
                self._release_splitzip_pack_buffers(data)
                if not data.deferred_writeback:
                    self._release_splitzip_codec_buffer(data)
                    data.source = None
                    data.out = None
        if not isinstance(data, _StagedPayload):
            return data
        for event in data.events:
            event.synchronize()
        return data.view

    def _ready_splitzip_payload(self, data: _SplitzipCudaPayload):
        for event in data.events:
            event.synchronize()
        overflow = bool(data.view and data.view[0] == 255)
        if data.bits == 4:
            self._finish_splitzip_top16_attempt(
                data.layer_index,
                data.top16_epoch,
                data.top16_probe,
                overflow,
            )
        if data.writeback:
            self.trace.event(
                "codec_writeback_done",
                layer=data.layer_index,
                bytes=data.encoded_bytes,
                raw_bytes=data.raw_bytes,
                segments=data.writeback_segments,
                overflow=overflow,
                elapsed_s=(
                    time.perf_counter() - data.payload_start
                    if data.payload_start is not None
                    else 0.0
                ),
            )
        view = data.view
        bits = data.bits
        fallback = data.fallback
        encoded_bytes = data.encoded_bytes
        fallback_encode_timing = None
        fallback_d2h_timing = None
        if data.view and data.view[0] == 255:
            data.deferred_writeback = False
            if data.source is None:
                raise RuntimeError("splitzip cuda codec overflow")
            from . import splitzip_cuda

            out = data.out
            torch = None
            has_timing = data.d2h_timing is not None
            if out is None or has_timing:
                import torch

            if out is None:
                out = torch.empty(
                    (data.raw_bytes,),
                    dtype=torch.uint8,
                    device=data.source.device,
                )
                self.trace.event(
                    "codec_emergency_buffer_alloc",
                    layer=data.layer_index,
                    bytes=data.raw_bytes,
                )
            fallback_stream = None
            fallback_encode_start = None
            if has_timing:
                fallback_stream = torch.cuda.current_stream(data.source.device)
                fallback_encode_start = torch.cuda.Event(enable_timing=True)
                fallback_encode_start.record(fallback_stream)
            try_fixed6 = bits in {4, 5}
            if bits == 5:
                if data.layer_index >= 0:
                    self._splitzip_fixed6_layers.add(data.layer_index)
                self.trace.event("splitzip_fixed5_overflow", layer=data.layer_index)
            if try_fixed6:
                encoded = splitzip_cuda.encode_bf16(data.source, out)
                if fallback_encode_start is not None:
                    fallback_encode_end = torch.cuda.Event(enable_timing=True)
                    fallback_encode_end.record(fallback_stream)
                    fallback_encode_timing = (
                        fallback_encode_start,
                        fallback_encode_end,
                    )
                if encoded:
                    fallback_d2h_start = None
                    if fallback_stream is not None:
                        fallback_d2h_start = torch.cuda.Event(enable_timing=True)
                        fallback_d2h_start.record(fallback_stream)
                    staged = out[:encoded].detach().cpu()
                    if fallback_d2h_start is not None:
                        fallback_d2h_end = torch.cuda.Event(enable_timing=True)
                        fallback_d2h_end.record(fallback_stream)
                        fallback_d2h_timing = (
                            fallback_d2h_start,
                            fallback_d2h_end,
                        )
                    view = memoryview(staged.numpy()).cast("B")
                    encoded_bytes = encoded
                bits = 6
                fallback = True
            if view and view[0] == 255:
                raw = data.source.detach().contiguous().cpu()
                if str(raw.dtype) == "torch.bfloat16":
                    if torch is None:
                        import torch

                    raw = raw.view(torch.uint16)
                raw_view = memoryview(raw.numpy()).cast("B")
                if raw_view.nbytes != data.raw_bytes:
                    raise RuntimeError("splitzip cuda raw fallback size mismatch")
                raw_payload = bytearray(raw_view.nbytes + 1)
                raw_payload[0] = 6
                raw_payload[1:] = raw_view
                view = memoryview(raw_payload)
                bits = 16
                encoded_bytes = len(view)
                self.trace.event("splitzip_fixed6_overflow", layer=data.layer_index)
        if (
            data.encode_timing is not None
            and data.d2h_timing is not None
            and data.payload_start is not None
        ):
            for timing in (fallback_encode_timing, fallback_d2h_timing):
                if timing is not None:
                    timing[1].synchronize()
            encode_start, encode_end = data.encode_timing
            d2h_start, d2h_end = data.d2h_timing
            encode_gpu_s = encode_start.elapsed_time(encode_end) / 1000.0
            d2h_gpu_s = d2h_start.elapsed_time(d2h_end) / 1000.0
            if fallback_encode_timing is not None:
                start, end = fallback_encode_timing
                encode_gpu_s += start.elapsed_time(end) / 1000.0
            if fallback_d2h_timing is not None:
                start, end = fallback_d2h_timing
                d2h_gpu_s += start.elapsed_time(end) / 1000.0
            self._record_spool_d2h(encoded_bytes, d2h_gpu_s)
            if self.trace_cuda_timing:
                self.trace.event(
                    "splitzip_payload_ready",
                    layer=data.layer_index,
                    raw_bytes=data.raw_bytes,
                    encoded_bytes=encoded_bytes,
                    bits=bits,
                    fallback=fallback,
                    encode_gpu_s=encode_gpu_s,
                    d2h_gpu_s=d2h_gpu_s,
                    payload_elapsed_s=time.perf_counter() - data.payload_start,
                )
        return view

    def _queue_layer_group(
        self,
        transfer_id: str,
        request_id: str,
        layer_index: int,
        data: object,
        block_size: int,
        block_count: int,
        layout: str,
    ) -> bool:
        if (
            self.transport != "tcp"
            or self.layer_group_size <= 1
            or self.pinned_staging
            or self.host_mirror_layers > 0
            or self.gpu_pack_layers > 0
        ):
            return False
        key = (transfer_id, request_id, layout)
        group = self._layer_groups.setdefault(key, [])
        group.append(_LayerGroupItem(layer_index, data, block_size, block_count))
        if self._early_stage_active:
            self._early_sent_layers.add((transfer_id, layer_index))
        if len(group) >= self.layer_group_size or layer_index >= self.expected_layers - 1:
            self._flush_layer_group(key)
        return True

    def _flush_layer_groups(self) -> None:
        for key in list(self._layer_groups):
            self._flush_layer_group(key)

    def _flush_layer_group(self, key: tuple[str, str, str]) -> None:
        group = self._layer_groups.pop(key, [])
        if not group:
            return
        transfer_id, request_id, layout = key
        first = group[0]
        self._submit_send(
            self._send_layer_group_blocks,
            transfer_id,
            request_id,
            first.layer_index,
            [item.data for item in group],
            first.block_size,
            first.block_count,
            layout,
        )

    def _restore_layer(self, layer_name: str, transfer_id: str) -> None:
        layer_index = self._layer_index(layer_name)
        key = (transfer_id, layer_index)
        try:
            self._restore_layer_once(layer_name, transfer_id)
        except BaseException as exc:
            self.store.fail(exc, transfer_id)
            with self._restore_lock:
                self._restore_errors.setdefault(key, exc)
                self._restore_done.setdefault(key, threading.Event()).set()
            self.trace.event(
                "kv_restore_error",
                transfer=transfer_id,
                layer=layer_index,
                error=str(exc)[:160],
            )
            raise

    def _restore_layer_once(self, layer_name: str, transfer_id: str) -> None:
        layer_index = self._layer_index(layer_name)
        key = (transfer_id, layer_index)
        if key in self._restored_layers:
            return
        session = self.sessions.get(transfer_id)
        kv_layer = self._kv_cache_for_layer(layer_name, layer_index)
        if session is None or kv_layer is None or not session.block_ids:
            reason = (
                "no_session"
                if session is None
                else "no_kv_cache"
                if kv_layer is None
                else "no_block_ids"
            )
            self.trace.event("kv_restore_skip", layer=layer_index, reason=reason)
            return
        block_ids = (
            session.block_ids[: self.max_blocks]
            if self.max_blocks > 0
            else session.block_ids
        )
        payloads = self.store.layer_payloads(transfer_id, layer_index)
        layout = self.store.layer_layout(transfer_id, layer_index)
        offset = 0
        restored_blocks = 0
        restore_start = time.perf_counter()
        self.trace.event(
            "kv_restore_start",
            transfer=transfer_id,
            request=session.request_id,
            block_digest=_block_id_digest(block_ids),
            layer=layer_index,
            blocks=len(block_ids),
            layout=layout,
        )
        if layout == "compressed_native":
            metadata = self.store.layer_metadata(transfer_id, layer_index)
            metadata["_layer_index"] = layer_index
            codec = metadata.get("codec", "")
            copy_start = time.perf_counter()
            if codec == "raw_passthrough":
                restored_blocks, offset = self._copy_native_blocks(
                    kv_layer, block_ids, payloads
                )
            elif codec == "splitzip_bf16":
                decode_start = None
                decode_stream = None
                if (
                    self.trace_cuda_timing
                    and getattr(getattr(kv_layer, "device", None), "type", "") == "cuda"
                ):
                    import torch

                    decode_stream = torch.cuda.current_stream(kv_layer.device)
                    decode_start = torch.cuda.Event(enable_timing=True)
                    decode_start.record(decode_stream)
                chunks_in_layer = int(metadata.get("chunks_in_layer", 1))
                if (
                    chunks_in_layer == 2
                    and len(payloads) == 2
                    and len(block_ids) % 2 == 0
                ):
                    half = len(block_ids) // 2
                    raw_block_size = int(metadata.get("raw_block_size", 0))
                    for chunk_index, payload in enumerate(payloads):
                        chunk_ids = block_ids[
                            chunk_index * half : (chunk_index + 1) * half
                        ]
                        chunk_metadata = dict(metadata)
                        chunk_metadata["raw_bytes"] = raw_block_size * len(chunk_ids)
                        chunk_metadata["encoded_bytes"] = len(payload)
                        restored, copied = self._copy_splitzip_bf16_native_blocks(
                            kv_layer, chunk_ids, [payload], chunk_metadata
                        )
                        restored_blocks += restored
                        offset += copied
                else:
                    restored_blocks, offset = self._copy_splitzip_bf16_native_blocks(
                        kv_layer, block_ids, payloads, metadata
                    )
                if decode_start is not None and restored_blocks == len(block_ids):
                    decode_end = torch.cuda.Event(enable_timing=True)
                    decode_end.record(decode_stream)
                    self._pending_codec_decode_events.append(
                        (
                            layer_index,
                            offset,
                            decode_start,
                            decode_end,
                        )
                    )
            else:
                raise ValueError(f"unsupported compressed native codec: {codec}")
            copy_elapsed_s = time.perf_counter() - copy_start
            if restored_blocks != len(block_ids):
                raise RuntimeError(
                    "compressed native restore incomplete: "
                    f"restored {restored_blocks}/{len(block_ids)} blocks"
                )
            self._mark_layer_restored(transfer_id, layer_index, True)
            self._trace_restore_done(
                layer_index,
                restored_blocks,
                offset,
                layout,
                restore_start,
                0.0,
                copy_elapsed_s,
            )
            return
        first_block = self._block_view(kv_layer, block_ids[0])
        block_size = first_block.numel() * first_block.element_size()
        expected = block_size * len(block_ids)
        if layout == "kv_major":
            copy_start = time.perf_counter()
            restored_blocks, offset = self._copy_kv_major_blocks(
                kv_layer, block_ids, payloads
            )
            copy_elapsed_s = time.perf_counter() - copy_start
            self._mark_layer_restored(
                transfer_id,
                layer_index,
                restored_blocks == len(block_ids),
            )
            self._trace_restore_done(
                layer_index,
                restored_blocks,
                offset,
                layout,
                restore_start,
                0.0,
                copy_elapsed_s,
            )
            return
        if layout == "native":
            copy_start = time.perf_counter()
            restored_blocks, offset = self._copy_native_blocks(
                kv_layer, block_ids, payloads
            )
            copy_elapsed_s = time.perf_counter() - copy_start
            self._mark_layer_restored(
                transfer_id,
                layer_index,
                restored_blocks == len(block_ids),
            )
            self._trace_restore_done(
                layer_index,
                restored_blocks,
                offset,
                layout,
                restore_start,
                0.0,
                copy_elapsed_s,
            )
            return
        payload_start = time.perf_counter()
        payload = self._restore_payload(payloads)
        payload_elapsed_s = time.perf_counter() - payload_start
        copy_start = time.perf_counter()
        if len(payload) == expected:
            self._copy_blocks_bytes(kv_layer, block_ids, payload)
            offset = expected
            restored_blocks = len(block_ids)
        else:
            for block_id in block_ids:
                target = self._block_view(kv_layer, block_id)
                size = target.numel() * target.element_size()
                block = payload[offset : offset + size]
                if len(block) != size:
                    self.trace.event(
                        "kv_restore_short",
                        layer=layer_index,
                        blocks=len(block_ids),
                        restored_blocks=restored_blocks,
                        bytes=offset,
                        payload_bytes=len(payload),
                    )
                    break
                self._copy_block_bytes(target, block)
                offset += size
                restored_blocks += 1
        copy_elapsed_s = time.perf_counter() - copy_start
        self._mark_layer_restored(
            transfer_id,
            layer_index,
            restored_blocks == len(block_ids),
        )
        self._trace_restore_done(
            layer_index,
            restored_blocks,
            offset,
            layout,
            restore_start,
            payload_elapsed_s,
            copy_elapsed_s,
        )

    def _mark_layer_restored(
        self, transfer_id: str, layer_index: int, complete: bool
    ) -> None:
        if not complete:
            raise RuntimeError(f"layer {layer_index} restore incomplete")
        self._restored_layers.add((transfer_id, layer_index))
        session = self.sessions.get(transfer_id)
        self.trace.event(
            "kv_restore_verified",
            transfer=transfer_id,
            request=session.request_id if session is not None else "",
            block_digest=_block_id_digest(session.block_ids) if session is not None else "",
            layer=layer_index,
        )
        released = self.store.release_payloads(transfer_id, layer_index)
        self.trace.event(
            "load_payload_released",
            transfer=transfer_id,
            layer=layer_index,
            bytes=released,
            retained_bytes=self.store.payload_bytes(transfer_id),
        )

    def _discard_restore_state(self, transfer_id: str) -> None:
        self._restored_layers = {
            key for key in self._restored_layers if key[0] != transfer_id
        }
        with self._restore_lock:
            for states in (self._restore_done, self._restore_errors):
                for key in [key for key in states if key[0] == transfer_id]:
                    del states[key]

    def _trace_restore_done(
        self,
        layer_index: int,
        restored_blocks: int,
        offset: int,
        layout: str,
        restore_start: float,
        payload_elapsed_s: float,
        copy_elapsed_s: float,
    ) -> None:
        self.trace.event(
            "kv_restore_done",
            layer=layer_index,
            blocks=restored_blocks,
            bytes=offset,
            layout=layout,
            restore_elapsed_s=time.perf_counter() - restore_start,
            payload_elapsed_s=payload_elapsed_s,
            copy_elapsed_s=copy_elapsed_s,
        )

    def _restore_payload(self, payloads: list[object]) -> object:
        if len(payloads) == 1:
            return payloads[0]
        return b"".join(payloads)

    def _kv_cache_for_layer(self, layer_name: str, layer_index: int):
        caches = getattr(self, "kv_caches", None)
        if caches is None:
            return None
        if isinstance(caches, dict):
            if layer_name in caches:
                return caches[layer_name]
            for name, cache in caches.items():
                if self._layer_index(str(name)) == layer_index:
                    return cache
            return None
        return list(caches)[layer_index]

    def _block_view(self, kv_layer, block_id: int):
        shape = getattr(kv_layer, "shape", ())
        dim = kv_layer.dim() if hasattr(kv_layer, "dim") else len(shape)
        return (
            kv_layer[:, block_id]
            if dim >= 3 and shape[0] == 2
            else kv_layer[block_id]
        )

    def _copy_block_bytes(self, target, data: object) -> None:
        import torch

        dtype = torch.uint16 if target.dtype == torch.bfloat16 else target.dtype
        source = self._source_tensor_from_payload(torch, data, dtype).reshape(
            target.shape
        )
        if target.dtype == torch.bfloat16:
            source = source.view(torch.bfloat16)
        target.copy_(source.to(device=target.device))

    def _copy_blocks_bytes(self, kv_layer, block_ids: list[int], data: object) -> None:
        import torch

        first = self._block_view(kv_layer, block_ids[0])
        dtype = torch.uint16 if first.dtype == torch.bfloat16 else first.dtype
        shape = (len(block_ids),) + tuple(first.shape)
        source = self._source_tensor_from_payload(torch, data, dtype).reshape(shape)
        if first.dtype == torch.bfloat16:
            source = source.view(torch.bfloat16)
        source = source.to(device=kv_layer.device)
        ids = torch.as_tensor(block_ids, dtype=torch.long, device=kv_layer.device)
        kv_shape = getattr(kv_layer, "shape", ())
        dim = kv_layer.dim() if hasattr(kv_layer, "dim") else len(kv_shape)
        if dim >= 3 and kv_shape[0] == 2:
            kv_layer.index_copy_(1, ids, source.transpose(0, 1).contiguous())
        else:
            kv_layer.index_copy_(0, ids, source)

    def _copy_kv_major_blocks(
        self, kv_layer, block_ids: list[int], payloads: list[object]
    ) -> tuple[int, int]:
        import torch

        kv_shape = getattr(kv_layer, "shape", ())
        dim = kv_layer.dim() if hasattr(kv_layer, "dim") else len(kv_shape)
        first = self._block_view(kv_layer, block_ids[0])
        if dim < 3 or kv_shape[0] != 2 or len(payloads) != 2:
            return 0, 0
        part = first[0]
        dtype = torch.uint16 if part.dtype == torch.bfloat16 else part.dtype
        shape = (len(block_ids),) + tuple(part.shape)
        expected = len(block_ids) * part.numel() * part.element_size()
        if len(payloads[0]) != expected or len(payloads[1]) != expected:
            return 0, 0
        key = self._source_tensor_from_payload(torch, payloads[0], dtype).reshape(shape)
        value = self._source_tensor_from_payload(torch, payloads[1], dtype).reshape(
            shape
        )
        if part.dtype == torch.bfloat16:
            key = key.view(torch.bfloat16)
            value = value.view(torch.bfloat16)
        ids = torch.as_tensor(block_ids, dtype=torch.long, device=kv_layer.device)
        kv_layer[0].index_copy_(0, ids, key.to(device=kv_layer.device))
        kv_layer[1].index_copy_(0, ids, value.to(device=kv_layer.device))
        return len(block_ids), expected * 2

    def _copy_native_blocks(
        self, kv_layer, block_ids: list[int], payloads: list[object]
    ) -> tuple[int, int]:
        import torch

        kv_shape = getattr(kv_layer, "shape", ())
        dim = kv_layer.dim() if hasattr(kv_layer, "dim") else len(kv_shape)
        first = self._block_view(kv_layer, block_ids[0])
        if dim < 3 or kv_shape[0] != 2 or len(payloads) != 1:
            return 0, 0
        dtype = torch.uint16 if first.dtype == torch.bfloat16 else first.dtype
        shape = (2, len(block_ids)) + tuple(first.shape[1:])
        expected = len(block_ids) * first.numel() * first.element_size()
        if len(payloads[0]) != expected:
            return 0, 0
        source = self._source_tensor_from_payload(torch, payloads[0], dtype).reshape(
            shape
        )
        if first.dtype == torch.bfloat16:
            source = source.view(torch.bfloat16)
        source = source.to(device=kv_layer.device)
        span = _contiguous_span(block_ids)
        if span is not None:
            start, count = span
            kv_layer[:, start : start + count].copy_(source)
        else:
            ids = torch.as_tensor(block_ids, dtype=torch.long, device=kv_layer.device)
            kv_layer.index_copy_(1, ids, source)
        return len(block_ids), expected

    def _copy_splitzip_bf16_native_blocks(
        self,
        kv_layer,
        block_ids: list[int],
        payloads: list[object],
        metadata: dict[str, object],
    ) -> tuple[int, int]:
        import torch

        kv_shape = getattr(kv_layer, "shape", ())
        dim = kv_layer.dim() if hasattr(kv_layer, "dim") else len(kv_shape)
        first = self._block_view(kv_layer, block_ids[0])
        raw_bytes = int(metadata.get("raw_bytes", 0))
        n = raw_bytes // 2
        encoded_bytes = int(metadata.get("encoded_bytes", 0))
        expected = len(block_ids) * first.numel() * first.element_size()
        if (
            dim < 3
            or kv_shape[0] != 2
            or len(payloads) != 1
            or first.dtype != torch.bfloat16
            or raw_bytes <= 0
            or raw_bytes % 2
            or raw_bytes != expected
        ):
            return 0, 0
        payload = memoryview(payloads[0]).cast("B")
        if encoded_bytes < 1 or len(payload) != encoded_bytes:
            return 0, 0
        mode = int(payload[0])
        if mode not in range(7):
            return 0, 0
        if mode == 3 and encoded_bytes != 65 + n + ((n + 3) // 4) * 3:
            return 0, 0
        if mode == 5 and encoded_bytes != 5 + n + (n + 1) // 2 + ((n + 199) // 200) * 5:
            return 0, 0
        comp = self._source_tensor_from_payload(torch, payload, torch.uint8).to(
            device=kv_layer.device
        )
        if comp.numel() != encoded_bytes or encoded_bytes < 1:
            return 0, 0
        if self.splitzip_native_decode and mode in {3, 5}:
            try:
                from . import splitzip_cuda

                copied = (
                    splitzip_cuda.decode_top16(
                        comp,
                        kv_layer,
                        block_ids,
                        raw_bytes,
                        int(metadata.get("_layer_index", -1)),
                    )
                    if mode == 5
                    else splitzip_cuda.decode_fixed6(
                        comp, kv_layer, block_ids, raw_bytes
                    )
                )
                if copied:
                    self.trace.event(
                        "splitzip_native_decode",
                        blocks=len(block_ids),
                        bytes=copied,
                        mode=mode,
                    )
                    return len(block_ids), copied
            except Exception as exc:
                self.trace.event(
                    "splitzip_native_decode_fallback", error=str(exc)[:160]
                )
        if mode == 0:
            header_bytes = 258
            if encoded_bytes < header_bytes:
                return 0, 0
            width = int(comp[1].cpu().item())
            if not (1 <= width < 8):
                return 0, 0
            code_bytes = (n * width + 7) // 8
            if encoded_bytes != header_bytes + n + code_bytes:
                return 0, 0
            palette = comp[2:258]
            low = comp[header_bytes : header_bytes + n].to(torch.int32)
            packed = comp[header_bytes + n :]
            codes = self._unpack_splitzip_bits(packed, n, width, torch)
            high = palette[codes.long()].to(torch.int32)
            words = torch.bitwise_or(torch.bitwise_left_shift(high, 8), low)
        elif mode == 1:
            header_bytes = 69
            if encoded_bytes < header_bytes:
                return 0, 0
            widths = [int(value) for value in comp[1:5].cpu().tolist()]
            if any(width not in {1, 2, 4} for width in widths):
                return 0, 0
            plane_sizes = [(n * width + 7) // 8 for width in widths]
            if encoded_bytes != header_bytes + sum(plane_sizes):
                return 0, 0
            words = torch.zeros((n,), dtype=torch.int32, device=kv_layer.device)
            offset = header_bytes
            for plane, (width, size) in enumerate(zip(widths, plane_sizes, strict=True)):
                palette = comp[5 + plane * 16 : 5 + (plane + 1) * 16]
                packed = comp[offset : offset + size]
                codes = torch.empty((n,), dtype=torch.uint8, device=kv_layer.device)
                if width == 4:
                    codes[0::2].copy_(torch.remainder(packed, 16))
                    high_count = n // 2
                    if high_count:
                        codes[1::2].copy_(
                            torch.bitwise_right_shift(
                                packed[:high_count].to(torch.int16), 4
                            ).to(torch.uint8)
                        )
                else:
                    mask = (1 << width) - 1
                    per_byte = 8 // width
                    for index in range(per_byte):
                        target = codes[index::per_byte]
                        if target.numel():
                            target.copy_(
                                torch.bitwise_and(
                                    torch.bitwise_right_shift(
                                        packed[: target.numel()].to(torch.int16),
                                        width * index,
                                    ),
                                    mask,
                                ).to(torch.uint8)
                            )
                nibble = palette[codes.long()].to(torch.int32)
                words = torch.bitwise_or(
                    words, torch.bitwise_left_shift(nibble, plane * 4)
                )
                offset += size
        elif mode == 2:
            header_bytes = 3
            if encoded_bytes < header_bytes:
                return 0, 0
            width = int(comp[1].cpu().item())
            mag_min = int(comp[2].cpu().item())
            if not (1 <= width < 7):
                return 0, 0
            sign_bytes = (n + 7) // 8
            mag_bytes = (n * width + 7) // 8
            if encoded_bytes != header_bytes + n + sign_bytes + mag_bytes:
                return 0, 0
            offset = header_bytes
            low = comp[offset : offset + n].to(torch.int32)
            offset += n
            sign = self._unpack_splitzip_bits(
                comp[offset : offset + sign_bytes], n, 1, torch
            ).to(torch.int32)
            offset += sign_bytes
            mag = self._unpack_splitzip_bits(
                comp[offset : offset + mag_bytes], n, width, torch
            ).to(torch.int32)
            high = torch.bitwise_or(torch.bitwise_left_shift(sign, 7), mag + mag_min)
            words = torch.bitwise_or(torch.bitwise_left_shift(high, 8), low)
        elif mode == 3:
            header_bytes = 65
            if encoded_bytes < header_bytes:
                return 0, 0
            code_bytes = ((n + 3) // 4) * 3
            if encoded_bytes != header_bytes + n + code_bytes:
                return 0, 0
            palette = comp[1:65]
            low = comp[header_bytes : header_bytes + n].to(torch.int32)
            packed = comp[header_bytes + n :]
            codes = self._unpack_splitzip_bits(packed, n, 6, torch)
            high = palette[codes.long()].to(torch.int32)
            words = torch.bitwise_or(torch.bitwise_left_shift(high, 8), low)
        elif mode == 4:
            header_bytes = 33
            if encoded_bytes < header_bytes:
                return 0, 0
            code_bytes = (n * 5 + 7) // 8
            if encoded_bytes != header_bytes + n + code_bytes:
                return 0, 0
            palette = comp[1:33]
            low = comp[header_bytes : header_bytes + n].to(torch.int32)
            packed = comp[header_bytes + n :]
            codes = self._unpack_splitzip_bits(packed, n, 5, torch)
            high = palette[codes.long()].to(torch.int32)
            words = torch.bitwise_or(torch.bitwise_left_shift(high, 8), low)
        elif mode == 5:
            from . import splitzip_cuda

            layer_index = int(metadata.get("_layer_index", -1))
            top16_layers = len(splitzip_cuda._TOP16_CODEBOOKS) // 32
            if not 0 <= layer_index < top16_layers:
                return 0, 0
            codebooks = splitzip_cuda._TOP16_CODEBOOKS[
                layer_index * 32 : (layer_index + 1) * 32
            ]
            raw = _decode_splitzip_top16_payload(payload, raw_bytes, codebooks)
            if raw is None:
                return 0, 0
            words = self._source_tensor_from_payload(
                torch, raw, torch.uint16
            ).to(device=kv_layer.device)
        elif mode == 6:
            if encoded_bytes != raw_bytes + 1:
                return 0, 0
            words = self._source_tensor_from_payload(
                torch, payload[1:], torch.uint16
            ).to(device=kv_layer.device)
        else:
            return 0, 0
        words = words.to(torch.uint16)
        shape = (2, len(block_ids)) + tuple(first.shape[1:])
        source = words.view(torch.bfloat16).reshape(shape)
        span = _contiguous_span(block_ids)
        if span is not None:
            start, count = span
            kv_layer[:, start : start + count].copy_(source)
        else:
            ids = torch.as_tensor(block_ids, dtype=torch.long, device=kv_layer.device)
            kv_layer.index_copy_(1, ids, source)
        return len(block_ids), raw_bytes

    def _use_compressed_native_codec(self, block_ids: list[int]) -> bool:
        return (
            self.codec in {"raw_passthrough", "splitzip_bf16"}
            and self.transport == "tcp"
            and self.native_layout_payload
            and len(block_ids) >= self.codec_min_blocks
        )

    def _source_tensor_from_payload(self, torch, data: object, dtype):
        view = memoryview(data)
        if not view.c_contiguous:
            view = memoryview(view.tobytes())
        if view.ndim != 1 or view.format != "B":
            view = view.cast("B")
        if view.readonly:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The given buffer is not writable",
                    category=UserWarning,
                )
                return torch.frombuffer(view, dtype=dtype)
        return torch.frombuffer(view, dtype=dtype)

    def _send_layer_blocks(
        self,
        transfer_id: str,
        request_id: str,
        layer_index: int,
        data: object,
        block_size: int,
        block_count: int,
    ) -> None:
        from .tcp_data import send_layer_blocks

        ready_start = time.perf_counter()
        data = self._ready_payload(data)
        ready_wait_s = time.perf_counter() - ready_start
        self.trace.event(
            "layer_send_ready",
            layer=layer_index,
            bytes=block_size * block_count,
            chunks=block_count,
            layout="block_major",
            ready_wait_s=ready_wait_s,
        )

        send_start = time.perf_counter()
        send_layer_blocks(
            self.peer,
            transfer_id,
            request_id,
            layer_index,
            data,
            block_size,
            block_count,
        )
        self.trace.event(
            "layer_send_done",
            layer=layer_index,
            chunks=block_count,
            bytes=block_size * block_count,
            layout="block_major",
            send_elapsed_s=time.perf_counter() - send_start,
        )

    def _send_layer_group_blocks(
        self,
        transfer_id: str,
        request_id: str,
        first_layer: int,
        payloads: list[object],
        block_size: int,
        block_count: int,
        layout: str,
    ) -> None:
        from .tcp_data import send_layer_group_blocks

        ready_start = time.perf_counter()
        ready_payloads = [self._ready_payload(payload) for payload in payloads]
        ready_wait_s = time.perf_counter() - ready_start
        self.trace.event(
            "layer_group_send_ready",
            layer=first_layer,
            layers=len(payloads),
            bytes=block_size * block_count * len(payloads),
            chunks=block_count,
            layout=layout,
            ready_wait_s=ready_wait_s,
        )

        send_start = time.perf_counter()
        send_layer_group_blocks(
            self.peer,
            transfer_id,
            request_id,
            first_layer,
            ready_payloads,
            block_size,
            block_count,
            layout=layout,
        )
        self.trace.event(
            "layer_group_send_done",
            layer=first_layer,
            layers=len(payloads),
            chunks=block_count,
            bytes=block_size * block_count * len(payloads),
            layout=layout,
            send_elapsed_s=time.perf_counter() - send_start,
        )

    def _send_kv_layer_blocks(
        self,
        transfer_id: str,
        request_id: str,
        layer_index: int,
        key_data: object,
        value_data: object,
        part_size: int,
        block_count: int,
    ) -> None:
        from .tcp_data import send_kv_layer_blocks

        ready_start = time.perf_counter()
        key_data = self._ready_payload(key_data)
        value_data = self._ready_payload(value_data)
        ready_wait_s = time.perf_counter() - ready_start
        self.trace.event(
            "layer_send_ready",
            layer=layer_index,
            bytes=part_size * block_count * 2,
            chunks=block_count,
            layout="kv_major",
            ready_wait_s=ready_wait_s,
        )

        send_start = time.perf_counter()
        send_kv_layer_blocks(
            self.peer,
            transfer_id,
            request_id,
            layer_index,
            key_data,
            value_data,
            part_size,
            block_count,
        )
        self.trace.event(
            "layer_send_done",
            layer=layer_index,
            chunks=block_count,
            bytes=part_size * block_count * 2,
            layout="kv_major",
            send_elapsed_s=time.perf_counter() - send_start,
        )

    def _send_native_layer_blocks(
        self,
        transfer_id: str,
        request_id: str,
        layer_index: int,
        data: object,
        block_size: int,
        block_count: int,
    ) -> None:
        from .tcp_data import send_native_layer_blocks

        ready_start = time.perf_counter()
        data = self._ready_payload(data)
        ready_wait_s = time.perf_counter() - ready_start
        self.trace.event(
            "layer_send_ready",
            layer=layer_index,
            bytes=block_size * block_count,
            chunks=block_count,
            layout="native",
            ready_wait_s=ready_wait_s,
        )

        send_start = time.perf_counter()
        send_native_layer_blocks(
            self.peer,
            transfer_id,
            request_id,
            layer_index,
            data,
            block_size,
            block_count,
        )
        self.trace.event(
            "layer_send_done",
            layer=layer_index,
            chunks=block_count,
            bytes=block_size * block_count,
            layout="native",
            send_elapsed_s=time.perf_counter() - send_start,
        )

    def _send_compressed_native_layer_blocks(
        self,
        transfer_id: str,
        request_id: str,
        layer_index: int,
        data: object,
        raw_block_size: int,
        block_count: int,
        raw_bytes: int,
        codec: str,
        chunk_index: int = 0,
        chunks_in_layer: int = 1,
        total_block_count: int | None = None,
        total_raw_bytes: int | None = None,
    ) -> None:
        from .tcp_data import send_compressed_native_layer_blocks

        ready_start = time.perf_counter()
        data = self._ready_payload(data)
        ready_wait_s = time.perf_counter() - ready_start
        encoded_bytes = len(data)
        self.trace.event(
            "layer_send_ready",
            layer=layer_index,
            bytes=encoded_bytes,
            raw_bytes=raw_bytes,
            chunks=block_count,
            layout="compressed_native",
            codec=codec,
            chunk=chunk_index,
            frame_chunks=chunks_in_layer,
            ready_wait_s=ready_wait_s,
        )

        send_start = time.perf_counter()
        send_compressed_native_layer_blocks(
            self.peer,
            transfer_id,
            request_id,
            layer_index,
            data,
            raw_block_size=raw_block_size,
            block_count=total_block_count or block_count,
            raw_bytes=total_raw_bytes or raw_bytes,
            codec=codec,
            chunk_index=chunk_index,
            chunks_in_layer=chunks_in_layer,
        )
        send_elapsed_s = time.perf_counter() - send_start
        self._record_spool_network(encoded_bytes, send_elapsed_s)
        self.trace.event(
            "layer_send_done",
            layer=layer_index,
            chunks=block_count,
            bytes=encoded_bytes,
            raw_bytes=raw_bytes,
            layout="compressed_native",
            codec=codec,
            chunk=chunk_index,
            frame_chunks=chunks_in_layer,
            send_elapsed_s=send_elapsed_s,
        )

    def _check_send_futures(self, done_only: bool) -> None:
        self._payload_ready_futures = [
            future for future in self._payload_ready_futures if not future.done()
        ]
        self._drain_pending_sends(ready_only=done_only)
        pending = []
        error = None
        for future in self.send_futures:
            try:
                if not self._finish_send_future(future, done_only=done_only):
                    pending.append(future)
            except BaseException as exc:
                if error is None:
                    error = exc
        self.send_futures = pending
        if error is not None:
            raise error

    def _submit_send(self, fn, *args) -> Future:
        try:
            args = self._stage_ready_send_args(args)
            self._drain_pending_sends(ready_only=True)
            self._wait_send_credit()
            if self.send_executor is None:
                raise RuntimeError("send executor is not configured")
            if not self._grant_enabled() and not self._send_args_ready(args):
                future = Future()
                future._kvx_pending_send = (fn, args)
                future._kvx_send_args = args
                self._pending_send_futures.append(future)
                self.send_futures.append(future)
                if self._early_stage_active:
                    self._early_sent_layers.add((str(args[0]), int(args[2])))
                return future
            future = self.send_executor.submit(fn, *args)
        except BaseException:
            self._cleanup_failed_send_args(args)
            raise
        future._kvx_send_args = args
        self.send_futures.append(future)
        if self._early_stage_active:
            self._early_sent_layers.add((str(args[0]), int(args[2])))
        return future

    def _stage_ready_send_args(self, args: tuple[object, ...]) -> tuple[object, ...]:
        if self.payload_ready_executor is None:
            return args

        def stage(data: object) -> object:
            if isinstance(data, _SplitzipCudaPayload) and data.gpu_pack_buffers:
                future = self.payload_ready_executor.submit(
                    self._materialize_ready_payload, data
                )
                self._payload_ready_futures.append(future)
                return _ReadyPayloadFuture(future, data)
            if isinstance(data, tuple):
                return tuple(stage(item) for item in data)
            if isinstance(data, list):
                return [stage(item) for item in data]
            return data

        return tuple(stage(arg) for arg in args)

    def _materialize_ready_payload(self, data: _SplitzipCudaPayload) -> object:
        started = time.perf_counter()
        try:
            ready = self._ready_payload(data)
        except BaseException as exc:
            self.trace.event(
                "payload_ready_failed",
                layer=data.layer_index,
                error=str(exc)[:160],
            )
            raise
        self.trace.event(
            "payload_ready_done",
            layer=data.layer_index,
            bytes=len(ready),
            elapsed_s=time.perf_counter() - started,
        )
        return ready

    def _drain_pending_sends(self, ready_only: bool) -> None:
        pending = []
        for future in self._pending_send_futures:
            if future.done():
                continue
            if ready_only and not self._future_send_ready(future):
                pending.append(future)
                continue
            self._launch_pending_send(future)
        self._pending_send_futures = pending

    def _future_send_ready(self, future: Future) -> bool:
        pending = getattr(future, "_kvx_pending_send", None)
        return pending is None or self._send_args_ready(pending[1])

    def _launch_pending_send(self, future: Future) -> None:
        pending = getattr(future, "_kvx_pending_send", None)
        if pending is None:
            return
        del future._kvx_pending_send
        fn, args = pending
        try:
            actual = self.send_executor.submit(fn, *args)
        except BaseException as exc:
            self._cleanup_failed_send_args(args)
            future.set_exception(exc)
            return

        def finish(actual_future: Future) -> None:
            if future.cancelled():
                return
            try:
                future.set_result(actual_future.result())
            except BaseException as exc:
                future.set_exception(exc)

        actual.add_done_callback(finish)

    def _send_args_ready(self, args: tuple[object, ...]) -> bool:
        return all(self._send_payload_ready(arg) for arg in args)

    def _send_payload_ready(self, data: object) -> bool:
        if isinstance(data, _ReadyPayloadFuture):
            return data.future.done()
        if isinstance(data, _SplitzipCudaPayload):
            ready = all(self._cuda_event_ready(event) for event in data.events)
            if not ready:
                return False
            if data.view and data.view[0] == 255:
                data.deferred_writeback = False
                return True
            self._release_splitzip_pack_buffers(data)
            if not data.deferred_writeback:
                self._release_splitzip_codec_buffer(data)
                data.source = None
                data.out = None
            return ready
        if isinstance(data, _StagedPayload):
            return all(self._cuda_event_ready(event) for event in data.events)
        if isinstance(data, (list, tuple)):
            return all(self._send_payload_ready(item) for item in data)
        return True

    def _cuda_event_ready(self, event: object) -> bool:
        query = getattr(event, "query", None)
        if query is None:
            return True
        try:
            return bool(query())
        except Exception:
            return False

    def _wait_send_credit(self) -> None:
        if self.send_inflight <= 0:
            return
        wait_start = None
        while True:
            self._check_send_futures(done_only=True)
            if len(self.send_futures) < self.send_inflight:
                elapsed = (
                    time.perf_counter() - wait_start
                    if wait_start is not None
                    else 0.0
                )
                if getattr(self, "request_spool_auto_mode", "off") != "off":
                    with self._spool_condition:
                        self._spool_send_waits.append(elapsed)
                if wait_start is not None:
                    self.trace.event(
                        "send_credit_wait",
                        pending=len(self.send_futures),
                        wait_s=elapsed,
                    )
                return
            if wait_start is None:
                wait_start = time.perf_counter()
            future = self.send_futures.pop(0)
            self._finish_send_future(future, done_only=False)

    def _finish_send_future(self, future: Future, done_only: bool) -> bool:
        if done_only and not future.done():
            return False
        if not done_only:
            self._launch_pending_send(future)
        try:
            future.result()
        except BaseException:
            args = getattr(future, "_kvx_send_args", ())
            self._cleanup_failed_send_args(args)
            raise
        finally:
            if hasattr(future, "_kvx_send_args"):
                del future._kvx_send_args
        host_bytes = getattr(future, "_kvx_host_bytes", 0)
        if host_bytes:
            self.trace.event(
                "host_send_released",
                bytes=host_bytes,
                reserved_bytes=self._pinned_stage_reserved_bytes(),
            )
            del future._kvx_host_bytes
        return True

    def _grant_enabled(self) -> bool:
        return self.grant_window > 0

    def _ensure_restore_thread(self) -> None:
        if not self._active_restore_ahead or self.restore_thread is not None:
            return
        if not self._active_sessions or not self._restore_ahead_pending():
            return
        self.restore_thread = threading.Thread(target=self._restore_ahead, daemon=True)
        self.restore_thread.start()

    def _restore_ahead_pending(self) -> bool:
        limit = min(self.expected_layers - 1, self._restore_ahead_limit)
        # ponytail: scan active sessions x ahead layers; add a pending counter
        # only if this bounded check appears in a profile.
        return any(
            not self._restore_event(session.transfer_id, layer).is_set()
            for session in self._active_sessions.values()
            for layer in range(limit + 1)
        )

    def _restore_ahead_for_wait(self, sessions) -> bool:
        if self._active_sessions:
            return self._active_restore_ahead
        return self._should_restore_ahead(sessions)

    def _should_restore_ahead(self, sessions) -> bool:
        return self.restore_ahead

    def _session_block_count(self, session: Session) -> int | None:
        if not session.block_ids:
            return None
        if self.max_blocks > 0:
            return min(len(session.block_ids), self.max_blocks)
        return len(session.block_ids)

    def _restore_ahead(self) -> None:
        self.trace.event(
            "restore_ahead_worker_start",
            active_sessions=len(self._active_sessions),
            limit=min(self.expected_layers - 1, self._restore_ahead_limit),
        )
        try:
            while True:
                sessions = list(self._active_sessions.values())
                progress = False
                pending = False
                limit = min(self.expected_layers - 1, self._restore_ahead_limit)
                for layer_index in range(limit + 1):
                    layer_name = f"model.layers.{layer_index}"
                    for session in sessions:
                        key = (session.transfer_id, layer_index)
                        event = self._restore_event(session.transfer_id, layer_index)
                        if event.is_set():
                            continue
                        pending = True
                        if not self.store.is_ready(session.transfer_id, layer_index):
                            continue
                        try:
                            self._restore_layer(layer_name, session.transfer_id)
                        except BaseException as exc:
                            self._restore_errors[key] = exc
                            self.trace.event(
                                "kv_restore_ahead_error",
                                layer=layer_index,
                                error=str(exc),
                            )
                        finally:
                            event.set()
                            progress = True
                if not pending and limit >= self.expected_layers - 1:
                    return
                if not progress:
                    time.sleep(0.001)
        finally:
            self.trace.event(
                "restore_ahead_worker_done",
                active_sessions=len(self._active_sessions),
                limit=min(self.expected_layers - 1, self._restore_ahead_limit),
            )
            if self.restore_thread is threading.current_thread():
                self.restore_thread = None

    def _allow_restore_through(self, layer_index: int) -> None:
        self._restore_ahead_limit = max(
            self._restore_ahead_limit, min(layer_index, self.expected_layers - 1)
        )

    def _restore_event(
        self, transfer_id: str, layer_index: int
    ) -> threading.Event:
        key = (transfer_id, layer_index)
        with self._restore_lock:
            return self._restore_done.setdefault(key, threading.Event())

    def _wait_restore_done(self, transfer_id: str, layer_index: int) -> None:
        event = self._restore_event(transfer_id, layer_index)
        if not event.wait(self.wait_timeout_s):
            raise TimeoutError(f"layer {layer_index} not restored for {transfer_id}")
        error = self._restore_errors.get((transfer_id, layer_index))
        if error is not None:
            raise error

    def _ensure_grant_receiver(self) -> None:
        if not self._grant_enabled() or self.grant_receiver is not None:
            return
        self.grant_receiver = make_receiver(
            self.bind, self.store, trace=self.trace, grants=self.grants
        )
        self.grant_receiver_thread = threading.Thread(
            target=self.grant_receiver.serve, daemon=True
        )
        self.grant_receiver_thread.start()

    def _ensure_prefix_receiver(self) -> None:
        if (
            not self.prefix_transfer_suppression
            or self.kv_role != "kv_producer"
            or self.prefix_receiver is not None
        ):
            return
        self.prefix_receiver = make_receiver(
            self.prefix_bind, self.store, trace=self.trace, grants=self.grants
        )
        self.prefix_receiver_thread = threading.Thread(
            target=self.prefix_receiver.serve, daemon=True
        )
        self.prefix_receiver_thread.start()

    def _send_layer_grant(self, layer_index: int, transfer_id: str) -> None:
        if layer_index < 0 or layer_index >= self.expected_layers:
            return
        key = (transfer_id, layer_index)
        if transport_name() == "tcp" and key in self._traced_grants_sent:
            return
        send_grant(self.peer, layer_index, transfer_id=transfer_id)
        if key not in self._traced_grants_sent:
            self._traced_grants_sent.add(key)
            self.trace.event("grant_sent", layer=layer_index, transfer=transfer_id)

    def _wait_for_grant(self, layer_index: int, transfer_id: str) -> None:
        self.trace.event(
            "grant_wait_start", layer=layer_index, transfer=transfer_id
        )
        self.grants.wait(
            layer_index, self.wait_timeout_s, transfer_id=transfer_id
        )
        self.trace.event(
            "grant_wait_done", layer=layer_index, transfer=transfer_id
        )

    def _send_restore_ack(self, transfer_id: str, layer_index: int) -> None:
        send_grant(
            self.peer,
            layer_index,
            kind="restore_ack",
            transfer_id=transfer_id,
        )
        self.trace.event(
            "restore_ack_sent",
            layer=layer_index,
            transfer=transfer_id,
        )

    def _wait_restore_ack(self, transfer_id: str, layer_index: int) -> None:
        self.trace.event(
            "restore_ack_wait_start",
            layer=layer_index,
            transfer=transfer_id,
        )
        self.grants.wait(
            layer_index,
            self.wait_timeout_s,
            kind="restore_ack",
            transfer_id=transfer_id,
        )
        self.trace.event(
            "restore_ack_wait_done",
            layer=layer_index,
            transfer=transfer_id,
        )

    def _restore_ack_ready(self, transfer_id: str, layer_index: int) -> bool:
        return self.grants.has(
            layer_index,
            kind="restore_ack",
            transfer_id=transfer_id,
        )

    def _sessions_to_load(self) -> list[Session]:
        if self._load_sessions:
            return list(self._load_sessions.values())
        if self._active_sessions:
            return list(self._active_sessions.values())
        if self.sessions:
            return list(self.sessions.values())
        transfer_id = self._active_transfer_id or self.transfer_id
        if transfer_id:
            return [Session(transfer_id, self._active_request_id, [])]
        return []

    def _layer_index(self, layer_name: str) -> int:
        digits = "".join(ch if ch.isdigit() else " " for ch in layer_name).split()
        return int(digits[-1]) if digits else 0
