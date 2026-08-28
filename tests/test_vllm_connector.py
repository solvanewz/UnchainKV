import json
import math
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from zlib import crc32

from unchain_kv.vllm_connector import UnchainKVConnector, PipelineMetadata, Session
from unchain_kv.protocol import Chunk, ChunkHeader
import unchain_kv.vllm_connector as vllm_connector


class VllmConnectorTest(unittest.TestCase):
    def test_codec_min_blocks_bypasses_short_requests_only(self):
        connector = object.__new__(UnchainKVConnector)
        connector.codec = "splitzip_bf16"
        connector.transport = "tcp"
        connector.native_layout_payload = True
        connector.codec_min_blocks = 64

        self.assertFalse(connector._use_compressed_native_codec(list(range(63))))
        self.assertTrue(connector._use_compressed_native_codec(list(range(64))))

    def test_codec_min_blocks_bypasses_short_request_early_stage(self):
        connector = object.__new__(UnchainKVConnector)
        connector.early_stage = True
        connector._early_stage_active = False
        connector.transport = "tcp"
        connector.codec_min_blocks = 64
        connector._early_sent_layers = set()
        connector._grant_enabled = lambda: False
        connector.trace = SimpleNamespace(event=lambda *_args, **_kwargs: None)
        calls = []
        connector.save_kv_layer = lambda *args: calls.append(args)
        short = Session("short", "short", list(range(63)))
        long = Session("long", "long", list(range(64)))
        connector.metadata = PipelineMetadata([short])

        connector._try_early_stage("model.layers.0", 0, object())
        self.assertEqual(calls, [])

        metadata = PipelineMetadata([short, long])
        connector.metadata = metadata
        connector._try_early_stage("model.layers.0", 0, object())
        self.assertEqual(calls, [])
        self.assertIs(connector.metadata, metadata)
        self.assertNotIn(("short", 0), connector._early_sent_layers)

        connector.metadata = PipelineMetadata([long])
        connector._try_early_stage("model.layers.0", 0, object())
        self.assertEqual(len(calls), 1)
        self.assertIn(("long", 0), connector._early_sent_layers)

    def _spool_connector(self, cap=1024, layers=2):
        env = {
            "UNCHAIN_KV_TRACE_ENABLED": "0",
            "UNCHAIN_KV_TRANSPORT": "tcp",
            "UNCHAIN_KV_TCP_LIB": "/tmp/libunchain_kv_tcp.so",
            "UNCHAIN_KV_CODEC": "splitzip_bf16",
            "UNCHAIN_KV_SPLITZIP_TOP16": "1",
            "UNCHAIN_KV_SPLITZIP_NATIVE_DECODE": "1",
            "UNCHAIN_KV_CODEC_WRITEBACK": "1",
            "UNCHAIN_KV_NATIVE_LAYOUT_PAYLOAD": "1",
            "UNCHAIN_KV_HOST_MIRROR_LAYERS": "4",
            "UNCHAIN_KV_HOST_MIRROR_BYTES": "1024",
            "UNCHAIN_KV_SEND_INFLIGHT": "1",
            "UNCHAIN_KV_SEND_WORKERS": "1",
            "UNCHAIN_KV_MAX_BLOCKS": "0",
            "UNCHAIN_KV_EXPECTED_LAYERS": str(layers),
            "UNCHAIN_KV_REQUEST_SPOOL_BYTES": str(cap),
        }
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
        )
        with patch.dict(os.environ, env, clear=True):
            return UnchainKVConnector(config, "worker")

    def test_host_memory_headroom_uses_cgroup_and_memavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meminfo = root / "meminfo"
            cgroup = root / "cgroup"
            cgroup.mkdir()
            meminfo.write_text("MemAvailable:       8192 kB\n")
            (cgroup / "memory.max").write_text(str(16 << 20))
            (cgroup / "memory.current").write_text(str(12 << 20))

            available, maximum, current, headroom = (
                vllm_connector._host_memory_headroom(meminfo, cgroup)
            )

        self.assertEqual(available, 8 << 20)
        self.assertEqual(maximum, 16 << 20)
        self.assertEqual(current, 12 << 20)
        self.assertEqual(headroom, 4 << 20)

    def test_adaptive_spool_target_is_zero_until_active_and_bounded(self):
        target = vllm_connector._adaptive_spool_target(
            batch_bound=1000,
            live_cap=10_000,
            compression_ratio=0.75,
            fill_bytes_s=2000,
            network_bytes_s=1000,
            sender_gap_s=0.1,
        )

        self.assertGreaterEqual(target, 1000)
        self.assertLessEqual(target, 10_000)
        self.assertEqual(
            vllm_connector._adaptive_spool_target(
                1000, 999, 0.75, 2000, 1000, 0.1
            ),
            0,
        )

    def test_auto_spool_requires_three_pressure_windows(self):
        connector = object.__new__(UnchainKVConnector)
        connector.request_spool_auto_mode = "auto"
        connector._spool_condition = threading.Condition()
        connector._spool_hard_bytes = 10_000
        connector._spool_live_bytes = 10_000
        connector._spool_target_bytes = 0
        connector._spool_auto_active = False
        connector._spool_last_control = 0.0
        connector._spool_pressure_windows = 0
        connector._spool_clear_windows = 0
        connector._spool_shrink_windows = 0
        connector._spool_compression_ratio = 0.75
        connector._spool_ready_samples = vllm_connector.deque(
            [1000.0] * 8, maxlen=32
        )
        connector._spool_network_rates = vllm_connector.deque(
            [500.0] * 8, maxlen=32
        )
        connector._spool_d2h_rates = vllm_connector.deque(
            [2000.0] * 2, maxlen=32
        )
        connector._spool_send_waits = vllm_connector.deque(
            [0.01] * 8, maxlen=32
        )
        connector._spool_sender_gaps = vllm_connector.deque(maxlen=32)
        connector._spool_reserved_bytes = 0
        connector._spool_resident_bytes = 0
        connector.spool_live_cap_file = None
        connector.spool_pressure_ratio = 1.15
        connector._refresh_spool_live_cap = lambda: 10_000
        rows = []
        connector.trace = SimpleNamespace(
            event=lambda name, **fields: rows.append((name, fields))
        )

        connector._update_auto_spool_control(0)
        self.assertEqual(connector._spool_pressure_windows, 0)
        self.assertEqual(rows, [])
        for expected in (False, False, True):
            connector._spool_last_control = 0.0
            connector._update_auto_spool_control(1000)
            self.assertEqual(connector._spool_auto_active, expected)

        self.assertEqual(connector._spool_target_bytes, 1000)
        self.assertEqual(rows[-1][0], "spool_auto_decision")

    def test_live_cap_cannot_drop_below_resident_spool(self):
        connector = object.__new__(UnchainKVConnector)
        connector.host_mirror_bytes = 0
        connector.host_guard_bytes = 0
        connector._spool_condition = threading.Condition()
        connector._spool_resident_bytes = 96 << 20
        connector._spool_hard_bytes = 256 << 20
        connector._pinned_stage_pool_bytes = lambda: 0
        with tempfile.TemporaryDirectory() as tmp:
            cap = Path(tmp) / "live-cap.txt"
            cap.write_text(str(64 << 20), encoding="utf-8")
            connector.spool_live_cap_file = cap
            with patch.object(
                vllm_connector,
                "_host_memory_headroom",
                return_value=(1 << 30, None, None, 1 << 30),
            ):
                live = connector._refresh_spool_live_cap()

        self.assertEqual(live, 96 << 20)

    def test_auto_spool_nonblocking_admission_is_batch_atomic(self):
        connector = self._spool_connector(cap=100)
        connector._spool_layer_block_bytes = [10, 20]
        connector._spool_reserved_bytes = 50

        admitted = connector._admit_request_spools(
            [Session("tx-a", "a", [1]), Session("tx-b", "b", [2])],
            cap_bytes=100,
            wait=False,
        )

        self.assertEqual(admitted, [])
        self.assertEqual(connector._spool_reserved_bytes, 50)
        self.assertEqual(connector._request_spools, {})
        connector.spool_executor.shutdown(wait=True)
        connector.send_executor.shutdown(wait=True)

    def test_auto_spool_configures_joint_host_and_gpu_caps(self):
        connector = object.__new__(UnchainKVConnector)
        connector.host_guard_bytes = 1 << 30
        connector.host_mirror_bytes = 512 << 20
        connector.request_spool_bytes = 0
        connector.auto_spool_hard_bytes = 2 << 30
        connector.max_blocks = 1
        connector.max_model_len = 0
        connector.block_size_tokens = 16
        connector._spool_layer_block_bytes = [64 << 20, 64 << 20]
        connector.gpu_pack_bytes = 0
        connector.gpu_guard_bytes = 512 << 20
        connector.codec_gpu_bytes = 0
        connector.codec_gpu_slots = 0
        connector._spool_capacity_ready = False
        connector.trace = SimpleNamespace(event=lambda *_args, **_kwargs: None)
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                mem_get_info=lambda _device: (2 << 30, 4 << 30)
            )
        )
        cache = SimpleNamespace(device="cuda:0")

        with patch.object(
            vllm_connector,
            "_host_memory_headroom",
            return_value=(4 << 30, None, None, 4 << 30),
        ), patch.dict(sys.modules, {"torch": fake_torch}):
            connector._configure_auto_spool_capacity([cache])

        self.assertEqual(connector._spool_hard_bytes, 2 << 30)
        self.assertEqual(connector._spool_pack_hard_bytes, 64 << 20)
        self.assertLessEqual(
            connector.codec_gpu_bytes + connector.gpu_pack_bytes,
            connector._spool_gpu_aux_bytes,
        )
        self.assertTrue(connector._spool_capacity_ready)

    def test_auto_spool_keeps_readiness_worker_while_inactive(self):
        env = {
            "UNCHAIN_KV_TRACE_ENABLED": "0",
            "UNCHAIN_KV_TRANSPORT": "tcp",
            "UNCHAIN_KV_TCP_LIB": "/tmp/libunchain_kv_tcp.so",
            "UNCHAIN_KV_CODEC": "splitzip_bf16",
            "UNCHAIN_KV_SPLITZIP_TOP16": "1",
            "UNCHAIN_KV_SPLITZIP_NATIVE_DECODE": "1",
            "UNCHAIN_KV_CODEC_WRITEBACK": "1",
            "UNCHAIN_KV_NATIVE_LAYOUT_PAYLOAD": "1",
            "UNCHAIN_KV_HOST_MIRROR_LAYERS": "4",
            "UNCHAIN_KV_HOST_MIRROR_BYTES": str(1 << 30),
            "UNCHAIN_KV_SEND_INFLIGHT": "8",
            "UNCHAIN_KV_SEND_WORKERS": "1",
            "UNCHAIN_KV_EXPECTED_LAYERS": "2",
            "UNCHAIN_KV_REQUEST_SPOOL_AUTO": "1",
            "UNCHAIN_KV_EXTENT_ALLOC": "prefer",
            "UNCHAIN_KV_GPU_PACK_LAYERS": "1",
            "UNCHAIN_KV_GPU_PACK_BYTES": str(64 << 20),
            "UNCHAIN_KV_GPU_PACK_STRICT": "1",
        }
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
            model_config=SimpleNamespace(max_model_len=32768),
        )

        with patch.dict(os.environ, env, clear=True):
            connector = UnchainKVConnector(config, "worker")

        self.assertTrue(connector.request_spool_capable)
        self.assertFalse(connector.request_spool_enabled)
        self.assertTrue(connector.payload_ready_enabled)
        self.assertEqual(connector.max_model_len, 32768)
        self.assertEqual(connector._spool_target_bytes, 0)
        connector.spool_executor.shutdown(wait=True)
        connector.payload_ready_executor.shutdown(wait=True)
        connector.send_executor.shutdown(wait=True)

    def test_auto_spool_short_batch_skips_capacity_control(self):
        connector = object.__new__(UnchainKVConnector)
        connector.request_spool_capable = True
        connector.request_spool_fixed = False
        connector._spool_capacity_ready = True
        connector.codec_min_blocks = 64
        connector._active_request_spools = [object()]
        connector.kv_role = "kv_consumer"
        connector.connector_role = "worker"
        checks = []
        controls = []
        rows = []
        connector._check_send_futures = lambda done_only: checks.append(done_only)
        connector._update_auto_spool_control = lambda bound: controls.append(bound)
        connector.trace = SimpleNamespace(
            event=lambda name, **fields: rows.append((name, fields))
        )

        connector.bind_connector_metadata(
            PipelineMetadata([Session("short", "req", list(range(63)))])
        )

        self.assertEqual(checks, [True])
        self.assertEqual(controls, [])
        self.assertEqual(connector._active_request_spools, [])
        self.assertEqual(rows[-1][1]["reason"], "short_fast_path")

    def test_fixed_spool_short_request_neither_enters_nor_waits_on_spool(self):
        connector = self._spool_connector()
        connector.codec_min_blocks = 64
        connector._spool_layer_block_bytes = [10, 20]
        checks = []
        connector._check_send_futures = lambda done_only: checks.append(done_only)

        connector.bind_connector_metadata(
            PipelineMetadata([Session("short", "req", list(range(63)))])
        )
        connector.wait_for_save()

        self.assertEqual(connector._active_request_spools, [])
        self.assertEqual(connector._request_spools, {})
        self.assertEqual(checks, [True, False])
        connector.spool_executor.shutdown(wait=True)
        connector.send_executor.shutdown(wait=True)

    def test_top16_cpu_fallback_reconstructs_exact_words(self):
        decoder = getattr(vllm_connector, "_decode_splitzip_top16_payload", None)
        self.assertIsNotNone(decoder)
        if decoder is None:
            return

        import unchain_kv.splitzip_cuda as splitzip_cuda

        count = 256
        codebooks = splitzip_cuda._TOP16_CODEBOOKS[:32]
        words = []
        body = bytearray(count)
        codes = bytearray((count + 1) // 2)
        escape_position = 3
        escape_exponent = 100
        for index in range(count):
            plane = index >= count // 2
            code = index % 16
            exponent = codebooks[plane * 16 + code]
            if index == escape_position:
                exponent = escape_exponent
                code = 0
            word = ((index % 2) << 15) | (exponent << 7) | (index % 128)
            words.append(word)
            body[index] = (word & 0x7F) | ((word >> 8) & 0x80)
            codes[index // 2] |= code << ((index % 2) * 4)
        escape_capacity = (count + 199) // 200
        payload = bytearray(5 + count + len(codes) + escape_capacity * 5)
        payload[0] = 5
        payload[1:5] = (1).to_bytes(4, "little")
        payload[5 : 5 + count] = body
        payload[5 + count : 5 + count + len(codes)] = codes
        entry = 5 + count + len(codes)
        payload[entry : entry + 4] = escape_position.to_bytes(4, "little")
        payload[entry + 4] = escape_exponent

        decoded = decoder(payload, count * 2, codebooks)

        expected = b"".join(word.to_bytes(2, "little") for word in words)
        self.assertEqual(decoded, expected)

    def test_contiguous_span(self):
        self.assertEqual(vllm_connector._contiguous_span([3, 4, 5]), (3, 3))
        self.assertIsNone(vllm_connector._contiguous_span([3, 5]))
        self.assertIsNone(vllm_connector._contiguous_span([]))

    def test_contiguous_runs_returns_tuple_for_descriptor_cache_keys(self):
        self.assertEqual(
            vllm_connector._contiguous_runs([1, 2, 5]),
            ((1, 2), (5, 1)),
        )

    def test_block_run_aggregation_off_stages_each_block(self):
        with patch.dict(
            os.environ,
            {"UNCHAIN_KV_TRACE_ENABLED": "0", "UNCHAIN_KV_BLOCK_RUNS": "0"},
            clear=True,
        ):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        seen = []
        connector._block_bytes = lambda _layer, block: seen.append(block) or bytes([block])

        data, block_size = connector._blocks_bytes(object(), [1, 2, 5])

        self.assertEqual(connector._block_runs([1, 2, 5]), ((1, 1), (2, 1), (5, 1)))
        self.assertEqual((data, block_size, seen), (b"\x01\x02\x05", 1, [1, 2, 5]))

    def test_instantiates_without_vllm_package(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )

            connector = UnchainKVConnector(config, "kv_consumer")

            self.assertEqual(connector.kv_role, "kv_consumer")
            self.assertIsNone(connector.kv_cache_config)
            self.assertFalse(connector.prefer_cross_layer_blocks)
            self.assertEqual(connector._layer_index("model.layers.12.self_attn"), 12)
            self.assertTrue(UnchainKVConnector.requires_piecewise_for_cudagraph({}))
            request = SimpleNamespace(
                kv_transfer_params={"do_remote_prefill": True},
                prompt_token_ids=[1, 2, 3],
            )
            self.assertEqual(
                connector.get_num_new_matched_tokens(request, 0),
                (3, True),
            )
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_trace_flags_default_to_enabled_writer_without_diagnostics(self):
        with patch.dict(os.environ, {}, clear=True):
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )

            connector = UnchainKVConnector(config, "kv_consumer")

        self.assertTrue(connector.trace_enabled)
        self.assertFalse(connector.trace_cuda_timing)
        self.assertFalse(connector.trace_bf16_exponents)

    def test_codec_writeback_eligibility_is_fail_closed_and_strict(self):
        env = {
            "UNCHAIN_KV_TRACE_ENABLED": "0",
            "UNCHAIN_KV_TRANSPORT": "tcp",
            "UNCHAIN_KV_CODEC": "splitzip_bf16",
            "UNCHAIN_KV_SPLITZIP_TOP16": "1",
            "UNCHAIN_KV_SPLITZIP_NATIVE_DECODE": "1",
            "UNCHAIN_KV_EARLY_STAGE": "1",
            "UNCHAIN_KV_HOST_MIRROR_LAYERS": "1",
            "UNCHAIN_KV_HOST_MIRROR_BYTES": "1048576",
            "UNCHAIN_KV_SEND_INFLIGHT": "1",
        }
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
        )
        with patch.dict(os.environ, env, clear=True):
            connector = UnchainKVConnector(config, "worker")
            self.assertTrue(connector.codec_writeback_requested)
            self.assertTrue(connector.codec_writeback)
            self.assertEqual(connector.codec_writeback_ineligible, [])
            connector.send_executor.shutdown(wait=True)

        preserved = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
            cache_config=SimpleNamespace(enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
        )
        with patch.dict(os.environ, env, clear=True):
            connector = UnchainKVConnector(preserved, "worker")
            self.assertTrue(connector.codec_writeback)
            self.assertFalse(connector.codec_writeback_overwrite)
            self.assertTrue(connector.codec_writeback_preserve_source)
            self.assertEqual(connector.codec_writeback_ineligible, [])
            connector.send_executor.shutdown(wait=True)

        strict_env = dict(env, UNCHAIN_KV_CODEC_WRITEBACK_STRICT="1")
        with patch.dict(os.environ, strict_env, clear=True):
            connector = UnchainKVConnector(preserved, "worker")
            connector.send_executor.shutdown(wait=True)

        unknown = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
            cache_config=SimpleNamespace(),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
        )
        with patch.dict(os.environ, strict_env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "prefix_cache"):
                UnchainKVConnector(unknown, "worker")

    def test_flush_codec_decode_events_keeps_pending_events_without_sync(self):
        class FakeEvent:
            def __init__(self, ready, elapsed_ms=0.0):
                self.ready = ready
                self.elapsed_ms = elapsed_ms
                self.synchronize_calls = 0

            def query(self):
                return self.ready

            def elapsed_time(self, _end):
                return self.elapsed_ms

            def synchronize(self):
                self.synchronize_calls += 1

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            with patch.dict(os.environ, {"UNCHAIN_KV_TRACE": str(trace_path)}):
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
                )
                connector = UnchainKVConnector(config, "worker")
                start = FakeEvent(True, 12.5)
                end = FakeEvent(False)
                connector._pending_codec_decode_events.append((3, 4096, start, end))

                connector._flush_codec_decode_events()

                self.assertEqual(len(connector._pending_codec_decode_events), 1)
                self.assertFalse(trace_path.exists())
                end.ready = True

                connector._flush_codec_decode_events()

                rows = [
                    json.loads(line)
                    for line in trace_path.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(connector._pending_codec_decode_events, [])
                self.assertEqual(
                    [(row["event"], row["layer"], row["bytes"], row["decode_gpu_s"])
                    for row in rows],
                    [("splitzip_decode_gpu_done", 3, 4096, 0.0125)],
                )
                self.assertEqual(start.synchronize_calls, 0)
                self.assertEqual(end.synchronize_calls, 0)

    def test_splitzip_restore_cuda_timing_queues_only_enabled_splitzip(self):
        class FakeEvent:
            def __init__(self):
                self.streams = []

            def record(self, stream):
                self.streams.append(stream)

        stream = object()
        cache_device = SimpleNamespace(type="cuda")

        def current_stream(device):
            self.assertIs(device, cache_device)
            return stream

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                Event=lambda enable_timing: FakeEvent(),
                current_stream=current_stream,
            )
        )

        def connector_for(codec, trace_cuda, device=cache_device):
            with patch.dict(
                os.environ,
                {
                    "UNCHAIN_KV_TRACE_CUDA": str(trace_cuda),
                    "UNCHAIN_KV_MAX_BLOCKS": "0",
                },
            ):
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
                )
                connector = UnchainKVConnector(config, "worker")
            connector.sessions["tx"] = Session("tx", "req", [1, 2])
            connector.kv_caches = {
                "model.layers.2.self_attn": SimpleNamespace(device=device)
            }
            connector.store.add(
                Chunk(ChunkHeader("tx", "req", 2, 0, 0, 1, 0, 3, 0), b"zip"),
                layout="compressed_native",
                metadata={
                    "block_count": 2,
                    "raw_block_size": 3,
                    "raw_bytes": 6,
                    "encoded_bytes": 3,
                    "codec": codec,
                },
            )
            return connector

        with patch.dict(sys.modules, {"torch": fake_torch}):
            enabled = connector_for("splitzip_bf16", 1)
            enabled._copy_splitzip_bf16_native_blocks = lambda *_args: (2, 6)
            enabled._restore_layer("model.layers.2.self_attn", "tx")

        self.assertEqual(len(enabled._pending_codec_decode_events), 1)
        layer, raw_bytes, start, end = enabled._pending_codec_decode_events[0]
        self.assertEqual((layer, raw_bytes), (2, 6))
        self.assertEqual(start.streams, [stream])
        self.assertEqual(end.streams, [stream])

        with patch.dict(sys.modules, {"torch": fake_torch}):
            partial = connector_for("splitzip_bf16", 1)
            events = []
            partial.trace.event = lambda name, **_fields: events.append(name)
            partial._copy_splitzip_bf16_native_blocks = lambda *_args: (0, 0)
            with self.assertRaisesRegex(RuntimeError, "restore incomplete"):
                partial._restore_layer("model.layers.2.self_attn", "tx")
            partial._flush_codec_decode_events()

        self.assertEqual(partial._pending_codec_decode_events, [])
        self.assertNotIn("splitzip_decode_gpu_done", events)
        self.assertNotIn(("tx", 2), partial._restored_layers)

        raw = connector_for("raw_passthrough", 1)
        raw._copy_native_blocks = lambda *_args: (2, 6)
        raw._restore_layer("model.layers.2.self_attn", "tx")
        self.assertEqual(raw._pending_codec_decode_events, [])

        disabled = connector_for("splitzip_bf16", 0)
        disabled._copy_splitzip_bf16_native_blocks = lambda *_args: (2, 6)
        disabled._restore_layer("model.layers.2.self_attn", "tx")
        self.assertEqual(disabled._pending_codec_decode_events, [])

    def test_splitzip_restore_cuda_timing_skips_cpu_cache(self):
        with patch.dict(
            os.environ,
            {"UNCHAIN_KV_TRACE_CUDA": "1", "UNCHAIN_KV_MAX_BLOCKS": "0"},
        ):
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
        connector.sessions["tx"] = Session("tx", "req", [1, 2])
        connector.kv_caches = {
            "model.layers.2.self_attn": SimpleNamespace(
                device=SimpleNamespace(type="cpu")
            )
        }
        connector.store.add(
            Chunk(ChunkHeader("tx", "req", 2, 0, 0, 1, 0, 3, 0), b"zip"),
            layout="compressed_native",
            metadata={
                "block_count": 2,
                "raw_block_size": 3,
                "raw_bytes": 6,
                "encoded_bytes": 3,
                "codec": "splitzip_bf16",
            },
        )
        connector._copy_splitzip_bf16_native_blocks = lambda *_args: (2, 6)

        connector._restore_layer("model.layers.2.self_attn", "tx")

        self.assertEqual(connector._pending_codec_decode_events, [])

    def test_connector_lifecycle_methods_flush_codec_decode_events(self):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
        )
        connector = UnchainKVConnector(config, "worker")
        with patch.object(connector, "_flush_codec_decode_events") as flush:
            connector.save_kv_layer("model.layers.0.self_attn", object(), None)
            connector.wait_for_layer_load("model.layers.0.self_attn")
            connector.get_finished()

        self.assertEqual(flush.call_count, 3)

    def test_trace_writer_can_be_disabled_through_connector(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            with patch.dict(
                os.environ,
                {
                    "UNCHAIN_KV_TRACE": str(trace_path),
                    "UNCHAIN_KV_TRACE_ENABLED": "0",
                },
                clear=True,
            ):
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
                )
                connector = UnchainKVConnector(config, "kv_consumer")
                connector.trace.event("ignored")

            self.assertIsNone(connector.trace.path)
            self.assertFalse(trace_path.exists())

    def test_builds_worker_metadata_from_allocated_request(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")
            request = SimpleNamespace(
                request_id="req",
                kv_transfer_params={
                    "do_remote_decode": True,
                    "transfer_id": "tx",
                },
            )

            connector.update_state_after_alloc(request, [1, 2], 0)
            metadata = connector.build_connector_meta(
                SimpleNamespace(
                    scheduled_new_reqs=[
                        SimpleNamespace(req_id="req", block_ids=[[3, 4]])
                    ]
                )
            )

            self.assertIsInstance(metadata, PipelineMetadata)
            self.assertEqual(metadata.requests[0].transfer_id, "tx")
            self.assertEqual(metadata.requests[0].block_ids, [1, 2])
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_chunked_producer_defers_until_final_chunk_and_accumulates_blocks(self):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
            cache_config=SimpleNamespace(enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        )
        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(config, "scheduler")
        connector._ensure_prefix_receiver = lambda: None
        request = SimpleNamespace(
            request_id="req",
            kv_transfer_params={"do_remote_decode": True, "transfer_id": "tx"},
        )
        connector.update_state_after_alloc(request, [1, 2], 0)
        connector.grants.add(0, kind="prefix_tokens", transfer_id="tx")

        first = connector.build_connector_meta(
            SimpleNamespace(
                scheduled_new_reqs=[
                    SimpleNamespace(
                        req_id="req",
                        prompt_token_ids=list(range(6)),
                        block_ids=([1, 2],),
                        num_computed_tokens=0,
                    )
                ],
                scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
                num_scheduled_tokens={"req": 4},
            )
        )
        self.assertEqual(first.requests, [])
        self.assertIn("req", connector._requests_need_save)

        final = connector.build_connector_meta(
            SimpleNamespace(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=SimpleNamespace(
                    req_ids=["req"],
                    resumed_req_ids=set(),
                    new_block_ids=[([3],)],
                    num_computed_tokens=[4],
                    num_output_tokens=[0],
                    all_token_ids={},
                ),
                num_scheduled_tokens={"req": 2},
            )
        )

        self.assertEqual(final.requests, [Session("tx", "req", [1, 2, 3])])
        self.assertNotIn("req", connector._requests_need_save)
        self.assertNotIn("req", connector._producer_prompt_tokens)

    def test_chunked_prefix_hit_can_send_complete_new_request_immediately(self):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
            cache_config=SimpleNamespace(enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        )
        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(config, "scheduler")
        connector._ensure_prefix_receiver = lambda: None
        request = SimpleNamespace(
            request_id="req",
            kv_transfer_params={"do_remote_decode": True, "transfer_id": "tx"},
        )
        connector.update_state_after_alloc(request, [99], 0)
        connector.grants.add(0, kind="prefix_tokens", transfer_id="tx")

        metadata = connector.build_connector_meta(
            SimpleNamespace(
                scheduled_new_reqs=[
                    SimpleNamespace(
                        req_id="req",
                        prompt_token_ids=list(range(6)),
                        block_ids=([10, 11],),
                        num_computed_tokens=4,
                    )
                ],
                scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
                num_scheduled_tokens={"req": 2},
            )
        )

        self.assertEqual(metadata.requests, [Session("tx", "req", [10, 11])])

    def test_prefix_transfer_suppression_uses_consumer_block_boundary(self):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_consumer"),
            cache_config=SimpleNamespace(enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        )
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))
            ]
        )
        with patch.dict(
            os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True
        ), patch.object(vllm_connector, "send_grant") as send:
            connector = UnchainKVConnector(config, "scheduler", kv_cache_config)
            for prefix_tokens in (32, 34):
                request = SimpleNamespace(
                    request_id=f"req-{prefix_tokens}",
                    kv_transfer_params={
                        "do_remote_prefill": True,
                        "transfer_id": f"tx-{prefix_tokens}",
                    },
                    prompt_token_ids=list(range(48)),
                )
                self.assertEqual(
                    connector.get_num_new_matched_tokens(request, prefix_tokens),
                    (48 - prefix_tokens, True),
                )
                if prefix_tokens == 34:
                    connector._consumer_prefix_tokens.clear()
                connector.update_state_after_alloc(
                    request, [10, 11, 12], 48 - prefix_tokens
                )
                self.assertEqual(
                    connector.sessions[f"tx-{prefix_tokens}"].block_ids, [12]
                )

            full = SimpleNamespace(
                request_id="req-full",
                kv_transfer_params={
                    "do_remote_prefill": True,
                    "transfer_id": "tx-full",
                },
                prompt_token_ids=list(range(48)),
            )
            self.assertEqual(
                connector.get_num_new_matched_tokens(full, 48), (0, True)
            )
            connector.update_state_after_alloc(full, [20, 21, 22], 0)
            self.assertNotIn("tx-full", connector.sessions)
            self.assertNotIn("req-full", connector._requests_need_load)
            self.assertIn("tx-full", connector._prefix_tokens_sent)
            connector.get_finished({"req-full"})
            self.assertNotIn("tx-full", connector._prefix_tokens_sent)
            self.assertNotIn("req-full", connector._consumer_prefix_transfers)

        self.assertEqual(send.call_count, 3)
        send.assert_any_call(
            connector.prefix_peer,
            34,
            kind="prefix_tokens",
            transfer_id="tx-34",
        )

        restored = []

        def copy_blocks(_kv_layer, block_ids, _payloads):
            restored.extend(block_ids)
            return len(block_ids), len(block_ids)

        connector.kv_caches = {"model.layers.0.self_attn": object()}
        connector._copy_native_blocks = copy_blocks
        connector.store.add(
            Chunk(ChunkHeader("tx-34", "req-34", 0, 0, 0, 1, 0, 1, 0), b"x"),
            layout="compressed_native",
            metadata={"codec": "raw_passthrough"},
        )
        connector._restore_layer("model.layers.0.self_attn", "tx-34")
        self.assertEqual(restored, [12])

    def test_full_prefix_hit_sends_grant_from_alloc_fallback(self):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_consumer"),
            cache_config=SimpleNamespace(enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        )
        request = SimpleNamespace(
            request_id="req-full",
            kv_transfer_params={
                "do_remote_prefill": True,
                "transfer_id": "tx-full",
            },
            prompt_token_ids=list(range(48)),
        )
        with patch.dict(
            os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True
        ), patch.object(vllm_connector, "send_grant") as send:
            connector = UnchainKVConnector(config, "scheduler")
            connector.update_state_after_alloc(request, [20, 21, 22], 0)

        send.assert_called_once_with(
            connector.prefix_peer,
            48,
            kind="prefix_tokens",
            transfer_id="tx-full",
        )
        self.assertNotIn("tx-full", connector.sessions)

    def test_zero_provisional_prefix_waits_for_final_alloc_value(self):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_consumer"),
            cache_config=SimpleNamespace(enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        )
        request = SimpleNamespace(
            request_id="req",
            kv_transfer_params={
                "do_remote_prefill": True,
                "transfer_id": "tx",
            },
            prompt_token_ids=list(range(48)),
        )
        with patch.dict(
            os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True
        ), patch.object(vllm_connector, "send_grant") as send:
            connector = UnchainKVConnector(config, "scheduler")
            self.assertEqual(
                connector.get_num_new_matched_tokens(request, 0),
                (48, True),
            )
            send.assert_not_called()
            connector.update_state_after_alloc(request, [10, 11, 12], 16)

        send.assert_called_once_with(
            connector.prefix_peer,
            32,
            kind="prefix_tokens",
            transfer_id="tx",
        )

    def test_short_fast_path_skips_transfer_when_full_hit_sends_no_grant(self):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
            cache_config=SimpleNamespace(enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        )
        with patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_TRACE_ENABLED": "0",
                "UNCHAIN_KV_CODEC_MIN_BLOCKS": "64",
                "UNCHAIN_KV_PREFIX_FAST_WAIT_S": "0",
            },
            clear=True,
        ):
            connector = UnchainKVConnector(config, "scheduler")

        self.assertFalse(
            connector._prepare_producer_prefix_session(
                Session("tx", "req", list(range(63)))
            )
        )

    def test_prefix_transfer_suppression_uses_decode_prefix_on_producer(self):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
            cache_config=SimpleNamespace(enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        )
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))
            ]
        )
        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(config, "scheduler", kv_cache_config)
        connector._ensure_prefix_receiver = lambda: None

        request = SimpleNamespace(
            request_id="req",
            prompt_token_ids=list(range(40)),
            kv_transfer_params={"do_remote_decode": True, "transfer_id": "tx"},
        )
        connector.update_state_after_alloc(request, [99], 0)
        connector.grants.add(34, kind="prefix_tokens", transfer_id="tx")
        metadata = connector.build_connector_meta(
            SimpleNamespace(
                scheduled_new_reqs=[
                    SimpleNamespace(
                        req_id="req",
                        prompt_token_ids=list(range(40)),
                        block_ids=([10, 11, 12],),
                        num_computed_tokens=32,
                    )
                ],
                scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
                num_scheduled_tokens={"req": 8},
            )
        )
        self.assertEqual(metadata.requests, [Session("tx", "req", [12])])

        full = SimpleNamespace(
            request_id="full",
            prompt_token_ids=list(range(48)),
            kv_transfer_params={"do_remote_decode": True, "transfer_id": "tx-full"},
        )
        connector.update_state_after_alloc(full, [20, 21, 22], 0)
        connector.grants.add(48, kind="prefix_tokens", transfer_id="tx-full")
        metadata = connector.build_connector_meta(
            SimpleNamespace(
                scheduled_new_reqs=[
                    SimpleNamespace(
                        req_id="full",
                        prompt_token_ids=list(range(48)),
                        block_ids=([20, 21, 22],),
                        num_computed_tokens=32,
                    )
                ],
                scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
                num_scheduled_tokens={"full": 16},
            )
        )
        self.assertEqual(metadata.requests, [])
        self.assertNotIn("full", connector._requests_need_save)

    def test_prefix_receiver_uses_dedicated_control_port(self):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
            cache_config=SimpleNamespace(enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        )
        served = threading.Event()
        receiver = SimpleNamespace(serve=served.set)
        with patch.dict(
            os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True
        ), patch.object(
            vllm_connector, "make_receiver", return_value=receiver
        ) as make_receiver:
            connector = UnchainKVConnector(config, "scheduler")
            connector._ensure_prefix_receiver()
            connector.prefix_receiver_thread.join(1)
            connector._ensure_prefix_receiver()

        self.assertTrue(served.is_set())
        self.assertEqual(make_receiver.call_count, 1)
        make_receiver.assert_called_once_with(
            connector.prefix_bind,
            connector.store,
            trace=connector.trace,
            grants=connector.grants,
        )

    def test_short_tcp_fast_path_starts_prefix_receiver_eagerly(self):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
            cache_config=SimpleNamespace(enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        )
        receiver = SimpleNamespace(serve=lambda: None)
        with patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_TRACE_ENABLED": "0",
                "UNCHAIN_KV_TRANSPORT": "tcp",
                "UNCHAIN_KV_CODEC_MIN_BLOCKS": "64",
            },
            clear=True,
        ), patch.object(
            vllm_connector, "make_receiver", return_value=receiver
        ) as make_receiver:
            connector = UnchainKVConnector(config, "scheduler")
            connector.prefix_receiver_thread.join(1)

        make_receiver.assert_called_once_with(
            connector.prefix_bind,
            connector.store,
            trace=connector.trace,
            grants=connector.grants,
        )

    def test_block_ids_prefers_complete_vllm_block_table(self):
        class FakeBlocks:
            def get_block_ids(self):
                return ([10, 11, 12],)

            def get_unhashed_block_ids(self):
                return [12]

            def get_unhashed_block_ids_all_groups(self):
                return [[12]]

        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")

            self.assertEqual(connector._block_ids(FakeBlocks()), [10, 11, 12])
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_block_ids_can_be_permuted_for_fragmentation_probe(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_permute = os.environ.get("UNCHAIN_KV_PERMUTE_BLOCK_IDS")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_PERMUTE_BLOCK_IDS"] = "odd_even"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")

            self.assertEqual(connector._block_ids([1, 2, 3, 4, 5]), [2, 4, 1, 3, 5])
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_permute is None:
            os.environ.pop("UNCHAIN_KV_PERMUTE_BLOCK_IDS", None)
        else:
            os.environ["UNCHAIN_KV_PERMUTE_BLOCK_IDS"] = old_permute

    def test_consumer_records_decode_request_id_during_match(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            request = SimpleNamespace(
                request_id="decode-req",
                kv_transfer_params={
                    "do_remote_prefill": True,
                    "transfer_id": "tx",
                },
                prompt_token_ids=[1, 2, 3],
            )

            self.assertEqual(connector.get_num_new_matched_tokens(request, 1), (2, True))
            self.assertEqual(connector.sessions["tx"].request_id, "decode-req")
            self.assertEqual(
                connector._requests_need_load["decode-req"].request_id,
                "decode-req",
            )
            match = json.loads(
                (Path(tmp) / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(match["prompt_tokens"], 3)
            self.assertEqual(match["prefix_hit_tokens"], 1)
            self.assertEqual(match["remote_tokens"], 2)
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_builds_worker_metadata_for_waiting_consumer_load(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector._requests_need_load["req"] = Session("tx", "req", [])

            metadata = connector.build_connector_meta(
                SimpleNamespace(scheduled_new_reqs=[])
            )

            self.assertEqual(metadata.requests, [Session("tx", "req", [])])
            self.assertEqual(connector._requests_need_load, {})
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_builds_worker_metadata_for_waiting_producer_save(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector._requests_need_save["req"] = Session("tx", "req", [1])

            metadata = connector.build_connector_meta(
                SimpleNamespace(scheduled_new_reqs=[])
            )

            self.assertEqual(metadata.requests, [Session("tx", "req", [1])])
            self.assertEqual(connector._requests_need_save, {})
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_producer_metadata_does_not_start_grant_receiver(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_window = os.environ.get("UNCHAIN_KV_GRANT_WINDOW")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_GRANT_WINDOW"] = "1"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector._requests_need_save["req"] = Session("tx", "req", [1])

            connector.build_connector_meta(SimpleNamespace(scheduled_new_reqs=[]))

            self.assertIsNone(connector.grant_receiver)
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_window is None:
            os.environ.pop("UNCHAIN_KV_GRANT_WINDOW", None)
        else:
            os.environ["UNCHAIN_KV_GRANT_WINDOW"] = old_window

    def test_get_finished_reports_ready_consumer_request(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_layers = os.environ.get("UNCHAIN_KV_EXPECTED_LAYERS")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = "2"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector._active_transfer_id = "tx"
            connector._active_request_id = "req"
            connector._active_sessions["tx"] = Session("tx", "req", [])
            for layer in range(2):
                payload = b"x"
                connector.store.add(
                    Chunk(
                        ChunkHeader(
                            "tx",
                            "req",
                            layer,
                            0,
                            0,
                            1,
                            0,
                            len(payload),
                            crc32(payload) & 0xFFFFFFFF,
                        ),
                        payload,
                    )
                )

            self.assertEqual(connector.get_finished(), (set(), {"req"}))
            self.assertEqual(connector.get_finished(), (None, None))
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_layers is None:
            os.environ.pop("UNCHAIN_KV_EXPECTED_LAYERS", None)
        else:
            os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = old_layers

    def test_get_finished_reports_after_layer0_ready(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_layers = os.environ.get("UNCHAIN_KV_EXPECTED_LAYERS")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = "2"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector._active_transfer_id = "tx"
            connector._active_request_id = "req"
            connector._active_sessions["tx"] = Session("tx", "req", [])
            payload = b"x"
            connector.store.add(
                Chunk(
                    ChunkHeader(
                        "tx",
                        "req",
                        0,
                        0,
                        0,
                        1,
                        0,
                        len(payload),
                        crc32(payload) & 0xFFFFFFFF,
                    ),
                    payload,
                )
            )

            self.assertEqual(connector.get_finished(), (set(), {"req"}))
            self.assertEqual(connector.get_finished(), (None, None))
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_layers is None:
            os.environ.pop("UNCHAIN_KV_EXPECTED_LAYERS", None)
        else:
            os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = old_layers

    def test_get_finished_uses_recorded_session_without_active_metadata(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_layers = os.environ.get("UNCHAIN_KV_EXPECTED_LAYERS")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = "1"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector.sessions["tx"] = Session("tx", "req", [0])
            payload = b"x"
            connector.store.add(
                Chunk(
                    ChunkHeader(
                        "tx",
                        "req",
                        0,
                        0,
                        0,
                        1,
                        0,
                        len(payload),
                        crc32(payload) & 0xFFFFFFFF,
                    ),
                    payload,
                )
            )

            self.assertEqual(connector.get_finished(), (set(), {"req"}))
            self.assertEqual(connector.get_finished(), (None, None))
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_layers is None:
            os.environ.pop("UNCHAIN_KV_EXPECTED_LAYERS", None)
        else:
            os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = old_layers


    def test_get_finished_reports_multiple_ready_consumer_requests(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_layers = os.environ.get("UNCHAIN_KV_EXPECTED_LAYERS")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = "1"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector._active_sessions = {
                "tx1": Session("tx1", "req1", []),
                "tx2": Session("tx2", "req2", []),
            }
            for transfer_id, request_id in [("tx1", "req1"), ("tx2", "req2")]:
                payload = b"x"
                connector.store.add(
                    Chunk(
                        ChunkHeader(
                            transfer_id,
                            request_id,
                            0,
                            0,
                            0,
                            1,
                            0,
                            len(payload),
                            crc32(payload) & 0xFFFFFFFF,
                        ),
                        payload,
                    )
                )

            self.assertEqual(connector.get_finished(), (set(), {"req1", "req2"}))
            self.assertEqual(connector.get_finished(), (None, None))
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_layers is None:
            os.environ.pop("UNCHAIN_KV_EXPECTED_LAYERS", None)
        else:
            os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = old_layers

    def test_get_finished_admits_ready_sessions_independently(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_layers = os.environ.get("UNCHAIN_KV_EXPECTED_LAYERS")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = "1"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector._active_sessions = {
                "tx1": Session("tx1", "req1", []),
                "tx2": Session("tx2", "req2", []),
            }
            payload = b"x"
            connector.store.add(
                Chunk(
                    ChunkHeader(
                        "tx1",
                        "req1",
                        0,
                        0,
                        0,
                        1,
                        0,
                        len(payload),
                        crc32(payload) & 0xFFFFFFFF,
                    ),
                    payload,
                )
            )

            self.assertEqual(connector.get_finished(), (set(), {"req1"}))
            self.assertEqual(
                connector._load_sessions,
                {"tx1": Session("tx1", "req1", [])},
            )

            connector.store.add(
                Chunk(
                    ChunkHeader(
                        "tx2",
                        "req2",
                        0,
                        0,
                        0,
                        1,
                        0,
                        len(payload),
                        crc32(payload) & 0xFFFFFFFF,
                    ),
                    payload,
                )
            )

            self.assertEqual(connector.get_finished(), (set(), {"req2"}))
            self.assertEqual(connector._load_sessions, connector._active_sessions)
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_layers is None:
            os.environ.pop("UNCHAIN_KV_EXPECTED_LAYERS", None)
        else:
            os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = old_layers

    def test_wait_for_layer_load_restores_multiple_active_sessions(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_layers = os.environ.get("UNCHAIN_KV_EXPECTED_LAYERS")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = "1"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector._active_sessions = {
                "tx1": Session("tx1", "req1", []),
                "tx2": Session("tx2", "req2", []),
            }
            restored = []
            connector._restore_layer = lambda _layer, transfer_id: restored.append(
                transfer_id
            )
            for transfer_id, request_id in [("tx1", "req1"), ("tx2", "req2")]:
                payload = b"x"
                connector.store.add(
                    Chunk(
                        ChunkHeader(
                            transfer_id,
                            request_id,
                            0,
                            0,
                            0,
                            1,
                            0,
                            len(payload),
                            crc32(payload) & 0xFFFFFFFF,
                        ),
                        payload,
                    )
                )

            connector.wait_for_layer_load("model.layers.0.self_attn")

            self.assertEqual(restored, ["tx1", "tx2"])
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_layers is None:
            os.environ.pop("UNCHAIN_KV_EXPECTED_LAYERS", None)
        else:
            os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = old_layers

    def test_wait_for_layer_load_uses_only_admitted_sessions(self):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
        )
        connector = UnchainKVConnector(config, "worker")
        connector._active_sessions = {
            "tx1": Session("tx1", "req1", []),
            "tx2": Session("tx2", "req2", []),
        }
        connector._load_sessions = {
            "tx1": connector._active_sessions["tx1"],
        }
        connector.store.add(
            Chunk(ChunkHeader("tx1", "req1", 0, 0, 0, 1, 0, 1, 0), b"x")
        )
        restored = []
        connector._restore_layer = lambda _layer, transfer: restored.append(transfer)

        connector.wait_for_layer_load("model.layers.0.self_attn")

        self.assertEqual(restored, ["tx1"])

    def test_frontier_grants_cover_active_unadmitted_sessions(self):
        with patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_GRANT_WINDOW": "1",
                "UNCHAIN_KV_TRACE_ENABLED": "0",
            },
        ):
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
        connector._active_sessions = {
            "tx1": Session("tx1", "req1", []),
            "tx2": Session("tx2", "req2", []),
        }
        connector._load_sessions = {
            "tx1": connector._active_sessions["tx1"],
        }
        connector.store.add(
            Chunk(ChunkHeader("tx1", "req1", 0, 0, 0, 1, 0, 1, 0), b"x")
        )
        restored = []
        grants = []
        connector._restore_layer = lambda _layer, transfer: restored.append(transfer)
        connector._send_layer_grant = lambda layer, transfer: grants.append(
            (transfer, layer)
        )

        connector.wait_for_layer_load("model.layers.0.self_attn")

        self.assertEqual(restored, ["tx1"])
        self.assertEqual(grants, [("tx1", 1), ("tx2", 1)])

    def test_wait_for_layer_load_skips_when_no_remote_session(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                trace_path = Path(tmp) / "trace.jsonl"
                os.environ["UNCHAIN_KV_TRACE"] = str(trace_path)
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
                )
                connector = UnchainKVConnector(config, "worker")

                connector.wait_for_layer_load("model.layers.0.self_attn")

                events = trace_path.read_text() if trace_path.exists() else ""
                self.assertNotIn("all_layers_ready", events)
        finally:
            if old_trace is None:
                os.environ.pop("UNCHAIN_KV_TRACE", None)
            else:
                os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_restore_ahead_restores_before_layer_wait(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_layers = os.environ.get("UNCHAIN_KV_EXPECTED_LAYERS")
        old_restore_ahead = os.environ.get("UNCHAIN_KV_RESTORE_AHEAD")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = "1"
            os.environ["UNCHAIN_KV_RESTORE_AHEAD"] = "1"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector.receiver = object()
            connector.metadata = PipelineMetadata([Session("tx", "req", [1])])
            restored = []
            restored_event = threading.Event()

            def restore(layer_name, transfer_id):
                restored.append((layer_name, transfer_id))
                restored_event.set()

            connector._restore_layer = restore
            connector.start_load_kv(None)
            connector.store.add(
                Chunk(
                    ChunkHeader(
                        "tx",
                        "req",
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        crc32(b"x") & 0xFFFFFFFF,
                    ),
                    b"x",
                )
            )

            self.assertTrue(restored_event.wait(1))
            connector.wait_for_layer_load("model.layers.0.self_attn")

            self.assertEqual(len(restored), 1)
            self.assertEqual(restored[0][1], "tx")
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_layers is None:
            os.environ.pop("UNCHAIN_KV_EXPECTED_LAYERS", None)
        else:
            os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = old_layers
        if old_restore_ahead is None:
            os.environ.pop("UNCHAIN_KV_RESTORE_AHEAD", None)
        else:
            os.environ["UNCHAIN_KV_RESTORE_AHEAD"] = old_restore_ahead

    def test_restore_ahead_does_not_restart_after_all_layers_complete(self):
        env = {
            "UNCHAIN_KV_TRACE_ENABLED": "0",
            "UNCHAIN_KV_EXPECTED_LAYERS": "2",
            "UNCHAIN_KV_RESTORE_AHEAD": "2",
        }
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
        )
        with patch.dict(os.environ, env, clear=True):
            connector = UnchainKVConnector(config, "worker")
            connector.receiver = object()
            connector.metadata = PipelineMetadata([Session("tx", "req", [1])])
            restored = []
            worker_starts = 0
            original_restore_ahead = connector._restore_ahead

            def restore_ahead():
                nonlocal worker_starts
                worker_starts += 1
                original_restore_ahead()

            def restore(layer_name, transfer_id):
                restored.append((layer_name, transfer_id))

            connector._restore_ahead = restore_ahead
            connector._restore_layer = restore
            connector.start_load_kv(None)
            for layer in range(2):
                connector.store.add(
                    Chunk(ChunkHeader("tx", "req", layer, 0, 0, 1, 0, 1, 0), b"x")
                )
            first_thread = connector.restore_thread
            self.assertIsNotNone(first_thread)
            first_thread.join(1)
            self.assertFalse(first_thread.is_alive())
            self.assertEqual(worker_starts, 1)
            self.assertEqual(len(restored), 2)

            connector.start_load_kv(None)
            for _ in range(3):
                connector.wait_for_layer_load("model.layers.0.self_attn")
                connector.wait_for_layer_load("model.layers.1.self_attn")

            self.assertEqual(worker_starts, 1)
            self.assertEqual(len(restored), 2)

    def test_restore_ahead_respects_window(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_layers = os.environ.get("UNCHAIN_KV_EXPECTED_LAYERS")
        old_restore_ahead = os.environ.get("UNCHAIN_KV_RESTORE_AHEAD")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
                os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = "4"
                os.environ["UNCHAIN_KV_RESTORE_AHEAD"] = "1"
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
                )
                connector = UnchainKVConnector(config, "worker")
                connector.receiver = object()
                connector.metadata = PipelineMetadata([Session("tx", "req", [1])])
                restored = []
                layer1_restored = threading.Event()
                layer2_restored = threading.Event()

                def restore(layer_name, transfer_id):
                    restored.append((layer_name, transfer_id))
                    if layer_name.startswith("model.layers.1"):
                        layer1_restored.set()
                    if layer_name.startswith("model.layers.2"):
                        layer2_restored.set()

                connector._restore_layer = restore
                connector.start_load_kv(None)
                for layer in range(3):
                    connector.store.add(
                        Chunk(ChunkHeader("tx", "req", layer, 0, 0, 1, 0, 1, 0), b"x")
                    )

                connector.wait_for_layer_load("model.layers.0.self_attn")
                self.assertTrue(layer1_restored.wait(1), restored)
                self.assertEqual(
                    [item[0] for item in restored],
                    ["model.layers.0", "model.layers.1"],
                )
                self.assertFalse(layer2_restored.wait(0.05), restored)

                connector.wait_for_layer_load("model.layers.1.self_attn")
                self.assertTrue(layer2_restored.wait(1), restored)
        finally:
            if old_trace is None:
                os.environ.pop("UNCHAIN_KV_TRACE", None)
            else:
                os.environ["UNCHAIN_KV_TRACE"] = old_trace
            if old_layers is None:
                os.environ.pop("UNCHAIN_KV_EXPECTED_LAYERS", None)
            else:
                os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = old_layers
            if old_restore_ahead is None:
                os.environ.pop("UNCHAIN_KV_RESTORE_AHEAD", None)
            else:
                os.environ["UNCHAIN_KV_RESTORE_AHEAD"] = old_restore_ahead

    def test_restore_ahead_restarts_for_next_request(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_layers = os.environ.get("UNCHAIN_KV_EXPECTED_LAYERS")
        old_restore_ahead = os.environ.get("UNCHAIN_KV_RESTORE_AHEAD")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
                os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = "1"
                os.environ["UNCHAIN_KV_RESTORE_AHEAD"] = "1"
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
                )
                connector = UnchainKVConnector(config, "worker")
                connector.receiver = object()
                restored = []
                tx2_restored = threading.Event()

                def restore(_layer_name, transfer_id):
                    restored.append(transfer_id)
                    if transfer_id == "tx2":
                        tx2_restored.set()

                connector._restore_layer = restore
                connector.metadata = PipelineMetadata([Session("tx1", "req1", [1])])
                connector.start_load_kv(None)
                connector.store.add(
                    Chunk(ChunkHeader("tx1", "req1", 0, 0, 0, 1, 0, 1, 0), b"x")
                )
                first_thread = connector.restore_thread
                self.assertIsNotNone(first_thread)
                first_thread.join(1)

                connector.metadata = PipelineMetadata([Session("tx2", "req2", [2])])
                connector.start_load_kv(None)
                connector.store.add(
                    Chunk(ChunkHeader("tx2", "req2", 0, 0, 0, 1, 0, 1, 0), b"y")
                )

                self.assertTrue(tx2_restored.wait(1), restored)
        finally:
            if old_trace is None:
                os.environ.pop("UNCHAIN_KV_TRACE", None)
            else:
                os.environ["UNCHAIN_KV_TRACE"] = old_trace
            if old_layers is None:
                os.environ.pop("UNCHAIN_KV_EXPECTED_LAYERS", None)
            else:
                os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = old_layers
            if old_restore_ahead is None:
                os.environ.pop("UNCHAIN_KV_RESTORE_AHEAD", None)
            else:
                os.environ["UNCHAIN_KV_RESTORE_AHEAD"] = old_restore_ahead

    def test_restore_ahead_picks_up_session_while_worker_is_running(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_layers = os.environ.get("UNCHAIN_KV_EXPECTED_LAYERS")
        old_restore_ahead = os.environ.get("UNCHAIN_KV_RESTORE_AHEAD")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
                os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = "1"
                os.environ["UNCHAIN_KV_RESTORE_AHEAD"] = "1"
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
                )
                connector = UnchainKVConnector(config, "worker")
                connector.receiver = object()
                restored = []
                tx2_restored = threading.Event()

                def restore(_layer_name, transfer_id):
                    restored.append(transfer_id)
                    if transfer_id == "tx2":
                        tx2_restored.set()

                connector._restore_layer = restore
                connector.metadata = PipelineMetadata([Session("tx1", "req1", [1])])
                connector.start_load_kv(None)
                first_thread = connector.restore_thread
                self.assertIsNotNone(first_thread)

                connector.metadata = PipelineMetadata([Session("tx2", "req2", [2])])
                connector.start_load_kv(None)
                self.assertIs(connector.restore_thread, first_thread)
                connector.store.add(
                    Chunk(ChunkHeader("tx2", "req2", 0, 0, 0, 1, 0, 1, 0), b"y")
                )

                self.assertTrue(tx2_restored.wait(1), restored)
                connector.store.add(
                    Chunk(ChunkHeader("tx1", "req1", 0, 0, 0, 1, 0, 1, 0), b"x")
                )
                first_thread.join(1)
        finally:
            if old_trace is None:
                os.environ.pop("UNCHAIN_KV_TRACE", None)
            else:
                os.environ["UNCHAIN_KV_TRACE"] = old_trace
            if old_layers is None:
                os.environ.pop("UNCHAIN_KV_EXPECTED_LAYERS", None)
            else:
                os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = old_layers
            if old_restore_ahead is None:
                os.environ.pop("UNCHAIN_KV_RESTORE_AHEAD", None)
            else:
                os.environ["UNCHAIN_KV_RESTORE_AHEAD"] = old_restore_ahead

    def test_start_load_adds_sessions_after_receiver_started(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector.receiver = object()
            connector.metadata = PipelineMetadata([Session("tx", "req", [1])])

            connector.start_load_kv(None)

            self.assertEqual(connector._active_sessions["tx"], Session("tx", "req", [1]))
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_start_load_accumulates_concurrent_request_sessions(self):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
        )
        connector = UnchainKVConnector(config, "worker")
        connector.receiver = object()
        connector.metadata = PipelineMetadata([Session("tx1", "req1", [1])])
        connector.start_load_kv(None)

        connector.metadata = PipelineMetadata([Session("tx2", "req2", [2])])
        connector.start_load_kv(None)

        self.assertEqual(
            connector._active_sessions,
            {
                "tx1": Session("tx1", "req1", [1]),
                "tx2": Session("tx2", "req2", [2]),
            },
        )

    def test_get_finished_removes_completed_request_sessions(self):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
        )
        connector = UnchainKVConnector(config, "worker")
        connector.sessions = {
            "tx1": Session("tx1", "req1", [1]),
            "tx2": Session("tx2", "req2", [2]),
        }
        connector._active_sessions = dict(connector.sessions)
        connector._load_sessions = dict(connector.sessions)
        connector._consumer_prefix_tokens = {"req1": 32, "req2": 0}
        connector._consumer_prefix_transfers = {"req1": "tx1", "req2": "tx2"}
        connector._prefix_tokens_sent = {"tx1", "tx2"}

        connector.get_finished({"req1"})

        self.assertEqual(
            connector._active_sessions,
            {"tx2": Session("tx2", "req2", [2])},
        )
        self.assertEqual(connector.sessions, connector._active_sessions)
        self.assertEqual(connector._load_sessions, connector._active_sessions)
        self.assertEqual(connector._consumer_prefix_tokens, {"req2": 0})
        self.assertEqual(connector._consumer_prefix_transfers, {"req2": "tx2"})
        self.assertEqual(connector._prefix_tokens_sent, {"tx2"})

    def test_tcp_get_finished_does_not_revive_payload_request_id(self):
        with patch.dict(os.environ, {"UNCHAIN_KV_TRANSPORT": "tcp"}):
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
        session = Session("tx", "decode-req", [])
        connector.sessions["tx"] = session
        connector._active_sessions["tx"] = session
        connector._load_sessions["tx"] = session
        connector._reported_recving.add("decode-req")
        connector.store.add(
            Chunk(
                ChunkHeader("tx", "producer-req", 0, 0, 0, 1, 0, 1, 0),
                b"x",
            )
        )

        connector.get_finished({"decode-req"})

        self.assertEqual(connector.get_finished(), (None, None))
        self.assertEqual(connector._active_sessions, {})

    def test_start_load_starts_receiver_before_initial_grants(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_window = os.environ.get("UNCHAIN_KV_GRANT_WINDOW")
        old_make_receiver = vllm_connector.make_receiver
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_GRANT_WINDOW"] = "1"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector.metadata = PipelineMetadata([Session("tx", "req", [1])])
            events = []

            class FakeReceiver:
                def serve(self):
                    return None

            def fake_make_receiver(*args, **kwargs):
                events.append("receiver")
                return FakeReceiver()

            vllm_connector.make_receiver = fake_make_receiver
            connector._send_layer_grant = lambda layer, transfer: events.append(
                ("grant", transfer, layer)
            )

            connector.start_load_kv(None)

            self.assertEqual(events[:2], ["receiver", ("grant", "tx", 0)])
        vllm_connector.make_receiver = old_make_receiver
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_window is None:
            os.environ.pop("UNCHAIN_KV_GRANT_WINDOW", None)
        else:
            os.environ["UNCHAIN_KV_GRANT_WINDOW"] = old_window

    def test_tcp_grants_are_sent_once_per_layer(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_transport = os.environ.get("UNCHAIN_KV_TRANSPORT")
        old_send_grant = vllm_connector.send_grant
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_TRANSPORT"] = "tcp"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            calls = []
            vllm_connector.send_grant = (
                lambda _peer, layer, **kwargs: calls.append(
                    (kwargs.get("transfer_id"), layer)
                )
            )

            connector._send_layer_grant(0, "tx-a")
            connector._send_layer_grant(0, "tx-a")
            connector._send_layer_grant(0, "tx-b")

            self.assertEqual(calls, [("tx-a", 0), ("tx-b", 0)])
        vllm_connector.send_grant = old_send_grant
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_transport is None:
            os.environ.pop("UNCHAIN_KV_TRANSPORT", None)
        else:
            os.environ["UNCHAIN_KV_TRANSPORT"] = old_transport

    def test_consumer_sends_frontier_grants(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_window = os.environ.get("UNCHAIN_KV_GRANT_WINDOW")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_GRANT_WINDOW"] = "1"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector.receiver = object()
            connector.metadata = PipelineMetadata([Session("tx", "req", [1])])
            sent = []
            connector._send_layer_grant = lambda layer, transfer: sent.append(
                (transfer, layer)
            )

            connector.start_load_kv(None)

            self.assertEqual(sent, [("tx", 0)])
            connector.store.add(
                Chunk(
                    ChunkHeader(
                        "tx",
                        "req",
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        crc32(b"x") & 0xFFFFFFFF,
                    ),
                    b"x",
                )
            )
            connector._restore_layer = lambda _layer, _transfer: None
            connector.wait_for_layer_load("model.layers.0.self_attn")

            self.assertEqual(sent, [("tx", 0), ("tx", 1)])
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_window is None:
            os.environ.pop("UNCHAIN_KV_GRANT_WINDOW", None)
        else:
            os.environ["UNCHAIN_KV_GRANT_WINDOW"] = old_window



    def test_tcp_codec_raw_passthrough_uses_compressed_frame(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_transport = os.environ.get("UNCHAIN_KV_TRANSPORT")
        old_codec = os.environ.get("UNCHAIN_KV_CODEC")
        old_native = os.environ.get("UNCHAIN_KV_NATIVE_LAYOUT_PAYLOAD")
        old_max_blocks = os.environ.get("UNCHAIN_KV_MAX_BLOCKS")
        import unchain_kv.tcp_data as tcp_data

        old_send = getattr(tcp_data, "send_compressed_native_layer_blocks", None)
        sent = []
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
                os.environ["UNCHAIN_KV_TRANSPORT"] = "tcp"
                os.environ["UNCHAIN_KV_CODEC"] = "raw_passthrough"
                os.environ["UNCHAIN_KV_NATIVE_LAYOUT_PAYLOAD"] = "1"
                os.environ["UNCHAIN_KV_MAX_BLOCKS"] = "0"
                tcp_data.send_compressed_native_layer_blocks = lambda *args, **kwargs: sent.append(
                    (args, kwargs)
                )
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                )
                connector = UnchainKVConnector(config, "worker")
                connector.metadata = PipelineMetadata([Session("tx", "req", [1, 2])])
                connector._native_blocks_bytes = lambda _kv_layer, _block_ids: (
                    b"native",
                    3,
                )

                connector.save_kv_layer("model.layers.2.self_attn", object(), None)
                connector.wait_for_save()

                self.assertEqual(sent[0][0][1:5], ("tx", "req", 2, b"native"))
                self.assertEqual(sent[0][1]["raw_block_size"], 3)
                self.assertEqual(sent[0][1]["block_count"], 2)
                self.assertEqual(sent[0][1]["codec"], "raw_passthrough")
        finally:
            if old_send is None:
                delattr(tcp_data, "send_compressed_native_layer_blocks")
            else:
                tcp_data.send_compressed_native_layer_blocks = old_send
            for name, old in (
                ("UNCHAIN_KV_TRACE", old_trace),
                ("UNCHAIN_KV_TRANSPORT", old_transport),
                ("UNCHAIN_KV_CODEC", old_codec),
                ("UNCHAIN_KV_NATIVE_LAYOUT_PAYLOAD", old_native),
                ("UNCHAIN_KV_MAX_BLOCKS", old_max_blocks),
            ):
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old

    def test_tcp_splitzip_codec_uses_compressed_frame(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_transport = os.environ.get("UNCHAIN_KV_TRANSPORT")
        old_codec = os.environ.get("UNCHAIN_KV_CODEC")
        old_native = os.environ.get("UNCHAIN_KV_NATIVE_LAYOUT_PAYLOAD")
        old_max_blocks = os.environ.get("UNCHAIN_KV_MAX_BLOCKS")
        import unchain_kv.tcp_data as tcp_data

        old_send = tcp_data.send_compressed_native_layer_blocks
        sent = []
        transfers = []
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
                os.environ["UNCHAIN_KV_TRANSPORT"] = "tcp"
                os.environ["UNCHAIN_KV_CODEC"] = "splitzip_bf16"
                os.environ["UNCHAIN_KV_NATIVE_LAYOUT_PAYLOAD"] = "1"
                os.environ["UNCHAIN_KV_MAX_BLOCKS"] = "0"
                tcp_data.send_compressed_native_layer_blocks = lambda *args, **kwargs: sent.append(
                    (args, kwargs)
                )
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                )
                connector = UnchainKVConnector(config, "worker")
                connector.metadata = PipelineMetadata([Session("tx", "req", [1, 2])])

                def fake_splitzip(
                    _kv_layer, _block_ids, _layer, transfer_id=""
                ):
                    transfers.append(transfer_id)
                    return b"zip", 3, 6, 3, 2.0

                connector._splitzip_bf16_native_blocks = fake_splitzip

                connector.save_kv_layer("model.layers.2.self_attn", object(), None)
                connector.wait_for_save()
                rows = [
                    json.loads(line)
                    for line in (Path(tmp) / "trace.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]

                self.assertEqual(sent[0][0][1:5], ("tx", "req", 2, b"zip"))
                self.assertEqual(sent[0][1]["raw_block_size"], 3)
                self.assertEqual(sent[0][1]["raw_bytes"], 6)
                self.assertEqual(sent[0][1]["codec"], "splitzip_bf16")
                self.assertEqual(transfers, ["tx"])
                submitted = next(
                    row for row in rows if row["event"] == "codec_encode_submitted"
                )
                self.assertIn("submit_elapsed_s", submitted)
                self.assertNotIn("codec_elapsed_s", submitted)
                self.assertNotIn("codec_encode_done", [row["event"] for row in rows])
        finally:
            tcp_data.send_compressed_native_layer_blocks = old_send
            for name, old in (
                ("UNCHAIN_KV_TRACE", old_trace),
                ("UNCHAIN_KV_TRANSPORT", old_transport),
                ("UNCHAIN_KV_CODEC", old_codec),
                ("UNCHAIN_KV_NATIVE_LAYOUT_PAYLOAD", old_native),
                ("UNCHAIN_KV_MAX_BLOCKS", old_max_blocks),
            ):
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old

    def test_tcp_splitzip_two_chunk_stages_and_sends_block_halves(self):
        import unchain_kv.tcp_data as tcp_data

        sent = []
        codec_calls = []
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_TRACE": str(Path(tmp) / "trace.jsonl"),
                "UNCHAIN_KV_TRANSPORT": "tcp",
                "UNCHAIN_KV_CODEC": "splitzip_bf16",
                "UNCHAIN_KV_NATIVE_LAYOUT_PAYLOAD": "1",
                "UNCHAIN_KV_MAX_BLOCKS": "0",
                "UNCHAIN_KV_SPLITZIP_CHUNKS": "2",
                "UNCHAIN_KV_SPLITZIP_FIXED5": "0",
                "UNCHAIN_KV_SPLITZIP_TOP16": "0",
            },
        ), patch.object(
            tcp_data,
            "send_compressed_native_layer_blocks",
            lambda *args, **kwargs: sent.append((args, kwargs)),
        ):
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector.metadata = PipelineMetadata(
                [Session("tx", "req", [0, 1, 2, 3])]
            )

            def fake_codec(_kv, block_ids, _layer, _transfer=""):
                codec_calls.append(list(block_ids))
                return (
                    b"a" if block_ids[0] == 0 else b"b",
                    3,
                    3 * len(block_ids),
                    1,
                    "splitzip_bf16",
                    6.0,
                    0.001,
                )

            connector._codec_native_blocks = fake_codec
            connector.save_kv_layer("model.layers.2.self_attn", object(), None)
            connector.wait_for_save()

        self.assertEqual(codec_calls, [[0, 1], [2, 3]])
        self.assertEqual([call[0][4] for call in sent], [b"a", b"b"])
        self.assertEqual([call[1]["chunk_index"] for call in sent], [0, 1])
        self.assertEqual([call[1]["chunks_in_layer"] for call in sent], [2, 2])
        self.assertEqual([call[1]["block_count"] for call in sent], [4, 4])
        self.assertEqual([call[1]["raw_bytes"] for call in sent], [12, 12])

    def test_splitzip_chunk_ring_waits_for_oldest_send_when_slots_are_full(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_TRACE": str(Path(tmp) / "trace.jsonl"),
                "UNCHAIN_KV_CODEC": "splitzip_bf16",
                "UNCHAIN_KV_CODEC_GPU_SLOTS": "1",
            },
        ):
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")
            wait_for_slot = getattr(connector, "_wait_for_codec_slot", None)
            self.assertIsNotNone(wait_for_slot)
            if wait_for_slot is None:
                return
            future = Future()
            connector._codec_pool = {
                (16, "cuda:0"): [SimpleNamespace(reserved=False, future=future)]
            }
            connector.send_futures = [future]
            connector._check_send_futures = lambda done_only: None
            finished = []

            def finish(current, done_only):
                finished.append((current, done_only))
                current.set_result(None)
                return True

            connector._finish_send_future = finish
            wait_for_slot()

            self.assertEqual(finished, [(future, False)])
            self.assertTrue(future.done())

    def test_compressed_native_raw_passthrough_restores_as_native(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_max_blocks = os.environ.get("UNCHAIN_KV_MAX_BLOCKS")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_MAX_BLOCKS"] = "0"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector.sessions["tx"] = Session("tx", "req", [1, 2])
            connector.kv_caches = {"model.layers.2.self_attn": object()}
            connector.store.add(
                Chunk(
                    ChunkHeader("tx", "req", 2, 0, 0, 1, 0, 6, 0),
                    b"native",
                ),
                layout="compressed_native",
                metadata={
                    "block_count": 2,
                    "raw_block_size": 3,
                    "raw_bytes": 6,
                    "encoded_bytes": 6,
                    "codec": "raw_passthrough",
                },
            )
            calls = []
            connector._copy_native_blocks = (
                lambda kv_layer, block_ids, payloads: calls.append(
                    (kv_layer, block_ids, payloads)
                )
                or (2, 6)
            )

            connector._restore_layer("model.layers.2.self_attn", "tx")

            self.assertEqual(calls[0][1], [1, 2])
            self.assertEqual([bytes(payload) for payload in calls[0][2]], [b"native"])
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_max_blocks is None:
            os.environ.pop("UNCHAIN_KV_MAX_BLOCKS", None)
        else:
            os.environ["UNCHAIN_KV_MAX_BLOCKS"] = old_max_blocks

    def test_compressed_native_splitzip_restores_with_decoder(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_max_blocks = os.environ.get("UNCHAIN_KV_MAX_BLOCKS")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_MAX_BLOCKS"] = "0"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector.sessions["tx"] = Session("tx", "req", [1, 2])
            connector.kv_caches = {"model.layers.2.self_attn": object()}
            connector.store.add(
                Chunk(
                    ChunkHeader("tx", "req", 2, 0, 0, 1, 0, 3, 0),
                    b"zip",
                ),
                layout="compressed_native",
                metadata={
                    "block_count": 2,
                    "raw_block_size": 3,
                    "raw_bytes": 6,
                    "encoded_bytes": 3,
                    "codec": "splitzip_bf16",
                },
            )
            calls = []
            connector._copy_splitzip_bf16_native_blocks = (
                lambda kv_layer, block_ids, payloads, metadata: calls.append(
                    (kv_layer, block_ids, payloads, metadata)
                )
                or (2, 6)
            )

            connector._restore_layer("model.layers.2.self_attn", "tx")

            self.assertEqual(calls[0][1], [1, 2])
            self.assertEqual([bytes(payload) for payload in calls[0][2]], [b"zip"])
            self.assertEqual(calls[0][3]["codec"], "splitzip_bf16")
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_max_blocks is None:
            os.environ.pop("UNCHAIN_KV_MAX_BLOCKS", None)
        else:
            os.environ["UNCHAIN_KV_MAX_BLOCKS"] = old_max_blocks

    def test_compressed_native_splitzip_restores_two_chunks_to_block_halves(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "UNCHAIN_KV_TRACE": str(Path(tmp) / "trace.jsonl"),
                    "UNCHAIN_KV_MAX_BLOCKS": "0",
                },
            ):
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
                )
                connector = UnchainKVConnector(config, "worker")
                connector.sessions["tx"] = Session("tx", "req", [10, 11, 12, 13])
                connector.kv_caches = {"model.layers.2.self_attn": object()}
                metadata = {
                    "block_count": 4,
                    "raw_block_size": 3,
                    "raw_bytes": 12,
                    "codec": "splitzip_bf16",
                    "chunks_in_layer": 2,
                }
                for chunk_index, payload in enumerate((b"zip", b"zap")):
                    connector.store.add(
                        Chunk(
                            ChunkHeader(
                                "tx", "req", 2, 0, chunk_index, 2, 0, 3, 0
                            ),
                            payload,
                        ),
                        layout="compressed_native",
                        metadata=metadata,
                    )
                calls = []
                connector._copy_splitzip_bf16_native_blocks = (
                    lambda _kv, ids, payloads, meta: calls.append(
                        (ids, [bytes(payload) for payload in payloads], dict(meta))
                    )
                    or (len(ids), int(meta["raw_bytes"]))
                )

                connector._restore_layer("model.layers.2.self_attn", "tx")

                self.assertEqual([call[0] for call in calls], [[10, 11], [12, 13]])
                self.assertEqual([call[1] for call in calls], [[b"zip"], [b"zap"]])
                self.assertEqual(
                    [(call[2]["raw_bytes"], call[2]["encoded_bytes"]) for call in calls],
                    [(6, 3), (6, 3)],
                )

    def test_splitzip_bf16_codec_round_trips_when_torch_available(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not installed")
        old_values = {
            name: os.environ.get(name)
            for name in (
                "UNCHAIN_KV_TRACE",
                "UNCHAIN_KV_CODEC",
                "UNCHAIN_KV_CODEC_GPU_SLOTS",
            )
        }
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_CODEC"] = "splitzip_bf16"
            os.environ["UNCHAIN_KV_CODEC_GPU_SLOTS"] = "1"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            index = torch.arange(256, dtype=torch.int32, device=device)
            high = torch.tensor(
                [0x3F, 0xBF, 0x40, 0xC0], dtype=torch.int32, device=device
            )
            words = torch.bitwise_or(
                torch.bitwise_left_shift(high[torch.remainder(index, 4)], 8),
                torch.remainder(index, 251),
            ).to(torch.uint16)
            kv_layer = words.view(torch.bfloat16).reshape(2, 16, 8)

            encoded = connector._splitzip_bf16_native_blocks(
                kv_layer, list(range(16))
            )

            self.assertIsNotNone(encoded)
            data, _, raw_bytes, encoded_bytes, _ = encoded
            self.assertLess(encoded_bytes, raw_bytes)
            target = torch.empty_like(kv_layer)
            restored, copied = connector._copy_splitzip_bf16_native_blocks(
                target,
                list(range(16)),
                [bytes(data)],
                {
                    "raw_bytes": raw_bytes,
                    "encoded_bytes": encoded_bytes,
                    "codec": "splitzip_bf16",
                },
            )
            self.assertEqual(restored, 16)
            self.assertEqual(copied, raw_bytes)
            self.assertTrue(
                torch.equal(
                    target.cpu().view(torch.uint16),
                    words.cpu().reshape(2, 16, 8),
                )
            )
        for name, old in old_values.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old

    def test_splitzip_bf16_codec_compresses_nibble_palette(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not installed")
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_codec = os.environ.get("UNCHAIN_KV_CODEC")
        old_slots = os.environ.get("UNCHAIN_KV_CODEC_GPU_SLOTS")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_CODEC"] = "splitzip_bf16"
            os.environ["UNCHAIN_KV_CODEC_GPU_SLOTS"] = "1"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            index = torch.arange(256, dtype=torch.int32, device=device)
            high = 0x30 + torch.remainder(index, 32)
            words = torch.bitwise_or(
                torch.bitwise_left_shift(high, 8),
                torch.remainder(index, 256),
            ).to(torch.uint16)
            self.assertGreater(torch.unique(high.to(torch.uint8)).numel(), 16)
            kv_layer = words.view(torch.bfloat16).reshape(2, 16, 8)

            encoded = connector._splitzip_bf16_native_blocks(
                kv_layer, list(range(16))
            )

            self.assertIsNotNone(encoded)
            data, _, raw_bytes, encoded_bytes, _ = encoded
            self.assertLess(encoded_bytes, raw_bytes)
            target = torch.empty_like(kv_layer)
            restored, copied = connector._copy_splitzip_bf16_native_blocks(
                target,
                list(range(16)),
                [bytes(data)],
                {"raw_bytes": raw_bytes, "encoded_bytes": encoded_bytes},
            )
            self.assertEqual(restored, 16)
            self.assertEqual(copied, raw_bytes)
            self.assertTrue(
                torch.equal(
                    target.cpu().view(torch.uint16),
                    words.cpu().reshape(2, 16, 8),
                )
            )
        for name, old in (
            ("UNCHAIN_KV_TRACE", old_trace),
            ("UNCHAIN_KV_CODEC", old_codec),
            ("UNCHAIN_KV_CODEC_GPU_SLOTS", old_slots),
        ):
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old

    def test_splitzip_bf16_restores_fixed5_palette(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not installed")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")
            index = torch.arange(256, dtype=torch.int32)
            palette = torch.arange(0x40, 0x60, dtype=torch.uint8)
            codes = torch.remainder(index, 32).to(torch.uint8)
            low = torch.remainder(index, 251).to(torch.uint8)
            high = palette[codes.long()].to(torch.int32)
            words = torch.bitwise_or(torch.bitwise_left_shift(high, 8), low.to(torch.int32))
            words = words.to(torch.uint16)
            code_bytes = (256 * 5 + 7) // 8
            comp = torch.empty((33 + 256 + code_bytes,), dtype=torch.uint8)
            comp[0] = 4
            comp[1:33].copy_(palette)
            comp[33 : 33 + 256].copy_(low)
            connector._pack_splitzip_bits(codes, 5, comp[33 + 256 :], torch)
            target = torch.empty((2, 16, 8), dtype=torch.bfloat16)

            restored, copied = connector._copy_splitzip_bf16_native_blocks(
                target,
                list(range(16)),
                [bytes(comp.tolist())],
                {"raw_bytes": 512, "encoded_bytes": comp.numel()},
            )

            self.assertEqual(restored, 16)
            self.assertEqual(copied, 512)
            self.assertTrue(torch.equal(target.view(torch.uint16), words.reshape(2, 16, 8)))

    def test_splitzip_fixed6_prefers_native_decoder(self):
        fake_bfloat16 = object()

        class FakeBlock:
            dtype = fake_bfloat16

            def numel(self):
                return 4

            def element_size(self):
                return 2

        class FakeTarget:
            shape = (2, 2, 2)
            device = SimpleNamespace(type="cuda")

            def dim(self):
                return 3

            def __getitem__(self, _key):
                return FakeBlock()

        class FakeScalar:
            def cpu(self):
                return self

            def item(self):
                return 255

        class FakeComp:
            def __init__(self, size):
                self.size = size

            def to(self, **_kwargs):
                return self

            def numel(self):
                return self.size

            def __getitem__(self, key):
                return FakeScalar() if key == 0 else self

        fake_torch = SimpleNamespace(bfloat16=fake_bfloat16, uint8=object())
        target = FakeTarget()
        raw_bytes = 16
        encoded_bytes = 65 + 8 + 6
        payload = bytearray(encoded_bytes)
        payload[0] = 3
        calls = []

        import unchain_kv.splitzip_cuda as splitzip_cuda

        with patch.dict(os.environ, {"UNCHAIN_KV_SPLITZIP_NATIVE_DECODE": "1"}), patch.dict(
            sys.modules, {"torch": fake_torch}
        ), patch.object(
            splitzip_cuda,
            "decode_fixed6",
            lambda source, out, block_ids, size: calls.append(
                (source, out, block_ids, size)
            )
            or size,
            create=True,
        ):
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector._source_tensor_from_payload = (
                lambda _torch, _payload, _dtype: FakeComp(encoded_bytes)
            )
            restored, copied = connector._copy_splitzip_bf16_native_blocks(
                target,
                [0, 1],
                [payload],
                {"raw_bytes": raw_bytes, "encoded_bytes": encoded_bytes},
            )

        self.assertEqual((restored, copied), (2, raw_bytes))
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][1], target)
        self.assertEqual(calls[0][2:], ([0, 1], raw_bytes))

    def test_splitzip_decode_rejects_bad_mode_and_length_before_cuda(self):
        fake_bfloat16 = object()

        class FakeBlock:
            dtype = fake_bfloat16

            def numel(self):
                return 4

            def element_size(self):
                return 2

        class FakeTarget:
            shape = (2, 2, 2)
            device = SimpleNamespace(type="cuda")

            def dim(self):
                return 3

            def __getitem__(self, _key):
                return FakeBlock()

        fake_torch = SimpleNamespace(bfloat16=fake_bfloat16, uint8=object())
        with patch.dict(sys.modules, {"torch": fake_torch}):
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector._source_tensor_from_payload = Mock(
                side_effect=AssertionError("invalid payload reached CUDA")
            )
            for payload, encoded_bytes in ((b"\xff", 1), (b"\x03", 79)):
                with self.subTest(payload=payload, encoded_bytes=encoded_bytes):
                    self.assertEqual(
                        connector._copy_splitzip_bf16_native_blocks(
                            FakeTarget(),
                            [0, 1],
                            [payload],
                            {"raw_bytes": 16, "encoded_bytes": encoded_bytes},
                        ),
                        (0, 0),
                    )
            connector._source_tensor_from_payload.assert_not_called()

    def test_splitzip_top16_decodes_without_native_decoder(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not installed")

        import unchain_kv.splitzip_cuda as splitzip_cuda

        count = 256
        codebooks = splitzip_cuda._TOP16_CODEBOOKS[:32]
        words = []
        body = bytearray(count)
        codes = bytearray((count + 1) // 2)
        escape_position = 3
        escape_exponent = 100
        for index in range(count):
            plane = index >= count // 2
            code = index % 16
            exponent = codebooks[plane * 16 + code]
            if index == escape_position:
                exponent = escape_exponent
                code = 0
            word = ((index % 2) << 15) | (exponent << 7) | (index % 128)
            words.append(word)
            body[index] = (word & 0x7F) | ((word >> 8) & 0x80)
            codes[index // 2] |= code << ((index % 2) * 4)
        escape_capacity = (count + 199) // 200
        payload = bytearray(5 + count + len(codes) + escape_capacity * 5)
        payload[0] = 5
        payload[1:5] = (1).to_bytes(4, "little")
        payload[5 : 5 + count] = body
        payload[5 + count : 5 + count + len(codes)] = codes
        entry = 5 + count + len(codes)
        payload[entry : entry + 4] = escape_position.to_bytes(4, "little")
        payload[entry + 4] = escape_exponent

        with patch.dict(
            os.environ, {"UNCHAIN_KV_SPLITZIP_NATIVE_DECODE": "0"}
        ):
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
        target = torch.zeros((2, 16, 8), dtype=torch.bfloat16)
        restored, copied = connector._copy_splitzip_bf16_native_blocks(
            target,
            list(range(16)),
            [payload],
            {
                "raw_bytes": count * 2,
                "encoded_bytes": len(payload),
                "_layer_index": 0,
            },
        )

        expected = torch.tensor(words, dtype=torch.int32).to(torch.uint16)
        self.assertEqual((restored, copied), (16, count * 2))
        self.assertTrue(torch.equal(target.view(torch.uint16).reshape(-1), expected))

    def test_splitzip_raw_mode_restores_bitwise(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not installed")

        words = torch.arange(256, dtype=torch.int32).to(torch.uint16)
        raw = bytes(words.view(torch.uint8).tolist())
        payload = bytes([6]) + raw
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
        )
        connector = UnchainKVConnector(config, "worker")
        target = torch.zeros((2, 16, 8), dtype=torch.bfloat16)

        restored, copied = connector._copy_splitzip_bf16_native_blocks(
            target,
            list(range(16)),
            [payload],
            {"raw_bytes": len(raw), "encoded_bytes": len(payload)},
        )

        self.assertEqual((restored, copied), (16, len(raw)))
        self.assertTrue(torch.equal(target.view(torch.uint16).reshape(-1), words))

    def test_splitzip_bf16_prefers_native_encoder(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_codec = os.environ.get("UNCHAIN_KV_CODEC")
        old_slots = os.environ.get("UNCHAIN_KV_CODEC_GPU_SLOTS")
        old_fixed5 = os.environ.get("UNCHAIN_KV_SPLITZIP_FIXED5")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_CODEC"] = "splitzip_bf16"
            os.environ["UNCHAIN_KV_CODEC_GPU_SLOTS"] = "1"
            os.environ.pop("UNCHAIN_KV_SPLITZIP_FIXED5", None)
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")

            class FakeDType:
                def __str__(self):
                    return "torch.bfloat16"

            class FakeTensor:
                dtype = FakeDType()
                device = "cuda:0"

                def __init__(self, size=32):
                    self.size = size

                def numel(self):
                    return self.size // 2

                def element_size(self):
                    return 2

                def __getitem__(self, _key):
                    return self

                def detach(self):
                    return self

                def numpy(self):
                    return bytearray(b"zip!")

            fake_torch = SimpleNamespace()
            fake_source = FakeTensor(1 << 20)
            fake_out = FakeTensor()
            calls = []

            connector._native_tensor_for_codec = (
                lambda _kv_layer, _block_ids, _torch: fake_source
            )
            connector._stage_cpu = lambda tensor: tensor

            import unchain_kv.splitzip_cuda as splitzip_cuda

            old_encode = splitzip_cuda.encode_bf16
            splitzip_cuda.encode_bf16 = lambda source, out: calls.append(
                (source, out)
            ) or 4
            connector._acquire_codec_buffer = lambda *_args: SimpleNamespace(
                tensor=fake_out
            )
            try:
                with patch.dict(sys.modules, {"torch": fake_torch}):
                    encoded = connector._splitzip_bf16_native_blocks(
                        object(), [0, 1]
                    )
            finally:
                splitzip_cuda.encode_bf16 = old_encode

            self.assertEqual(calls, [(fake_source, fake_out)])
            self.assertEqual(connector._ready_payload(encoded[0]).tobytes(), b"zip!")
            self.assertEqual(encoded[1:], (524288, 1048576, 4, 262144.0))
        for name, old in (
            ("UNCHAIN_KV_TRACE", old_trace),
            ("UNCHAIN_KV_CODEC", old_codec),
            ("UNCHAIN_KV_CODEC_GPU_SLOTS", old_slots),
            ("UNCHAIN_KV_SPLITZIP_FIXED5", old_fixed5),
        ):
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old

    def test_splitzip_bf16_uses_top16_large_and_fixed6_small(self):
        class FakeDType:
            def __str__(self):
                return "torch.bfloat16"

        class FakeTensor:
            dtype = FakeDType()
            device = "cuda:0"

            def __init__(self, size=32):
                self.size = size

            def numel(self):
                return self.size // 2

            def element_size(self):
                return 2

            def __getitem__(self, _key):
                return self

            def detach(self):
                return self

            def numpy(self):
                return bytearray(b"top!")

        fake_torch = SimpleNamespace()
        fake_source = FakeTensor(1 << 20)
        fake_out = FakeTensor()
        top16_calls = []
        fixed6_calls = []

        import unchain_kv.splitzip_cuda as splitzip_cuda

        with patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_CODEC": "splitzip_bf16",
                "UNCHAIN_KV_CODEC_GPU_SLOTS": "1",
                "UNCHAIN_KV_SPLITZIP_TOP16": "1",
                "UNCHAIN_KV_SPLITZIP_NATIVE_DECODE": "1",
            },
        ), patch.dict(sys.modules, {"torch": fake_torch}), patch.object(
            splitzip_cuda,
            "encode_top16",
            lambda source, out, layer: top16_calls.append((source, out, layer)) or 4,
            create=True,
        ), patch.object(
            splitzip_cuda,
            "encode_bf16",
            lambda source, out: fixed6_calls.append((source, out)) or 4,
        ):
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector._native_tensor_for_codec = (
                lambda _kv_layer, _block_ids, _torch: fake_source
            )
            connector._stage_cpu = lambda tensor: tensor
            connector._acquire_codec_buffer = lambda *_args: SimpleNamespace(
                tensor=fake_out
            )
            encoded = connector._splitzip_bf16_native_blocks(
                object(), [0, 1], layer_index=7
            )
            fake_source.size = 32768
            small = connector._splitzip_bf16_native_blocks(
                object(), [0], layer_index=7
            )

        self.assertEqual(top16_calls, [(fake_source, fake_out, 7)])
        self.assertEqual(fixed6_calls, [(fake_source, fake_out)])
        self.assertEqual(connector._ready_payload(encoded[0]).tobytes(), b"top!")
        self.assertEqual(connector._ready_payload(small[0]).tobytes(), b"top!")

    def test_splitzip_top16_backoff_single_flight_and_recovery(self):
        connector = object.__new__(UnchainKVConnector)
        connector.splitzip_top16 = True
        connector.splitzip_top16_max_cooldown_payloads = 32
        connector._splitzip_top16_unavailable_layers = set()
        connector._splitzip_top16_cooldowns = {}
        connector._splitzip_top16_windows = {}
        connector._splitzip_top16_probe_layers = set()
        connector._splitzip_top16_epochs = {}
        connector._splitzip_top16_lock = threading.Lock()
        rows = []
        connector.trace = SimpleNamespace(
            event=lambda event, **fields: rows.append((event, fields))
        )

        attempt = connector._should_try_splitzip_top16(7, 1 << 20)
        self.assertEqual(attempt, (True, 0, False))
        connector._finish_splitzip_top16_attempt(7, *attempt[1:], True)
        self.assertEqual(connector._splitzip_top16_windows, {7: 1})

        for expected_window in (2, 4, 8, 16, 32, 32):
            while connector._splitzip_top16_cooldowns[7] > 0:
                self.assertFalse(
                    connector._should_try_splitzip_top16(7, 1 << 20)[0]
                )
            with ThreadPoolExecutor(max_workers=8) as pool:
                attempts = list(
                    pool.map(
                        lambda _: connector._should_try_splitzip_top16(
                            7, 1 << 20
                        ),
                        range(16),
                    )
                )
            probes = [item for item in attempts if item[0]]
            self.assertEqual(len(probes), 1)
            connector._finish_splitzip_top16_attempt(
                7, probes[0][1], probes[0][2], True
            )
            self.assertEqual(connector._splitzip_top16_windows[7], expected_window)

        while connector._splitzip_top16_cooldowns[7] > 0:
            connector._should_try_splitzip_top16(7, 1 << 20)
        probe = connector._should_try_splitzip_top16(7, 1 << 20)
        connector._finish_splitzip_top16_attempt(7, *probe[1:], False)
        self.assertNotIn(7, connector._splitzip_top16_cooldowns)
        self.assertNotIn(7, connector._splitzip_top16_windows)
        self.assertTrue(
            any(event == "splitzip_top16_probe_success" for event, _ in rows)
        )

        before = dict(connector._splitzip_top16_cooldowns)
        self.assertFalse(
            connector._should_try_splitzip_top16(7, (1 << 20) - 1)[0]
        )
        self.assertEqual(connector._splitzip_top16_cooldowns, before)

    def test_splitzip_top16_stale_ready_result_cannot_reset_backoff(self):
        connector = object.__new__(UnchainKVConnector)
        connector.splitzip_top16 = True
        connector.splitzip_top16_max_cooldown_payloads = 32
        connector._splitzip_top16_unavailable_layers = set()
        connector._splitzip_top16_cooldowns = {}
        connector._splitzip_top16_windows = {}
        connector._splitzip_top16_probe_layers = set()
        connector._splitzip_top16_epochs = {}
        connector._splitzip_top16_lock = threading.Lock()
        rows = []
        connector.trace = SimpleNamespace(
            event=lambda event, **fields: rows.append((event, fields))
        )

        first = connector._should_try_splitzip_top16(7, 1 << 20)
        late = connector._should_try_splitzip_top16(7, 1 << 20)
        connector._finish_splitzip_top16_attempt(7, *first[1:], True)
        connector._finish_splitzip_top16_attempt(7, *late[1:], False)

        self.assertEqual(connector._splitzip_top16_cooldowns, {7: 1})
        self.assertEqual(connector._splitzip_top16_windows, {7: 1})
        self.assertEqual(rows[-1][0], "splitzip_top16_stale_result")

    def test_splitzip_top16_unavailable_sticks_to_fixed6(self):
        class FakeDType:
            def __str__(self):
                return "torch.bfloat16"

        class FakeTensor:
            dtype = FakeDType()
            device = "cuda:0"

            def numel(self):
                return 1 << 19

            def element_size(self):
                return 2

            def __getitem__(self, _key):
                return self

            def detach(self):
                return self

            def numpy(self):
                return bytearray(b"f6ok")

        import unchain_kv.splitzip_cuda as splitzip_cuda

        top16_calls = []
        fixed6_calls = []
        with patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_CODEC": "splitzip_bf16",
                "UNCHAIN_KV_CODEC_GPU_SLOTS": "1",
                "UNCHAIN_KV_SPLITZIP_TOP16": "1",
                "UNCHAIN_KV_SPLITZIP_TOP16_MAX_COOLDOWN": "8",
            },
        ), patch.dict(sys.modules, {"torch": SimpleNamespace()}), patch.object(
            splitzip_cuda,
            "encode_top16",
            side_effect=lambda *_args: top16_calls.append(1) or None,
        ), patch.object(
            splitzip_cuda,
            "encode_bf16",
            side_effect=lambda *_args, **_kwargs: fixed6_calls.append(1) or 4,
        ):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
            tensor = FakeTensor()
            connector._native_tensor_for_codec = lambda *_args: tensor
            connector._stage_cpu = lambda value: value
            connector._acquire_codec_buffer = lambda *_args: SimpleNamespace(
                tensor=tensor
            )

            first = connector._splitzip_bf16_native_blocks(object(), [0], 7)
            second = connector._splitzip_bf16_native_blocks(object(), [0], 7)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(len(top16_calls), 1)
        self.assertEqual(len(fixed6_calls), 2)
        self.assertEqual(connector.splitzip_top16_max_cooldown_payloads, 8)
        self.assertIn(7, connector._splitzip_top16_unavailable_layers)

    def test_splitzip_top16_uses_writeback_payload_when_enabled(self):
        class FakeDType:
            def __str__(self):
                return "torch.bfloat16"

        class FakeTensor:
            dtype = FakeDType()
            device = "cuda:0"

            def __init__(self, size=1 << 20):
                self.size = size

            def numel(self):
                return self.size // 2

            def element_size(self):
                return 2

            def __getitem__(self, _key):
                return self

        fake_source = FakeTensor()
        fake_out = FakeTensor()
        writebacks = []
        stages = []

        import unchain_kv.splitzip_cuda as splitzip_cuda

        with patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_TRANSPORT": "tcp",
                "UNCHAIN_KV_CODEC": "splitzip_bf16",
                "UNCHAIN_KV_CODEC_GPU_SLOTS": "1",
                "UNCHAIN_KV_SPLITZIP_TOP16": "1",
                "UNCHAIN_KV_SPLITZIP_NATIVE_DECODE": "1",
                "UNCHAIN_KV_CODEC_WRITEBACK": "1",
                "UNCHAIN_KV_HOST_MIRROR_LAYERS": "1",
                "UNCHAIN_KV_HOST_MIRROR_BYTES": "1048576",
                "UNCHAIN_KV_SEND_INFLIGHT": "1",
                "UNCHAIN_KV_TRACE_BF16_EXPONENTS": "0",
            },
        ), patch.dict(sys.modules, {"torch": SimpleNamespace()}), patch.object(
            splitzip_cuda,
            "encode_top16",
            lambda _source, _out, _layer: 4,
            create=True,
        ):
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
                cache_config=SimpleNamespace(enable_prefix_caching=False),
                scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
            )
            connector = UnchainKVConnector(config, "worker")
            connector._native_tensor_for_codec = (
                lambda _kv_layer, _block_ids, _torch: fake_source
            )
            connector._acquire_codec_buffer = lambda *_args: SimpleNamespace(
                tensor=fake_out
            )
            connector._try_splitzip_writeback = lambda *args: (
                writebacks.append(args) or memoryview(b"back")
            )
            connector._stage_cpu = lambda tensor: stages.append(tensor)
            encoded = connector._splitzip_bf16_native_blocks(
                object(), [0, 1], layer_index=7
            )

        self.assertEqual(len(writebacks), 1)
        self.assertEqual(stages, [])
        self.assertEqual(connector._ready_payload(encoded[0]).tobytes(), b"back")

    def test_splitzip_writeback_preserves_kv_for_prefix_and_chunked_prefill(self):
        class FakeDType:
            def __str__(self):
                return "torch.bfloat16"

        class FakeTensor:
            dtype = FakeDType()
            device = "cuda:0"

            def __init__(self, size=1 << 20, data=b"top!"):
                self.size = size
                self.data = bytearray(data)

            def numel(self):
                return self.size // 2

            def element_size(self):
                return 2

            def __getitem__(self, _key):
                return self

            def detach(self):
                return self

            def numpy(self):
                return self.data

        import unchain_kv.splitzip_cuda as splitzip_cuda

        env = {
            "UNCHAIN_KV_TRACE_ENABLED": "0",
            "UNCHAIN_KV_TRANSPORT": "tcp",
            "UNCHAIN_KV_CODEC": "splitzip_bf16",
            "UNCHAIN_KV_CODEC_GPU_SLOTS": "1",
            "UNCHAIN_KV_SPLITZIP_TOP16": "1",
            "UNCHAIN_KV_SPLITZIP_NATIVE_DECODE": "1",
            "UNCHAIN_KV_CODEC_WRITEBACK": "1",
            "UNCHAIN_KV_CODEC_WRITEBACK_STRICT": "1",
            "UNCHAIN_KV_HOST_MIRROR_LAYERS": "1",
            "UNCHAIN_KV_HOST_MIRROR_BYTES": "1048576",
            "UNCHAIN_KV_SEND_INFLIGHT": "1",
        }
        for prefix_cache, chunked_prefill in ((True, False), (False, True)):
            with self.subTest(
                prefix_cache=prefix_cache,
                chunked_prefill=chunked_prefill,
            ), patch.dict(os.environ, env, clear=True), patch.dict(
                sys.modules, {"torch": SimpleNamespace()}
            ), patch.object(
                splitzip_cuda, "encode_top16", return_value=4, create=True
            ):
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
                    cache_config=SimpleNamespace(
                        enable_prefix_caching=prefix_cache
                    ),
                    scheduler_config=SimpleNamespace(
                        enable_chunked_prefill=chunked_prefill
                    ),
                )
                connector = UnchainKVConnector(config, "worker")
                source = FakeTensor()
                codec = vllm_connector._CodecBuffer(FakeTensor(), reserved=True)
                connector._native_tensor_for_codec = lambda *_args: source
                connector._wait_for_codec_buffer = lambda *_args: codec
                connector._try_splitzip_writeback = lambda *_args: self.fail(
                    "source-preserving mode must not overwrite KV"
                )
                connector._stage_cpu = lambda tensor: tensor

                encoded = connector._splitzip_bf16_native_blocks(
                    object(), [0, 1], layer_index=7
                )

                self.assertTrue(connector.codec_writeback)
                self.assertTrue(connector.codec_writeback_preserve_source)
                self.assertFalse(connector.codec_writeback_overwrite)
                self.assertEqual(
                    connector._ready_payload(encoded[0]).tobytes(), b"top!"
                )
                connector.send_executor.shutdown(wait=True)

    def test_bf16_exponent_trace_emits_separate_summaries_once(self):
        class FakeTensor:
            def __init__(self, values):
                self.values = values

            def detach(self):
                return self

            def view(self, _dtype):
                return self

            def reshape(self, _size):
                return self

            def to(self, _dtype):
                return self

        class FakeSource:
            def __init__(self, key_words, value_words):
                self.planes = (FakeTensor(key_words), FakeTensor(value_words))

            def __getitem__(self, index):
                return self.planes[index]

        class FakeStack:
            def __init__(self, rows, fake_torch):
                self.rows = rows
                self.fake_torch = fake_torch

            def cpu(self):
                self.fake_torch.cpu_calls += 1
                return self

            def tolist(self):
                return self.rows

        class FakeTorch:
            uint16 = object()
            int32 = object()

            def __init__(self):
                self.bincount_calls = 0
                self.cpu_calls = 0

            def bitwise_right_shift(self, tensor, bits):
                return FakeTensor([value >> bits for value in tensor.values])

            def bitwise_and(self, tensor, mask):
                return FakeTensor([value & mask for value in tensor.values])

            def bincount(self, tensor, minlength):
                self.bincount_calls += 1
                counts = [0] * minlength
                for value in tensor.values:
                    counts[value] += 1
                return FakeTensor(counts)

            def stack(self, tensors):
                return FakeStack([tensor.values for tensor in tensors], self)

        source = FakeSource(
            [0x8000 | (1 << 7), (1 << 7) | 1, 2 << 7, (255 << 7) | 0x7F],
            [3 << 7, 0x8000 | (4 << 7), (4 << 7) | 1, (4 << 7) | 2],
        )
        fake_torch = FakeTorch()
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            with patch.dict(
                os.environ,
                {
                    "UNCHAIN_KV_TRACE": str(trace_path),
                    "UNCHAIN_KV_TRACE_BF16_EXPONENTS": "1",
                },
                clear=True,
            ):
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                )
                connector = UnchainKVConnector(config, "worker")

                connector._trace_bf16_exponents(source, "tx", 7, fake_torch)
                connector._trace_bf16_exponents(source, "tx", 7, fake_torch)

            rows = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event"], "bf16_exponent_stats")
        self.assertEqual(rows[0]["transfer"], "tx")
        self.assertEqual(rows[0]["layer"], 7)
        self.assertEqual(rows[0]["k"]["words"], 4)
        self.assertEqual(rows[0]["k"]["top8_exponents"], [1, 2, 255])
        self.assertEqual(rows[0]["v"]["words"], 4)
        self.assertEqual(rows[0]["v"]["top8_exponents"], [4, 3])
        self.assertEqual(fake_torch.bincount_calls, 2)
        self.assertEqual(fake_torch.cpu_calls, 1)

    def test_bf16_exponent_trace_retries_after_transient_failure(self):
        class FakePlane:
            def detach(self):
                return self

            def view(self, _dtype):
                return self

            def reshape(self, _size):
                return self

            def to(self, _dtype):
                return self

        class FakeCounts:
            def __init__(self, values):
                self.values = values

        class FakeStack:
            def __init__(self, rows):
                self.rows = rows

            def cpu(self):
                return self

            def tolist(self):
                return self.rows

        class FlakyTorch:
            uint16 = object()
            int32 = object()

            def __init__(self):
                self.shift_calls = 0
                self.bincount_calls = 0

            def bitwise_right_shift(self, tensor, _bits):
                self.shift_calls += 1
                if self.shift_calls == 1:
                    raise RuntimeError("transient failure")
                return tensor

            def bitwise_and(self, tensor, _mask):
                return tensor

            def bincount(self, _tensor, minlength):
                counts = [0] * minlength
                counts[self.bincount_calls + 1] = 1
                self.bincount_calls += 1
                return FakeCounts(counts)

            def stack(self, tensors):
                return FakeStack([tensor.values for tensor in tensors])

        fake_torch = FlakyTorch()
        source = (FakePlane(), FakePlane())
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            with patch.dict(
                os.environ,
                {"UNCHAIN_KV_TRACE": str(trace_path)},
                clear=True,
            ):
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                )
                connector = UnchainKVConnector(config, "worker")

                connector._trace_bf16_exponents(source, "tx", 8, fake_torch)
                connector._trace_bf16_exponents(source, "tx", 8, fake_torch)

            rows = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            [row["event"] for row in rows],
            ["bf16_exponent_stats_error", "bf16_exponent_stats"],
        )

    def test_bf16_exponent_trace_swallows_error_writer_failure(self):
        class BrokenTorch:
            uint16 = object()
            int32 = object()

            @staticmethod
            def bitwise_and(tensor, _mask):
                return tensor

            @staticmethod
            def bitwise_right_shift(_tensor, _bits):
                raise RuntimeError("stats failed")

        class FakePlane:
            def detach(self):
                return self

            def view(self, _dtype):
                return self

            def reshape(self, _size):
                return self

            def to(self, _dtype):
                return self

        class RaisingTrace:
            def __init__(self):
                self.events = []

            def event(self, name, **_fields):
                self.events.append(name)
                raise OSError("trace unavailable")

        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
        )
        with patch.dict(os.environ, {}, clear=True):
            connector = UnchainKVConnector(config, "worker")
        connector.trace = RaisingTrace()

        result = connector._trace_bf16_exponents(
            (FakePlane(), FakePlane()), "tx", 9, BrokenTorch()
        )

        self.assertIsNone(result)
        self.assertEqual(connector.trace.events, ["bf16_exponent_stats_error"])

    def test_bf16_exponent_trace_errors_do_not_break_codec_diagnostics(self):
        class BrokenTorch:
            uint16 = object()
            int32 = object()

            @staticmethod
            def bitwise_and(tensor, _mask):
                return tensor

            @staticmethod
            def bitwise_right_shift(_tensor, _bits):
                raise RuntimeError("diagnostic failed")

        class FakePlane:
            def detach(self):
                return self

            def view(self, _dtype):
                return self

            def reshape(self, _size):
                return self

            def to(self, _dtype):
                return self

        source = (FakePlane(), FakePlane())
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            with patch.dict(
                os.environ,
                {"UNCHAIN_KV_TRACE": str(trace_path)},
                clear=True,
            ):
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                )
                connector = UnchainKVConnector(config, "worker")

                connector._trace_bf16_exponents(source, "tx", 8, BrokenTorch())

            row = json.loads(trace_path.read_text(encoding="utf-8"))

        self.assertEqual(row["event"], "bf16_exponent_stats_error")
        self.assertEqual(row["transfer"], "tx")
        self.assertEqual(row["layer"], 8)
        self.assertIn("diagnostic failed", row["error"])

    def test_splitzip_bf16_fixed5_overflow_downgrades_layer(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_codec = os.environ.get("UNCHAIN_KV_CODEC")
        old_slots = os.environ.get("UNCHAIN_KV_CODEC_GPU_SLOTS")
        old_fixed5 = os.environ.get("UNCHAIN_KV_SPLITZIP_FIXED5")
        old_trace_cuda = os.environ.get("UNCHAIN_KV_TRACE_CUDA")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
                os.environ["UNCHAIN_KV_CODEC"] = "splitzip_bf16"
                os.environ["UNCHAIN_KV_CODEC_GPU_SLOTS"] = "1"
                os.environ["UNCHAIN_KV_SPLITZIP_FIXED5"] = "1"
                os.environ["UNCHAIN_KV_TRACE_CUDA"] = "1"
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                )
                connector = UnchainKVConnector(config, "worker")

                class FakeDType:
                    def __str__(self):
                        return "torch.bfloat16"

                class FakeTensor:
                    dtype = FakeDType()
                    device = "cuda:0"

                    def __init__(self, size=1 << 20):
                        self.size = size

                    def numel(self):
                        return self.size // 2

                    def element_size(self):
                        return 2

                    def __getitem__(self, _key):
                        return self

                    def detach(self):
                        return self

                    def numpy(self):
                        return bytearray(b"zip!")

                order = []
                stream = object()

                class FakeEvent:
                    created = []

                    def __init__(self, enable_timing=False):
                        self.enable_timing = enable_timing
                        self.created.append(self)

                    def record(self, recorded_stream):
                        order.append(("record", self, recorded_stream))

                fake_torch = SimpleNamespace(
                    cuda=SimpleNamespace(
                        Event=FakeEvent,
                        current_stream=lambda _device: stream,
                    )
                )
                fake_source = FakeTensor()
                fake_out = FakeTensor()
                calls = []
                connector._native_tensor_for_codec = (
                    lambda _kv_layer, _block_ids, _torch: fake_source
                )
                connector._stage_cpu = lambda tensor: tensor
                connector._acquire_codec_buffer = lambda *_args: SimpleNamespace(
                    tensor=fake_out
                )
                import unchain_kv.splitzip_cuda as splitzip_cuda

                old_encode = splitzip_cuda.encode_bf16
                splitzip_cuda.encode_bf16 = lambda _source, _out, bits=6: (
                    calls.append(bits)
                    or order.append(("encode", bits))
                    or (0 if bits == 5 else 4)
                )
                try:
                    with patch.dict(sys.modules, {"torch": fake_torch}):
                        encoded = connector._splitzip_bf16_native_blocks(
                            object(), [0, 1], 3
                        )
                        connector._splitzip_bf16_native_blocks(object(), [0, 1], 3)
                finally:
                    splitzip_cuda.encode_bf16 = old_encode

                self.assertIsNotNone(encoded)
                self.assertEqual(calls, [5, 6, 6])
                self.assertIn(3, connector._splitzip_fixed6_layers)
                payload = encoded[0]
                self.assertEqual(payload.bits, 6)
                self.assertTrue(payload.fallback)
                self.assertEqual(payload.encoded_bytes, 4)
                self.assertEqual(payload.encode_timing, tuple(FakeEvent.created[:2]))
                self.assertTrue(all(event.enable_timing for event in FakeEvent.created))
                self.assertEqual(
                    order[:4],
                    [
                        ("record", FakeEvent.created[0], stream),
                        ("encode", 5),
                        ("encode", 6),
                        ("record", FakeEvent.created[1], stream),
                    ],
                )
        finally:
            for name, old in (
                ("UNCHAIN_KV_TRACE", old_trace),
                ("UNCHAIN_KV_CODEC", old_codec),
                ("UNCHAIN_KV_CODEC_GPU_SLOTS", old_slots),
                ("UNCHAIN_KV_SPLITZIP_FIXED5", old_fixed5),
                ("UNCHAIN_KV_TRACE_CUDA", old_trace_cuda),
            ):
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old

    def test_splitzip_bf16_ready_payload_traces_cuda_timing_once(self):
        class FakeEvent:
            def __init__(self, elapsed_ms=0.0):
                self.elapsed_ms = elapsed_ms
                self.synchronize_calls = 0

            def synchronize(self):
                self.synchronize_calls += 1

            def elapsed_time(self, end):
                return end.elapsed_ms

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            with patch.dict(
                os.environ,
                {
                    "UNCHAIN_KV_TRACE": str(trace_path),
                    "UNCHAIN_KV_TRACE_CUDA": "1",
                },
                clear=True,
            ):
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                )
                connector = UnchainKVConnector(config, "worker")
                encode_start = FakeEvent()
                encode_end = FakeEvent(4.0)
                d2h_start = FakeEvent()
                ready = FakeEvent(2.0)
                payload = vllm_connector._SplitzipCudaPayload(
                    memoryview(b"zip"),
                    [ready],
                    layer_index=4,
                    encode_timing=(encode_start, encode_end),
                    d2h_timing=(d2h_start, ready),
                    raw_bytes=12,
                    encoded_bytes=3,
                    bits=5,
                    fallback=False,
                    payload_start=10.0,
                )

                with patch.object(
                    vllm_connector.time,
                    "perf_counter",
                    side_effect=[10.25, 99.0],
                ):
                    result = connector._ready_payload(payload)

            rows = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]

        timing_rows = [
            row for row in rows if row["event"] == "splitzip_payload_ready"
        ]
        self.assertEqual(result.tobytes(), b"zip")
        self.assertEqual(len(timing_rows), 1)
        self.assertEqual(
            {
                key: timing_rows[0][key]
                for key in (
                    "layer",
                    "raw_bytes",
                    "encoded_bytes",
                    "bits",
                    "fallback",
                    "encode_gpu_s",
                    "d2h_gpu_s",
                    "payload_elapsed_s",
                )
            },
            {
                "layer": 4,
                "raw_bytes": 12,
                "encoded_bytes": 3,
                "bits": 5,
                "fallback": False,
                "encode_gpu_s": 0.004,
                "d2h_gpu_s": 0.002,
                "payload_elapsed_s": 0.25,
            },
        )
        self.assertEqual(ready.synchronize_calls, 1)
        self.assertEqual(encode_start.synchronize_calls, 0)
        self.assertEqual(encode_end.synchronize_calls, 0)
        self.assertEqual(d2h_start.synchronize_calls, 0)

    def test_splitzip_bf16_ready_payload_overflow_falls_back_to_fixed6(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_trace_cuda = os.environ.get("UNCHAIN_KV_TRACE_CUDA")
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            os.environ["UNCHAIN_KV_TRACE"] = str(trace_path)
            os.environ["UNCHAIN_KV_TRACE_CUDA"] = "1"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")

            class FakeEvent:
                created = []
                fallback_elapsed_ms = iter((0.0, 5.0, 0.0, 2.0))

                def __init__(self, elapsed_ms=None, enable_timing=False):
                    self.elapsed_ms = (
                        next(self.fallback_elapsed_ms)
                        if elapsed_ms is None
                        else elapsed_ms
                    )
                    self.enable_timing = enable_timing
                    self.synchronize_calls = 0
                    self.recorded_streams = []
                    self.created.append(self)

                def synchronize(self):
                    self.synchronize_calls += 1

                def elapsed_time(self, end):
                    return end.elapsed_ms

                def record(self, stream):
                    self.recorded_streams.append(stream)

            class FakeOut:
                def __getitem__(self, _key):
                    return self

                def detach(self):
                    return self

                def cpu(self):
                    return self

                def numpy(self):
                    return bytearray(b"fixed6")

            stream = object()
            source = SimpleNamespace(device="cuda:0")
            out = FakeOut()
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(
                    Event=FakeEvent,
                    current_stream=lambda _device: stream,
                )
            )
            calls = []
            import unchain_kv.splitzip_cuda as splitzip_cuda

            old_encode = splitzip_cuda.encode_bf16
            splitzip_cuda.encode_bf16 = lambda src, dst: calls.append((src, dst)) or 6
            try:
                encode_start = FakeEvent(0.0)
                encode_end = FakeEvent(3.0)
                d2h_start = FakeEvent(0.0)
                ready = FakeEvent(1.0)
                payload = vllm_connector._SplitzipCudaPayload(
                    memoryview(bytearray([255])),
                    [ready],
                    source,
                    out,
                    7,
                    encode_timing=(encode_start, encode_end),
                    d2h_timing=(d2h_start, ready),
                    raw_bytes=16,
                    encoded_bytes=1,
                    bits=4,
                    fallback=False,
                    payload_start=time.perf_counter(),
                    codec_buffer=vllm_connector._CodecBuffer(out, reserved=True),
                    top16_epoch=0,
                )
                self.assertTrue(connector._send_payload_ready(payload))
                self.assertTrue(payload.codec_buffer.reserved)
                with patch.dict(sys.modules, {"torch": fake_torch}):
                    result = connector._ready_payload(payload)
                rows = [
                    json.loads(line)
                    for line in trace_path.read_text(encoding="utf-8").splitlines()
                ]
            finally:
                splitzip_cuda.encode_bf16 = old_encode
                if old_trace is None:
                    os.environ.pop("UNCHAIN_KV_TRACE", None)
                else:
                    os.environ["UNCHAIN_KV_TRACE"] = old_trace
                if old_trace_cuda is None:
                    os.environ.pop("UNCHAIN_KV_TRACE_CUDA", None)
                else:
                    os.environ["UNCHAIN_KV_TRACE_CUDA"] = old_trace_cuda

            self.assertEqual(calls, [(source, out)])
            self.assertEqual(result.tobytes(), b"fixed6")
            self.assertIsNone(payload.codec_buffer)
            self.assertEqual(connector._splitzip_top16_cooldowns, {7: 1})
            self.assertEqual(connector._splitzip_top16_windows, {7: 1})
            self.assertNotIn(7, connector._splitzip_fixed6_layers)
            timing_rows = [
                row for row in rows if row["event"] == "splitzip_payload_ready"
            ]
            self.assertEqual(len(timing_rows), 1)
            self.assertTrue(timing_rows[0]["fallback"])
            self.assertEqual(timing_rows[0]["bits"], 6)
            self.assertEqual(timing_rows[0]["encoded_bytes"], 6)
            self.assertEqual(timing_rows[0]["encode_gpu_s"], 0.008)
            self.assertEqual(timing_rows[0]["d2h_gpu_s"], 0.003)
            self.assertEqual(ready.synchronize_calls, 1)
            self.assertEqual(encode_start.synchronize_calls, 0)
            self.assertEqual(encode_end.synchronize_calls, 0)
            self.assertEqual(d2h_start.synchronize_calls, 0)
            self.assertEqual(len(FakeEvent.created), 8)
            self.assertEqual(
                [event.synchronize_calls for event in FakeEvent.created[4:]],
                [0, 1, 0, 1],
            )

    def test_writeback_overflow_uses_emergency_buffer_and_releases_pack(self):
        class FakeEvent:
            def __init__(self):
                self.synchronize_calls = 0

            def synchronize(self):
                self.synchronize_calls += 1

        class FakeOut:
            def __init__(self):
                self.data = bytearray(b"\x06ok")

            def __getitem__(self, _key):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.data

        allocations = []
        out = FakeOut()
        fake_torch = SimpleNamespace(
            uint8=object(),
            empty=lambda shape, dtype, device: (
                allocations.append((shape, dtype, device)) or out
            ),
        )
        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        ready = FakeEvent()
        pack = vllm_connector._GpuPackBuffer(object(), reserved=True)
        payload = vllm_connector._SplitzipCudaPayload(
            memoryview(bytearray([255])),
            [ready],
            source=SimpleNamespace(device="cuda:0"),
            layer_index=7,
            raw_bytes=16,
            encoded_bytes=1,
            bits=4,
            gpu_pack_buffers=[pack],
            writeback=True,
        )

        import unchain_kv.splitzip_cuda as splitzip_cuda

        with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(
            splitzip_cuda, "encode_bf16", return_value=3
        ):
            result = connector._ready_payload(payload)

        self.assertEqual(result.tobytes(), b"\x06ok")
        self.assertEqual(allocations, [((16,), fake_torch.uint8, "cuda:0")])
        self.assertEqual(ready.synchronize_calls, 1)
        self.assertFalse(pack.reserved)
        self.assertEqual(payload.gpu_pack_buffers, [])
        self.assertIsNone(payload.source)

    def test_splitzip_payload_exception_releases_gpu_buffers(self):
        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        codec = vllm_connector._CodecBuffer(object(), reserved=True)
        pack = vllm_connector._GpuPackBuffer(object(), reserved=True)
        connector._pending_codec_buffers.append(codec)
        payload = vllm_connector._SplitzipCudaPayload(
            memoryview(bytearray([255])),
            [],
            source=None,
            codec_buffer=codec,
            gpu_pack_buffers=[pack],
        )

        with self.assertRaisesRegex(RuntimeError, "codec overflow"):
            connector._ready_payload(payload)

        self.assertFalse(codec.reserved)
        self.assertFalse(pack.reserved)
        self.assertEqual(connector._pending_codec_buffers, [])
        self.assertIsNone(payload.codec_buffer)
        self.assertEqual(payload.gpu_pack_buffers, [])

    def test_splitzip_bf16_fixed6_overflow_falls_back_to_raw_mode(self):
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
        )
        connector = UnchainKVConnector(config, "worker")

        class FakeEvent:
            def synchronize(self):
                pass

        class FakeTensor:
            device = "cuda:0"
            dtype = object()

            def __init__(self, data):
                self.data = bytearray(data)

            def __getitem__(self, _key):
                return self

            def detach(self):
                return self

            def contiguous(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.data

        source = FakeTensor(b"raw!")
        out = FakeTensor(b"\xffx")
        payload = vllm_connector._SplitzipCudaPayload(
            memoryview(bytearray([255])),
            [FakeEvent()],
            source,
            out,
            layer_index=7,
            raw_bytes=4,
            encoded_bytes=1,
            bits=4,
        )

        import unchain_kv.splitzip_cuda as splitzip_cuda

        with patch.object(splitzip_cuda, "encode_bf16", return_value=2):
            result = connector._ready_payload(payload)

        self.assertEqual(result.tobytes(), b"\x06raw!")

    def test_splitzip_bf16_primary_fixed6_overflow_falls_back_to_raw_mode(self):
        class FakeDType:
            def __str__(self):
                return "torch.bfloat16"

        class FakeTensor:
            dtype = FakeDType()
            device = "cuda:0"

            def __init__(self, data):
                self.data = bytearray(data)

            def numel(self):
                return len(self.data) // 2

            def element_size(self):
                return 2

            def __getitem__(self, _key):
                return self

            def detach(self):
                return self

            def contiguous(self):
                return self

            def cpu(self):
                return self

            def view(self, _dtype):
                return self

            def numpy(self):
                return self.data

        raw = bytes(1 << 20)
        source = FakeTensor(raw)
        out = FakeTensor(b"\xffx")
        fake_torch = SimpleNamespace(uint16=object())
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
        )
        with patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_CODEC": "splitzip_bf16",
                "UNCHAIN_KV_CODEC_GPU_SLOTS": "1",
            },
        ):
            connector = UnchainKVConnector(config, "worker")
        connector._native_tensor_for_codec = lambda *_args: source
        connector._stage_cpu = lambda tensor: tensor
        connector._acquire_codec_buffer = lambda *_args: SimpleNamespace(tensor=out)

        import unchain_kv.splitzip_cuda as splitzip_cuda

        encode_calls = []
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(
            splitzip_cuda,
            "encode_bf16",
            side_effect=lambda *_args: encode_calls.append(1) or 2,
        ):
            encoded = connector._splitzip_bf16_native_blocks(
                object(), [0, 1], layer_index=7
            )
            result = connector._ready_payload(encoded[0])

        self.assertEqual(len(encode_calls), 1)
        self.assertEqual(len(result), len(raw) + 1)
        self.assertEqual(result[0], 6)
        self.assertEqual(result[1:], raw)

    def test_splitzip_fixed5_layer_list_parses_from_env(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_layers = os.environ.get("UNCHAIN_KV_SPLITZIP_FIXED5_LAYERS")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
                os.environ["UNCHAIN_KV_SPLITZIP_FIXED5_LAYERS"] = "1, 3,5"
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                )

                connector = UnchainKVConnector(config, "worker")

                self.assertEqual(connector.splitzip_fixed5_layers, {1, 3, 5})
        finally:
            for name, old in (
                ("UNCHAIN_KV_TRACE", old_trace),
                ("UNCHAIN_KV_SPLITZIP_FIXED5_LAYERS", old_layers),
            ):
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old

    def test_splitzip_bf16_defaults_to_one_gib_codec_buffer_cap(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_codec = os.environ.get("UNCHAIN_KV_CODEC")
        old_bytes = os.environ.get("UNCHAIN_KV_CODEC_GPU_BYTES")
        old_slots = os.environ.get("UNCHAIN_KV_CODEC_GPU_SLOTS")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
                os.environ["UNCHAIN_KV_CODEC"] = "splitzip_bf16"
                os.environ.pop("UNCHAIN_KV_CODEC_GPU_BYTES", None)
                os.environ.pop("UNCHAIN_KV_CODEC_GPU_SLOTS", None)
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                )

                connector = UnchainKVConnector(config, "worker")

                self.assertEqual(connector.codec_gpu_bytes, 1 << 30)
                self.assertEqual(connector.codec_gpu_slots, 0)
        finally:
            for name, old in (
                ("UNCHAIN_KV_TRACE", old_trace),
                ("UNCHAIN_KV_CODEC", old_codec),
                ("UNCHAIN_KV_CODEC_GPU_BYTES", old_bytes),
                ("UNCHAIN_KV_CODEC_GPU_SLOTS", old_slots),
            ):
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old

    def test_codec_buffer_pool_grows_within_byte_cap_and_bypasses_when_full(self):
        class FakeTensor:
            def __init__(self, size):
                self.size = size

            def numel(self):
                return self.size

        class FakeTorch:
            uint8 = object()

            @staticmethod
            def empty(shape, dtype, device):
                del dtype, device
                return FakeTensor(shape[0])

        class FakeDType:
            def __str__(self):
                return "torch.bfloat16"

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_TRACE": str(Path(tmp) / "trace.jsonl"),
                "UNCHAIN_KV_CODEC": "splitzip_bf16",
                "UNCHAIN_KV_CODEC_GPU_BYTES": "64",
                "UNCHAIN_KV_SPLITZIP_TOP16": "1",
            },
        ):
            os.environ.pop("UNCHAIN_KV_CODEC_GPU_SLOTS", None)
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")

            first = connector._acquire_codec_buffer(16, "cuda:0", FakeTorch)
            connector._release_pending_codec_buffers()
            grown = connector._acquire_codec_buffer(32, "cuda:0", FakeTorch)
            second = connector._acquire_codec_buffer(32, "cuda:0", FakeTorch)
            full = connector._acquire_codec_buffer(1, "cuda:0", FakeTorch)

            self.assertTrue(
                all(
                    entry is not first
                    for entries in connector._codec_pool.values()
                    for entry in entries
                )
            )
            self.assertIsNotNone(second)
            self.assertIsNone(full)
            self.assertEqual(connector._codec_pool_bytes(), 64)

            source = SimpleNamespace(
                dtype=FakeDType(),
                device="cuda:0",
                numel=lambda: 1 << 19,
                element_size=lambda: 2,
            )
            connector._native_tensor_for_codec = (
                lambda _kv_layer, _block_ids, _torch: source
            )
            connector._native_blocks_bytes = (
                lambda _kv_layer, _block_ids: (memoryview(b"rawraw00"), 4)
            )
            with patch.dict(sys.modules, {"torch": FakeTorch}):
                encoded = connector._codec_native_blocks(object(), [0, 1], 0)

            self.assertEqual(encoded[4], "raw_passthrough")
            self.assertEqual(connector._splitzip_top16_cooldowns, {})
            self.assertEqual(connector._splitzip_top16_unavailable_layers, set())
            reused = connector._acquire_codec_buffer(8, "cuda:0", FakeTorch)
            self.assertIn(reused, (grown, second))
            self.assertEqual(connector._codec_pool_bytes(), 64)

    def test_codec_and_pack_events_prevent_unsafe_reuse_without_cpu_sync(self):
        class FakeTensor:
            def __init__(self, size):
                self.size = size

            def numel(self):
                return self.size

        class FakeEvent:
            def __init__(self, ready=False):
                self.ready = ready
                self.synchronize_calls = 0

            def query(self):
                return self.ready

            def synchronize(self):
                self.synchronize_calls += 1

        class FakeStream:
            def __init__(self):
                self.waited = []

            def wait_event(self, event):
                self.waited.append(event)

        stream = FakeStream()
        fake_torch = SimpleNamespace(
            uint8=object(),
            cuda=SimpleNamespace(current_stream=lambda _device: stream),
        )
        with patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_TRACE_ENABLED": "0",
                "UNCHAIN_KV_CODEC": "splitzip_bf16",
                "UNCHAIN_KV_CODEC_GPU_BYTES": "32",
                "UNCHAIN_KV_GPU_PACK_LAYERS": "1",
            },
            clear=True,
        ):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        connector.codec_writeback = True
        codec_event = FakeEvent()
        codec = vllm_connector._CodecBuffer(FakeTensor(16), event=codec_event)
        connector._codec_pool["cuda:0"] = [codec]

        self.assertFalse(connector._buffer_entry_available(codec))
        self.assertIs(
            connector._wait_for_codec_buffer(8, "cuda:0", fake_torch), codec
        )
        self.assertEqual(stream.waited, [codec_event])
        self.assertEqual(codec_event.synchronize_calls, 0)

        pack_event = FakeEvent()
        pack = vllm_connector._GpuPackBuffer(FakeTensor(4), event=pack_event)
        connector._gpu_pack_pool[((4,), "bf16", "cuda:0")] = [pack]
        self.assertIsNone(
            connector._acquire_gpu_pack_buffer(
                (4,), "bf16", "cuda:0", fake_torch
            )
        )
        pack_event.ready = True
        self.assertIs(
            connector._acquire_gpu_pack_buffer(
                (4,), "bf16", "cuda:0", fake_torch
            ),
            pack,
        )
        self.assertEqual(pack_event.synchronize_calls, 0)

    def test_gpu_pack_reclaims_idle_buffer_for_new_shape(self):
        class FakeTensor:
            def __init__(self, shape):
                self.shape = shape

            def numel(self):
                return 1

            def element_size(self):
                return 1

        connector = object.__new__(UnchainKVConnector)
        connector.gpu_pack_layers = 1
        old_key = ((2, 8, 1), "bf16", "cuda:0")
        connector._gpu_pack_pool = {
            old_key: [vllm_connector._GpuPackBuffer(FakeTensor(old_key[0]))]
        }
        connector._pending_gpu_pack_buffers = []
        connector.trace = SimpleNamespace(event=lambda *_args, **_kwargs: None)
        fake_torch = SimpleNamespace(empty=lambda shape, **_kwargs: FakeTensor(shape))

        entry = connector._acquire_gpu_pack_buffer(
            (2, 16, 1), "bf16", "cuda:0", fake_torch
        )

        self.assertEqual(entry.tensor.shape, (2, 16, 1))
        self.assertNotIn(old_key, connector._gpu_pack_pool)
        self.assertEqual(
            sum(map(len, connector._gpu_pack_pool.values())), 1
        )

    def test_gpu_pack_byte_cap_reclaims_idle_shape_and_rejects_oversize(self):
        class DType:
            itemsize = 1

            def __str__(self):
                return "u8"

        class FakeTensor:
            def __init__(self, shape):
                self.shape = shape

            def numel(self):
                return math.prod(self.shape)

            def element_size(self):
                return 1

        rows = []
        connector = object.__new__(UnchainKVConnector)
        connector.gpu_pack_layers = 2
        connector.gpu_pack_bytes = 16
        connector._gpu_pack_pool = {
            ((8,), "u8", "cuda:0"): [
                vllm_connector._GpuPackBuffer(FakeTensor((8,)))
            ]
        }
        connector._pending_gpu_pack_buffers = []
        connector.trace = SimpleNamespace(
            event=lambda name, **fields: rows.append((name, fields))
        )
        fake_torch = SimpleNamespace(
            empty=lambda shape, **_kwargs: FakeTensor(tuple(shape))
        )

        entry = connector._acquire_gpu_pack_buffer(
            (12,), DType(), "cuda:0", fake_torch
        )
        self.assertEqual(entry.tensor.shape, (12,))
        self.assertLessEqual(connector._gpu_pack_pool_bytes(), 16)
        connector._release_pending_gpu_pack_buffers()
        self.assertIsNone(
            connector._acquire_gpu_pack_buffer(
                (17,), DType(), "cuda:0", fake_torch
            )
        )
        self.assertTrue(any(name == "gpu_pack_oversize" for name, _ in rows))

    def test_buffer_leases_pair_monotonically_across_reuse(self):
        class FakeTensor:
            shape = (8,)

            def numel(self):
                return 8

            def element_size(self):
                return 1

        class DType:
            itemsize = 1

            def __str__(self):
                return "u8"

        rows = []
        connector = object.__new__(UnchainKVConnector)
        connector.gpu_pack_layers = 1
        connector.gpu_pack_bytes = 8
        connector._gpu_pack_pool = {}
        connector._pending_gpu_pack_buffers = []
        connector.trace = SimpleNamespace(
            event=lambda name, **fields: rows.append((name, fields))
        )
        fake_torch = SimpleNamespace(empty=lambda *_args, **_kwargs: FakeTensor())

        first = connector._acquire_gpu_pack_buffer(
            (8,), DType(), "cuda:0", fake_torch
        )
        connector._release_pending_gpu_pack_buffers()
        second = connector._acquire_gpu_pack_buffer(
            (8,), DType(), "cuda:0", fake_torch
        )
        connector._release_pending_gpu_pack_buffers()

        self.assertIs(first, second)
        acquired = [fields["lease"] for name, fields in rows if name == "buffer_acquire"]
        released = [fields["lease"] for name, fields in rows if name == "buffer_release"]
        self.assertEqual(acquired, [1, 2])
        self.assertEqual(released, acquired)

    def test_pack_buffer_is_not_reusable_until_lease_release_is_recorded(self):
        class FakeTensor:
            shape = (8,)

            def numel(self):
                return 8

            def element_size(self):
                return 1

        class DType:
            itemsize = 1

            def __str__(self):
                return "u8"

        connector = object.__new__(UnchainKVConnector)
        connector.gpu_pack_layers = 1
        connector.gpu_pack_bytes = 8
        connector._gpu_pack_pool = {}
        connector._pending_gpu_pack_buffers = []
        connector.trace = SimpleNamespace(event=lambda *_args, **_kwargs: None)
        fake_torch = SimpleNamespace(empty=lambda *_args, **_kwargs: FakeTensor())
        entry = connector._acquire_gpu_pack_buffer(
            (8,), DType(), "cuda:0", fake_torch
        )
        observed = []

        def trace(name, **_fields):
            if name == "buffer_release":
                observed.append(
                    connector._acquire_gpu_pack_buffer(
                        (8,), DType(), "cuda:0", fake_torch
                    )
                )

        connector.trace = SimpleNamespace(event=trace)
        connector._release_pending_gpu_pack_buffers()

        self.assertEqual(observed, [None])
        self.assertIs(
            connector._acquire_gpu_pack_buffer((8,), DType(), "cuda:0", fake_torch),
            entry,
        )

    def test_buffer_release_marks_network_future_and_flushes_pack_timing(self):
        class FakeTensor:
            def numel(self):
                return 8

            def element_size(self):
                return 1

        rows = []
        connector = object.__new__(UnchainKVConnector)
        connector.trace = SimpleNamespace(
            event=lambda name, **fields: rows.append((name, fields))
        )
        future = Future()
        future.set_result(None)
        entry = vllm_connector._GpuPackBuffer(
            FakeTensor(), future=future, lease_id=7
        )

        self.assertTrue(connector._buffer_entry_available(entry))
        self.assertTrue(rows[-1][1]["waited_network_future"])

        start = SimpleNamespace(elapsed_time=lambda _end: 2.5)
        end = SimpleNamespace(query=lambda: True)
        connector._pending_codec_decode_events = []
        connector._pending_gpu_pack_events = [(3, 8, start, end)]
        connector._flush_codec_decode_events()

        self.assertEqual(connector._pending_gpu_pack_events, [])
        self.assertEqual(
            rows[-1],
            ("gpu_pack_done", {"layer": 3, "bytes": 8, "pack_gpu_s": 0.0025}),
        )

    def test_gpu_pack_wait_uses_cuda_event_without_cpu_sync(self):
        class DType:
            itemsize = 1

            def __str__(self):
                return "u8"

        class FakeTensor:
            def numel(self):
                return 8

            def element_size(self):
                return 1

        class Event:
            synchronize_calls = 0

            def query(self):
                return False

            def synchronize(self):
                self.synchronize_calls += 1

        event = Event()
        stream = SimpleNamespace(wait_event=lambda current: setattr(current, "ready", True))
        connector = object.__new__(UnchainKVConnector)
        connector.gpu_pack_layers = 1
        connector.gpu_pack_bytes = 8
        connector.request_spool_enabled = False
        connector._early_stage_active = False
        connector._gpu_pack_pool = {
            ((8,), "u8", "cuda:0"): [
                vllm_connector._GpuPackBuffer(FakeTensor(), event=event)
            ]
        }
        connector._pending_gpu_pack_buffers = []
        rows = []
        connector.trace = SimpleNamespace(
            event=lambda name, **fields: rows.append((name, fields))
        )
        fake_torch = SimpleNamespace(
            empty=lambda *_args, **_kwargs: FakeTensor(),
            cuda=SimpleNamespace(current_stream=lambda _device: stream),
        )

        entry = connector._wait_for_gpu_pack_buffer(
            (8,), DType(), "cuda:0", fake_torch
        )

        self.assertIsNotNone(entry)
        self.assertEqual(event.synchronize_calls, 0)
        self.assertTrue(any(name == "gpu_pack_credit_wait" for name, _ in rows))

    def test_gpu_pack_wait_advances_oldest_send_when_entry_is_still_owned(self):
        connector = object.__new__(UnchainKVConnector)
        connector.request_spool_enabled = False
        connector._early_stage_active = False
        connector._gpu_pack_pool = {}
        futures = [object(), object()]
        connector.send_futures = list(futures)
        sentinel = object()
        calls = []
        connector._acquire_gpu_pack_buffer = (
            lambda *_args: None if len(calls) < 2 else sentinel
        )
        connector._finish_send_future = (
            lambda current, done_only: calls.append((current, done_only))
        )
        rows = []
        connector.trace = SimpleNamespace(
            event=lambda name, **fields: rows.append((name, fields))
        )

        entry = connector._wait_for_gpu_pack_buffer(
            (8,), SimpleNamespace(itemsize=1), "cuda:0", SimpleNamespace()
        )

        self.assertIs(entry, sentinel)
        self.assertEqual(calls, [(futures[0], False), (futures[1], False)])
        self.assertEqual(rows[-1][1]["source"], "send_future")
        self.assertEqual(rows[-1][1]["futures"], 2)

    def test_pack_payload_readiness_releases_credit_before_network_send(self):
        class FakeTensor:
            def numel(self):
                return 8

            def element_size(self):
                return 1

        connector = object.__new__(UnchainKVConnector)
        connector.payload_ready_executor = vllm_connector.ThreadPoolExecutor(
            max_workers=1
        )
        connector._payload_ready_futures = []
        connector._gpu_pack_pool = {}
        connector._pending_gpu_pack_buffers = []
        connector.request_spool_enabled = False
        connector._early_stage_active = False
        connector.send_futures = []
        connector.trace_cuda_timing = False
        rows = []
        connector.trace = SimpleNamespace(
            event=lambda name, **fields: rows.append((name, fields))
        )
        pack = vllm_connector._GpuPackBuffer(FakeTensor(), reserved=True)
        payload = vllm_connector._SplitzipCudaPayload(
            memoryview(b"ready"),
            [],
            layer_index=3,
            gpu_pack_buffers=[pack],
        )

        args = connector._stage_ready_send_args(("tx", payload))
        ready = args[1]
        self.assertIsInstance(ready, vllm_connector._ReadyPayloadFuture)
        self.assertEqual(ready.future.result(), memoryview(b"ready"))
        self.assertFalse(pack.reserved)
        self.assertEqual(payload.gpu_pack_buffers, [])
        self.assertTrue(any(name == "payload_ready_done" for name, _ in rows))
        connector.payload_ready_executor.shutdown(wait=True)

    def test_gpu_pack_wait_prefers_payload_readiness_over_send_future(self):
        connector = object.__new__(UnchainKVConnector)
        connector.request_spool_enabled = False
        connector._early_stage_active = False
        ready = Future()
        ready.set_result(memoryview(b"ok"))
        connector._payload_ready_futures = [ready]
        connector.send_futures = [self.fail]
        sentinel = object()
        acquires = []

        def acquire(*_args):
            acquires.append(None)
            return None if len(acquires) == 1 else sentinel

        connector._acquire_gpu_pack_buffer = acquire
        connector._finish_send_future = lambda *_args: self.fail(
            "network future must not be drained"
        )
        rows = []
        connector.trace = SimpleNamespace(
            event=lambda name, **fields: rows.append((name, fields))
        )
        entry = connector._wait_for_gpu_pack_buffer(
            (8,), SimpleNamespace(itemsize=1), "cuda:0", SimpleNamespace()
        )

        self.assertIs(entry, sentinel)
        self.assertEqual(rows[-1][1]["source"], "payload_ready")

    def test_early_spool_waits_for_stage_credit_instead_of_partial_layer(self):
        connector = object.__new__(UnchainKVConnector)
        connector.request_spool_enabled = True
        connector._early_stage_active = True
        sentinel = object()
        attempts = []
        progress = []

        def acquire(*_args):
            attempts.append(None)
            return None if len(attempts) == 1 else sentinel

        connector._acquire_codec_buffer = acquire
        connector._wait_for_spool_stage_progress = (
            lambda: progress.append(None) or True
        )

        entry = connector._wait_for_codec_buffer(8, "cuda:0", SimpleNamespace())

        self.assertIs(entry, sentinel)
        self.assertEqual(len(progress), 1)


    def test_early_submission_is_recorded_before_batch_finishes(self):
        connector = object.__new__(UnchainKVConnector)
        connector.payload_ready_executor = None
        connector._payload_ready_futures = []
        connector._pending_send_futures = []
        connector.send_futures = []
        connector.send_inflight = 0
        connector.grant_window = 0
        connector._early_stage_active = True
        connector._early_sent_layers = set()
        connector.send_executor = vllm_connector.ThreadPoolExecutor(max_workers=1)
        try:
            future = connector._submit_send(lambda *_args: None, "tx", "req", 2)
            future.result()
        finally:
            connector.send_executor.shutdown(wait=True)

        self.assertIn(("tx", 2), connector._early_sent_layers)


    def test_tcp_producer_uses_bulk_layer_sender(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_transport = os.environ.get("UNCHAIN_KV_TRANSPORT")
        old_max_blocks = os.environ.get("UNCHAIN_KV_MAX_BLOCKS")
        old_tcp_timeout = os.environ.get("UNCHAIN_KV_TCP_CONNECT_TIMEOUT_S")
        old_tcp_retry = os.environ.get("UNCHAIN_KV_TCP_RETRY_INTERVAL_S")
        import unchain_kv.tcp_data as tcp_data

        old_send_layer_blocks = tcp_data.send_layer_blocks
        calls = []
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
                os.environ["UNCHAIN_KV_TRANSPORT"] = "tcp"
                os.environ["UNCHAIN_KV_MAX_BLOCKS"] = "0"
                os.environ["UNCHAIN_KV_TCP_CONNECT_TIMEOUT_S"] = "0.1"
                os.environ["UNCHAIN_KV_TCP_RETRY_INTERVAL_S"] = "0.01"
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                )
                connector = UnchainKVConnector(config, "worker")
                connector.metadata = PipelineMetadata([Session("tx", "req", [1, 2])])
                connector._blocks_bytes = lambda _kv_layer, block_ids: (b"xy", 1)
                tcp_data.send_layer_blocks = (
                    lambda _peer, transfer, request, layer, data, size, count: calls.append(
                        (transfer, request, layer, data, size, count)
                    )
                )

                connector.save_kv_layer("model.layers.2.self_attn", object(), None)
                connector.wait_for_save()

                self.assertEqual(calls, [("tx", "req", 2, b"xy", 1, 2)])
        finally:
            tcp_data.send_layer_blocks = old_send_layer_blocks
            if old_trace is None:
                os.environ.pop("UNCHAIN_KV_TRACE", None)
            else:
                os.environ["UNCHAIN_KV_TRACE"] = old_trace
            if old_transport is None:
                os.environ.pop("UNCHAIN_KV_TRANSPORT", None)
            else:
                os.environ["UNCHAIN_KV_TRANSPORT"] = old_transport
            if old_max_blocks is None:
                os.environ.pop("UNCHAIN_KV_MAX_BLOCKS", None)
            else:
                os.environ["UNCHAIN_KV_MAX_BLOCKS"] = old_max_blocks
            if old_tcp_timeout is None:
                os.environ.pop("UNCHAIN_KV_TCP_CONNECT_TIMEOUT_S", None)
            else:
                os.environ["UNCHAIN_KV_TCP_CONNECT_TIMEOUT_S"] = old_tcp_timeout
            if old_tcp_retry is None:
                os.environ.pop("UNCHAIN_KV_TCP_RETRY_INTERVAL_S", None)
            else:
                os.environ["UNCHAIN_KV_TCP_RETRY_INTERVAL_S"] = old_tcp_retry

    def test_tcp_layer_group_buffers_until_group_is_full(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_transport = os.environ.get("UNCHAIN_KV_TRANSPORT")
        old_group = os.environ.get("UNCHAIN_KV_LAYER_GROUP_SIZE")
        old_max_blocks = os.environ.get("UNCHAIN_KV_MAX_BLOCKS")
        import unchain_kv.tcp_data as tcp_data

        old_send_group = getattr(tcp_data, "send_layer_group_blocks", None)
        calls = []
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
                os.environ["UNCHAIN_KV_TRANSPORT"] = "tcp"
                os.environ["UNCHAIN_KV_LAYER_GROUP_SIZE"] = "2"
                os.environ["UNCHAIN_KV_MAX_BLOCKS"] = "0"
                tcp_data.send_layer_group_blocks = lambda *args, **kwargs: calls.append(
                    (args, kwargs)
                )
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                )
                connector = UnchainKVConnector(config, "worker")
                connector.metadata = PipelineMetadata([Session("tx", "req", [1, 2])])
                connector._blocks_bytes = lambda _kv_layer, _block_ids: (b"abcd", 2)

                connector.save_kv_layer("model.layers.0.self_attn", object(), None)
                self.assertEqual(calls, [])

                connector.save_kv_layer("model.layers.1.self_attn", object(), None)
                connector.wait_for_save()

                self.assertEqual(len(calls), 1)
                args, kwargs = calls[0]
                self.assertEqual(args[3], 0)
                self.assertEqual(args[4], [b"abcd", b"abcd"])
                self.assertEqual(kwargs["layout"], "block_major")
        finally:
            if old_send_group is None:
                delattr(tcp_data, "send_layer_group_blocks")
            else:
                tcp_data.send_layer_group_blocks = old_send_group
            if old_trace is None:
                os.environ.pop("UNCHAIN_KV_TRACE", None)
            else:
                os.environ["UNCHAIN_KV_TRACE"] = old_trace
            if old_transport is None:
                os.environ.pop("UNCHAIN_KV_TRANSPORT", None)
            else:
                os.environ["UNCHAIN_KV_TRANSPORT"] = old_transport
            if old_group is None:
                os.environ.pop("UNCHAIN_KV_LAYER_GROUP_SIZE", None)
            else:
                os.environ["UNCHAIN_KV_LAYER_GROUP_SIZE"] = old_group
            if old_max_blocks is None:
                os.environ.pop("UNCHAIN_KV_MAX_BLOCKS", None)
            else:
                os.environ["UNCHAIN_KV_MAX_BLOCKS"] = old_max_blocks

    def test_send_inflight_credit_waits_for_oldest_future(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_transport = os.environ.get("UNCHAIN_KV_TRANSPORT")
        old_inflight = os.environ.get("UNCHAIN_KV_SEND_INFLIGHT")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
                os.environ["UNCHAIN_KV_TRANSPORT"] = "tcp"
                os.environ["UNCHAIN_KV_SEND_INFLIGHT"] = "1"
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                )
                connector = UnchainKVConnector(config, "worker")
                future = Future()
                connector.send_futures.append(future)
                done = []
                thread = threading.Thread(
                    target=lambda: (connector._wait_send_credit(), done.append(True))
                )

                thread.start()
                thread.join(0.05)
                self.assertEqual(done, [])

                future.set_result(None)
                thread.join(1)

                self.assertEqual(done, [True])
                self.assertEqual(connector.send_futures, [])
        finally:
            if old_trace is None:
                os.environ.pop("UNCHAIN_KV_TRACE", None)
            else:
                os.environ["UNCHAIN_KV_TRACE"] = old_trace
            if old_transport is None:
                os.environ.pop("UNCHAIN_KV_TRANSPORT", None)
            else:
                os.environ["UNCHAIN_KV_TRANSPORT"] = old_transport
            if old_inflight is None:
                os.environ.pop("UNCHAIN_KV_SEND_INFLIGHT", None)
            else:
                os.environ["UNCHAIN_KV_SEND_INFLIGHT"] = old_inflight



    def test_host_mirror_replaces_idle_buffer_for_new_shape(self):
        class FakeTorch:
            @staticmethod
            def empty(shape, dtype, device, pin_memory):
                return (tuple(shape), dtype, device, pin_memory)

        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
        )
        with patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_TRANSPORT": "tcp",
                "UNCHAIN_KV_HOST_MIRROR_LAYERS": "1",
            },
        ):
            connector = UnchainKVConnector(config, "worker")
            first_tensor = SimpleNamespace(shape=(2, 4), dtype="bf16")
            second_tensor = SimpleNamespace(shape=(2, 8), dtype="bf16")

            first = connector._acquire_pinned_stage_buffer(first_tensor, FakeTorch)
            first.future = Future()
            first.future.set_result(None)
            first.reserved = False

            second = connector._acquire_pinned_stage_buffer(second_tensor, FakeTorch)

            self.assertIsNotNone(second)
            self.assertEqual(
                sum(len(entries) for entries in connector._pinned_stage_pool.values()),
                1,
            )
            self.assertNotIn(((2, 4), "bf16"), connector._pinned_stage_pool)

    def test_host_byte_credit_backpressures_and_reuses_completed_send(self):
        class FakeTensor:
            dtype = "u8"

            def __init__(self, shape):
                self.shape = tuple(shape)

            def numel(self):
                size = 1
                for value in self.shape:
                    size *= value
                return size

            def element_size(self):
                return 1

        class FakeTorch:
            @staticmethod
            def empty(shape, dtype, device, pin_memory):
                del dtype, device, pin_memory
                return FakeTensor(shape)

        class CompletingFuture:
            def __init__(self):
                self.finished = False
                self.result_calls = 0

            def done(self):
                return self.finished

            def result(self):
                self.result_calls += 1
                self.finished = True

        with patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_TRACE_ENABLED": "0",
                "UNCHAIN_KV_TRANSPORT": "tcp",
                "UNCHAIN_KV_HOST_MIRROR_LAYERS": "1",
                "UNCHAIN_KV_HOST_MIRROR_BYTES": "8",
            },
            clear=True,
        ):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        try:
            tensor = FakeTensor((8,))
            first = connector._acquire_pinned_stage_buffer(tensor, FakeTorch)
            future = CompletingFuture()
            connector._bind_stage_buffers(future)
            connector.send_futures.append(future)

            reused = connector._wait_for_pinned_stage_buffer(tensor, FakeTorch)

            self.assertIs(reused, first)
            self.assertEqual(future.result_calls, 1)
            self.assertEqual(connector.send_futures, [])
            self.assertFalse(hasattr(future, "_kvx_host_bytes"))
            self.assertIsNone(
                connector._wait_for_pinned_stage_buffer(FakeTensor((9,)), FakeTorch)
            )
        finally:
            connector.send_executor.shutdown(wait=True)

    def test_writeback_reserves_host_credit_before_overwriting_kv(self):
        class FakeDType:
            def __str__(self):
                return "torch.bfloat16"

        class FakeTensor:
            device = "cuda:0"

            def __init__(self, pointer, size):
                self.pointer = pointer
                self.size = size
                self.copies = []

            def data_ptr(self):
                return self.pointer

            def numel(self):
                return self.size

            def element_size(self):
                return 1

            def __getitem__(self, _key):
                return self

            def copy_(self, source, non_blocking):
                self.copies.append((source, non_blocking))

        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        segment = FakeTensor(2000, 16)
        source = FakeTensor(1000, 8)
        codec = vllm_connector._CodecBuffer(FakeTensor(3000, 16), reserved=True)
        connector._kv_writeback_segments = lambda *_args: [segment]
        connector._wait_for_pinned_stage_buffer = lambda *_args: None
        kv_layer = SimpleNamespace(
            device=SimpleNamespace(type="cuda"), dtype=FakeDType()
        )

        result = connector._try_splitzip_writeback(
            kv_layer, [0], source, codec, 4, 8, 0, object()
        )

        self.assertIsNone(result)
        self.assertEqual(segment.copies, [])
        self.assertTrue(codec.reserved)

    def test_writeback_skips_fragmented_scatter_before_overwriting_kv(self):
        class FakeDType:
            def __str__(self):
                return "torch.bfloat16"

        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        connector._kv_writeback_segments = lambda *_args: (_ for _ in ()).throw(
            AssertionError("fragmented preflight must not materialize CUDA views")
        )
        connector._wait_for_pinned_stage_buffer = lambda *_args: (_ for _ in ()).throw(
            AssertionError("fragmented writeback must not reserve host credit")
        )
        kv_layer = SimpleNamespace(
            device=SimpleNamespace(type="cuda"), dtype=FakeDType()
        )
        codec = vllm_connector._CodecBuffer(object(), reserved=True)

        result = connector._try_splitzip_writeback(
            kv_layer, [0, 2], object(), codec, 4, 8, 0, object()
        )

        self.assertIsNone(result)
        self.assertTrue(codec.reserved)

    def test_early_stage_defers_writeback_until_attention_done(self):
        class FakeTensor:
            device = "cuda:0"

            def __init__(self, pointer, size):
                self.pointer = pointer
                self.size = size
                self.copies = []

            def data_ptr(self):
                return self.pointer

            def numel(self):
                return self.size

            def element_size(self):
                return 1

            def __getitem__(self, _key):
                return self

            def copy_(self, source, non_blocking):
                self.copies.append((source, non_blocking))

        class FakeEvent:
            def __init__(self):
                self.recorded = []

            def record(self, stream):
                self.recorded.append(stream)

        class FakeStream:
            def __init__(self):
                self.waited = []

            def wait_event(self, event):
                self.waited.append(event)

        stream = FakeStream()
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                current_stream=lambda _device: stream,
                Event=FakeEvent,
            )
        )
        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        source = FakeTensor(1000, 8)
        segment = FakeTensor(2000, 8)
        codec = vllm_connector._CodecBuffer(FakeTensor(3000, 8), reserved=True)
        ready = FakeEvent()
        payload = vllm_connector._SplitzipCudaPayload(
            memoryview(b"top!"),
            [ready],
            source=source,
            raw_bytes=8,
            encoded_bytes=4,
            bits=4,
            codec_buffer=codec,
            deferred_writeback=True,
        )
        connector._pending_codec_buffers.append(codec)
        connector._deferred_codec_writebacks[("tx", 2)] = ([1, 2], payload)
        connector._kv_writeback_segments = lambda *_args: [segment]

        with patch.dict(sys.modules, {"torch": fake_torch}):
            connector._finish_deferred_codec_writebacks(
                object(), 2, [Session("tx", "req", [1, 2])]
            )

        self.assertEqual(stream.waited, [ready])
        self.assertEqual(segment.copies, [(codec.tensor, True)])
        self.assertFalse(codec.reserved)
        self.assertIsInstance(codec.event, FakeEvent)
        self.assertEqual(codec.event.recorded, [stream])
        self.assertIsNone(payload.codec_buffer)
        self.assertFalse(payload.deferred_writeback)

    def test_fragmented_deferred_writeback_releases_pack_at_d2h_ready(self):
        class FakeEvent:
            def query(self):
                return True

        class FakeStream:
            def __init__(self):
                self.waited = []

            def wait_event(self, event):
                self.waited.append(event)

        stream = FakeStream()
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(current_stream=lambda _device: stream)
        )
        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        ready = FakeEvent()
        codec = vllm_connector._CodecBuffer(
            SimpleNamespace(device="cuda:0"), reserved=True
        )
        pack = vllm_connector._GpuPackBuffer(object(), reserved=True)
        payload = vllm_connector._SplitzipCudaPayload(
            memoryview(b"top!"),
            [ready],
            source=object(),
            raw_bytes=8,
            encoded_bytes=4,
            bits=4,
            codec_buffer=codec,
            gpu_pack_buffers=[pack],
            deferred_writeback=True,
        )
        connector._deferred_codec_writebacks[("tx", 2)] = ([1, 3], payload)

        self.assertTrue(connector._send_payload_ready(payload))
        self.assertFalse(pack.reserved)
        self.assertTrue(codec.reserved)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            connector._finish_deferred_codec_writebacks(
                object(), 2, [Session("tx", "req", [1, 3])]
            )

        self.assertEqual(stream.waited, [ready])
        self.assertFalse(codec.reserved)
        self.assertIs(codec.event, ready)
        self.assertIsNone(payload.codec_buffer)
        self.assertEqual(payload.gpu_pack_buffers, [])
        self.assertFalse(payload.deferred_writeback)

    def test_deferred_top16_overflow_stays_owned_by_sender(self):
        class FakeEvent:
            def query(self):
                return True

        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        ready = FakeEvent()
        codec = vllm_connector._CodecBuffer(object(), reserved=True)
        pack = vllm_connector._GpuPackBuffer(object(), reserved=True)
        payload = vllm_connector._SplitzipCudaPayload(
            memoryview(bytearray([255])),
            [ready],
            source=object(),
            out=object(),
            layer_index=2,
            codec_buffer=codec,
            gpu_pack_buffers=[pack],
            deferred_writeback=True,
        )
        connector._deferred_codec_writebacks[("tx", 2)] = ([1, 3], payload)
        connector._kv_writeback_segments = lambda *_args: self.fail(
            "overflow payload must not be written back"
        )

        self.assertTrue(connector._send_payload_ready(payload))
        with patch.dict(sys.modules, {"torch": SimpleNamespace()}):
            connector._finish_deferred_codec_writebacks(
                object(), 2, [Session("tx", "req", [1, 3])]
            )

        self.assertFalse(payload.deferred_writeback)
        self.assertTrue(codec.reserved)
        self.assertTrue(pack.reserved)
        connector._release_failed_send_payload(payload)
        self.assertFalse(codec.reserved)
        self.assertFalse(pack.reserved)

    def test_deferred_writeback_exception_releases_codec_credit(self):
        class ReadyEvent:
            def query(self):
                return True

        class RecordedEvent:
            def __init__(self):
                self.stream = None

            def record(self, stream):
                self.stream = stream

        class FakeStream:
            def wait_event(self, _event):
                pass

        stream = FakeStream()
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                current_stream=lambda _device: stream,
                Event=RecordedEvent,
            )
        )
        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        codec = vllm_connector._CodecBuffer(
            SimpleNamespace(device="cuda:0"), reserved=True
        )
        payload = vllm_connector._SplitzipCudaPayload(
            memoryview(b"top!"),
            [ReadyEvent()],
            source=object(),
            layer_index=2,
            codec_buffer=codec,
            deferred_writeback=True,
        )
        connector._deferred_codec_writebacks[("tx", 2)] = ([1, 2], payload)
        connector._kv_writeback_segments = lambda *_args: (_ for _ in ()).throw(
            RuntimeError("copy failed")
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            connector._finish_deferred_codec_writebacks(
                object(), 2, [Session("tx", "req", [1, 2])]
            )

        self.assertFalse(codec.reserved)
        self.assertIsInstance(codec.event, RecordedEvent)
        self.assertIs(codec.event.stream, stream)
        self.assertIsNone(payload.codec_buffer)
        self.assertFalse(payload.deferred_writeback)

    def test_writeback_segments_scatter_non_contiguous_block_runs(self):
        class Segment:
            def __init__(self, key):
                self.key = key

            def is_contiguous(self):
                return True

            def view(self, _dtype):
                return self

            def reshape(self, _size):
                return self

        class KvLayer:
            shape = (2, 10, 4)

            def dim(self):
                return 3

            def __getitem__(self, key):
                plane, blocks = key
                return Segment((plane, blocks.start, blocks.stop))

        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )

        segments = connector._kv_writeback_segments(
            KvLayer(), [1, 2, 5, 7, 8], SimpleNamespace(uint8=object())
        )

        self.assertEqual(
            [segment.key for segment in segments],
            [
                (0, 1, 3),
                (0, 5, 6),
                (0, 7, 9),
                (1, 1, 3),
                (1, 5, 6),
                (1, 7, 9),
            ],
        )
        self.assertIsNone(
            connector._kv_writeback_segments(
                KvLayer(), [1, 1], SimpleNamespace(uint8=object())
            )
        )














    def test_single_send_worker_skips_unready_payload(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_transport = os.environ.get("UNCHAIN_KV_TRANSPORT")
        old_workers = os.environ.get("UNCHAIN_KV_SEND_WORKERS")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
                os.environ["UNCHAIN_KV_TRANSPORT"] = "tcp"
                os.environ["UNCHAIN_KV_SEND_WORKERS"] = "1"
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                )
                connector = UnchainKVConnector(config, "worker")
                sent = []
                gate = threading.Event()

                class BlockingEvent:
                    def query(self):
                        return gate.is_set()

                    def synchronize(self):
                        gate.wait(1.0)

                def send(layer, payload):
                    connector._ready_payload(payload)
                    sent.append(layer)

                blocked = vllm_connector._StagedPayload(
                    memoryview(bytearray(b"slow")), [BlockingEvent()]
                )
                ready = memoryview(bytearray(b"fast"))
                try:
                    slow_future = connector._submit_send(send, 1, blocked)
                    fast_future = connector._submit_send(send, 2, ready)
                    deadline = time.time() + 0.5
                    while time.time() < deadline and not fast_future.done():
                        connector._check_send_futures(done_only=True)
                        time.sleep(0.01)

                    self.assertEqual(sent, [2])
                    self.assertFalse(slow_future.done())

                    gate.set()
                    connector._check_send_futures(done_only=False)
                    self.assertEqual(sent, [2, 1])
                finally:
                    gate.set()
                    connector.send_executor.shutdown(wait=True)
        finally:
            for name, old in (
                ("UNCHAIN_KV_TRACE", old_trace),
                ("UNCHAIN_KV_TRANSPORT", old_transport),
                ("UNCHAIN_KV_SEND_WORKERS", old_workers),
            ):
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old

    def test_grant_window_submits_unready_payload_without_later_drain(self):
        with patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_TRANSPORT": "tcp",
                "UNCHAIN_KV_SEND_WORKERS": "1",
                "UNCHAIN_KV_GRANT_WINDOW": "1",
            },
        ):
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")
        gate = threading.Event()
        started = threading.Event()

        class BlockingEvent:
            def query(self):
                return gate.is_set()

            def synchronize(self):
                started.set()
                gate.wait(1.0)

        def send(payload):
            connector._ready_payload(payload)

        payload = vllm_connector._StagedPayload(
            memoryview(bytearray(b"slow")), [BlockingEvent()]
        )
        try:
            future = connector._submit_send(send, payload)
            self.assertTrue(started.wait(0.5))
            self.assertFalse(future.done())

            gate.set()
            future.result(1.0)
        finally:
            gate.set()
            connector.send_executor.shutdown(wait=True)

    def test_pinned_staging_flag_is_configurable(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_pinned = os.environ.get("UNCHAIN_KV_PINNED_STAGING")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
                os.environ["UNCHAIN_KV_PINNED_STAGING"] = "1"
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                )

                connector = UnchainKVConnector(config, "worker")

                self.assertTrue(connector.pinned_staging)
        finally:
            if old_trace is None:
                os.environ.pop("UNCHAIN_KV_TRACE", None)
            else:
                os.environ["UNCHAIN_KV_TRACE"] = old_trace
            if old_pinned is None:
                os.environ.pop("UNCHAIN_KV_PINNED_STAGING", None)
            else:
                os.environ["UNCHAIN_KV_PINNED_STAGING"] = old_pinned


    def test_pinned_stage_buffers_bind_to_send_future(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                )
                connector = UnchainKVConnector(config, "worker")
                entry = vllm_connector._PinnedStageBuffer(object(), reserved=True)
                future = Future()
                connector._pending_stage_buffers.append(entry)

                connector._bind_stage_buffers(future)

                self.assertIs(entry.future, future)
                self.assertFalse(entry.reserved)
                self.assertEqual(connector._pending_stage_buffers, [])
        finally:
            if old_trace is None:
                os.environ.pop("UNCHAIN_KV_TRACE", None)
            else:
                os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_codec_buffer_releases_when_host_payload_is_ready(self):
        class Tensor:
            def numel(self):
                return 16

        class Event:
            def query(self):
                return True

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_TRACE": str(Path(tmp) / "trace.jsonl"),
                "UNCHAIN_KV_CODEC": "splitzip_bf16",
                "UNCHAIN_KV_CODEC_GPU_BYTES": "16",
            },
        ):
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")
            codec = vllm_connector._CodecBuffer(Tensor(), reserved=True)
            host = vllm_connector._PinnedStageBuffer(object(), reserved=True)
            payload = vllm_connector._SplitzipCudaPayload(
                memoryview(bytearray([5])),
                [Event()],
                source=object(),
                out=codec.tensor,
                codec_buffer=codec,
            )
            connector._codec_pool["cuda:0"] = [codec]
            connector._pending_codec_buffers.append(codec)
            connector._pending_stage_buffers.append(host)

            self.assertTrue(connector._send_payload_ready(payload))
            send_future = Future()
            connector._bind_stage_buffers(send_future)

            self.assertIsNone(codec.future)
            self.assertIs(host.future, send_future)
            self.assertIsNone(payload.codec_buffer)
            self.assertIs(connector._acquire_codec_buffer(8, "cuda:0", object()), codec)

    def test_direct_codec_payload_releases_gpu_pack_when_d2h_is_ready(self):
        class FakeDType:
            def __str__(self):
                return "torch.bfloat16"

        class FakeTensor:
            dtype = FakeDType()
            device = "cuda:0"

            def __init__(self, size=1 << 20, data=b"top!"):
                self.size = size
                self.data = bytearray(data)

            def numel(self):
                return self.size // 2

            def element_size(self):
                return 2

            def __getitem__(self, _key):
                return self

            def detach(self):
                return self

            def numpy(self):
                return self.data

        source = FakeTensor()
        out = FakeTensor()
        pack = vllm_connector._GpuPackBuffer(source, reserved=True)
        codec = vllm_connector._CodecBuffer(out, reserved=True)
        import unchain_kv.splitzip_cuda as splitzip_cuda

        with patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_TRACE_ENABLED": "0",
                "UNCHAIN_KV_CODEC": "splitzip_bf16",
                "UNCHAIN_KV_CODEC_GPU_SLOTS": "1",
                "UNCHAIN_KV_SPLITZIP_TOP16": "1",
            },
            clear=True,
        ), patch.dict(sys.modules, {"torch": SimpleNamespace()}), patch.object(
            splitzip_cuda, "encode_top16", return_value=4
        ):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
            connector._native_tensor_for_codec = lambda *_args: source
            connector._wait_for_codec_buffer = lambda *_args: codec
            connector._stage_cpu = lambda tensor: tensor
            connector._pending_gpu_pack_buffers.append(pack)

            encoded = connector._splitzip_bf16_native_blocks(
                object(), [0, 2], layer_index=7
            )
            payload = encoded[0]

            self.assertEqual(payload.gpu_pack_buffers, [pack])
            self.assertEqual(connector._pending_gpu_pack_buffers, [])
            self.assertTrue(connector._send_payload_ready(payload))

        self.assertFalse(pack.reserved)
        self.assertEqual(payload.gpu_pack_buffers, [])
        self.assertIsNone(payload.source)

    def test_raw_fragmented_gpu_pack_releases_on_d2h_not_send_future(self):
        class FakeTensor:
            dtype = "u8"
            device = "cuda:0"

            def __init__(self, data=b"raw!"):
                self.data = bytearray(data)

            def detach(self):
                return self

            def numpy(self):
                return self.data

        class FakeKv:
            shape = (2, 4, 1)
            dtype = "bf16"
            device = "cuda:0"

            def index_select(self):
                pass

        class FakeEvent:
            def query(self):
                return False

        event = FakeEvent()
        pack = vllm_connector._GpuPackBuffer(FakeTensor(), reserved=True)
        fake_torch = SimpleNamespace(
            long=object(),
            as_tensor=lambda *_args, **_kwargs: object(),
            index_select=lambda *_args, **_kwargs: None,
        )
        with patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_TRACE_ENABLED": "0",
                "UNCHAIN_KV_GPU_PACK_LAYERS": "1",
            },
            clear=True,
        ):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )

        def acquire(*_args):
            connector._pending_gpu_pack_buffers.append(pack)
            return pack

        connector._acquire_gpu_pack_buffer = acquire
        connector._stage_cpu = lambda _tensor: vllm_connector._StagedTensor(
            FakeTensor(), [event]
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            data, block_size = connector._gpu_pack_native_blocks_bytes(
                FakeKv(), [0, 2]
            )

        self.assertEqual(data.view.tobytes(), b"raw!")
        self.assertEqual(block_size, 2)
        self.assertFalse(pack.reserved)
        self.assertIs(pack.event, event)
        self.assertIsNone(pack.future)
        self.assertEqual(connector._pending_gpu_pack_buffers, [])
        send_future = Future()
        connector._bind_stage_buffers(send_future)
        self.assertIsNone(pack.future)

    def test_strict_gpu_pack_rejects_capacity_bypass(self):
        class FakeKv:
            shape = (2, 4, 1)
            dtype = "bf16"
            device = "cuda:0"

            def index_select(self):
                pass

        with patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_TRACE_ENABLED": "0",
                "UNCHAIN_KV_GPU_PACK_LAYERS": "1",
                "UNCHAIN_KV_GPU_PACK_STRICT": "1",
            },
            clear=True,
        ):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        connector._acquire_gpu_pack_buffer = lambda *_args: None
        with patch.dict(sys.modules, {"torch": SimpleNamespace()}):
            with self.assertRaisesRegex(RuntimeError, "gpu pack capacity exceeded"):
                connector._gpu_pack_native_blocks_bytes(FakeKv(), [0, 2])

    def test_strict_gpu_pack_skips_early_stage_when_credit_is_busy(self):
        class FakeKv:
            shape = (2, 4, 1)
            dtype = "bf16"
            device = "cuda:0"

            def index_select(self):
                pass

        with patch.dict(
            os.environ,
            {
                "UNCHAIN_KV_TRACE_ENABLED": "0",
                "UNCHAIN_KV_GPU_PACK_LAYERS": "1",
                "UNCHAIN_KV_GPU_PACK_STRICT": "1",
            },
            clear=True,
        ):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        rows = []
        connector.trace = SimpleNamespace(
            event=lambda name, **fields: rows.append((name, fields))
        )
        connector._early_stage_active = True
        connector._wait_for_gpu_pack_buffer = lambda *_args: None
        with patch.dict(sys.modules, {"torch": SimpleNamespace()}):
            with self.assertRaises(vllm_connector._EarlyStageSkip):
                connector._gpu_pack_native_blocks_bytes(FakeKv(), [0, 2])
        self.assertFalse(any(name == "gpu_pack_full" for name, _ in rows))

    def test_send_submit_failure_releases_splitzip_gpu_credits(self):
        class ReadyEvent:
            def query(self):
                return True

        class FailingExecutor:
            def submit(self, *_args):
                raise RuntimeError("submit failed")

        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        connector.send_executor = FailingExecutor()
        codec = vllm_connector._CodecBuffer(object(), reserved=True)
        pack = vllm_connector._GpuPackBuffer(object(), reserved=True)
        payload = vllm_connector._SplitzipCudaPayload(
            memoryview(bytearray([255])),
            [ReadyEvent()],
            source=object(),
            out=object(),
            codec_buffer=codec,
            gpu_pack_buffers=[pack],
            deferred_writeback=True,
        )

        with self.assertRaisesRegex(RuntimeError, "submit failed"):
            connector._submit_send(lambda _payload: None, payload)

        self.assertFalse(codec.reserved)
        self.assertFalse(pack.reserved)
        self.assertIsNone(payload.codec_buffer)
        self.assertEqual(payload.gpu_pack_buffers, [])
        self.assertFalse(payload.deferred_writeback)

    def test_send_credit_failure_releases_unsubmitted_splitzip_payload(self):
        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        codec = vllm_connector._CodecBuffer(object(), reserved=True)
        pack = vllm_connector._GpuPackBuffer(object(), reserved=True)
        payload = vllm_connector._SplitzipCudaPayload(
            memoryview(b"top!"),
            [],
            codec_buffer=codec,
            gpu_pack_buffers=[pack],
            deferred_writeback=True,
        )
        connector._wait_send_credit = lambda: (_ for _ in ()).throw(
            RuntimeError("old send failed")
        )

        with self.assertRaisesRegex(RuntimeError, "old send failed"):
            connector._submit_send(lambda _payload: None, payload)

        self.assertFalse(codec.reserved)
        self.assertFalse(pack.reserved)
        self.assertIsNone(payload.codec_buffer)
        self.assertEqual(payload.gpu_pack_buffers, [])
        self.assertFalse(payload.deferred_writeback)

    def test_blocking_send_drain_cleans_all_failed_futures_before_raising(self):
        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        packs = [
            vllm_connector._GpuPackBuffer(object(), reserved=True),
            vllm_connector._GpuPackBuffer(object(), reserved=True),
        ]
        futures = []
        for index, pack in enumerate(packs):
            payload = vllm_connector._SplitzipCudaPayload(
                memoryview(b"top!"), [], gpu_pack_buffers=[pack]
            )
            future = Future()
            future._kvx_send_args = (payload,)
            future.set_exception(RuntimeError(f"failed-{index}"))
            futures.append(future)
        connector.send_futures = futures

        with self.assertRaisesRegex(RuntimeError, "failed-0"):
            connector._check_send_futures(done_only=False)

        self.assertEqual(connector.send_futures, [])
        self.assertTrue(all(not pack.reserved for pack in packs))

    def test_save_finishes_deferred_writeback_before_send_failure(self):
        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        connector.metadata = PipelineMetadata([Session("tx", "req", [1])])
        calls = []
        connector._finish_deferred_codec_writebacks = lambda *_args: calls.append(
            "writeback"
        )

        def fail_send_futures(*_args, **_kwargs):
            calls.append("send_failure")
            raise RuntimeError("network failed")

        connector._check_send_futures = fail_send_futures

        with self.assertRaisesRegex(RuntimeError, "network failed"):
            connector.save_kv_layer(
                "model.layers.2.self_attn", object(), None
            )

        self.assertEqual(calls, ["writeback", "send_failure"])

    def test_clear_metadata_cleans_ready_deferred_gpu_credits(self):
        class ReadyEvent:
            def query(self):
                return True

        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        codec = vllm_connector._CodecBuffer(object(), reserved=True)
        pack = vllm_connector._GpuPackBuffer(object(), reserved=True)
        payload = vllm_connector._SplitzipCudaPayload(
            memoryview(bytearray([5])),
            [ReadyEvent()],
            layer_index=2,
            codec_buffer=codec,
            gpu_pack_buffers=[pack],
            deferred_writeback=True,
        )
        connector.metadata = PipelineMetadata([Session("tx", "req", [1])])
        connector._deferred_codec_writebacks[("tx", 2)] = ([1], payload)

        connector.clear_connector_metadata()

        self.assertIsNone(connector.metadata)
        self.assertEqual(connector._deferred_codec_writebacks, {})
        self.assertFalse(codec.reserved)
        self.assertFalse(pack.reserved)

    def test_wait_for_save_reports_writeback_resource_state(self):
        with patch.dict(os.environ, {"UNCHAIN_KV_TRACE_ENABLED": "0"}, clear=True):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        rows = []
        connector.trace = SimpleNamespace(
            event=lambda name, **fields: rows.append((name, fields))
        )
        connector.codec_writeback_requested = True
        connector._codec_pool["cuda:0"] = [
            vllm_connector._CodecBuffer(object(), reserved=False)
        ]
        connector._gpu_pack_pool[((), "u8", "cuda:0")] = [
            vllm_connector._GpuPackBuffer(object(), reserved=False)
        ]

        connector.wait_for_save()

        self.assertEqual(
            rows[-1],
            (
                "writeback_resource_state",
                {
                    "deferred": 0,
                    "codec_reserved": 0,
                    "pack_reserved": 0,
                    "pending_codec": 0,
                    "pending_pack": 0,
                    "spool_reserved": 0,
                    "spool_resident": 0,
                    "spool_requests": 0,
                },
            ),
        )

    def test_request_spool_wait_for_save_does_not_wait_for_network(self):
        class FakeTensor:
            def numel(self):
                return 7

            def element_size(self):
                return 1

        connector = self._spool_connector()
        connector._spool_layer_block_bytes = [10, 20]
        connector.bind_connector_metadata(
            PipelineMetadata([Session("tx", "req", [1])])
        )
        spool = connector._active_request_spools[0]
        sending = threading.Event()
        unblock = threading.Event()

        def send(*_args):
            sending.set()
            unblock.wait(1)

        connector._send_compressed_native_layer_blocks = send
        pinned = vllm_connector._PinnedStageBuffer(FakeTensor(), reserved=True)
        connector._pinned_stage_pool[((7,), "u8")] = [pinned]
        connector._pending_stage_buffers.append(pinned)
        connector._submit_spooled_layer(
            spool, b"abc", 0, 1, 1, 2, "splitzip_bf16"
        )
        connector._submit_spooled_layer(
            spool, b"defg", 1, 1, 1, 2, "splitzip_bf16"
        )

        connector.wait_for_save()

        self.assertTrue(sending.wait(1))
        self.assertTrue(connector.send_futures)
        self.assertFalse(connector.send_futures[0].done())
        self.assertTrue(connector._buffer_entry_available(pinned))
        self.assertEqual(connector._spool_reserved_bytes, 32)
        unblock.set()
        connector._check_send_futures(done_only=False)
        self.assertEqual(connector._spool_reserved_bytes, 0)
        self.assertEqual(connector._spool_resident_bytes, 0)
        self.assertEqual(connector._request_spools, {})
        connector.spool_executor.shutdown(wait=True)
        connector.send_executor.shutdown(wait=True)

    def test_spool_capable_followup_save_only_polls_pending_sender(self):
        connector = object.__new__(UnchainKVConnector)
        connector.request_spool_capable = True
        connector._active_request_spools = []
        connector._cancel_deferred_codec_writebacks = lambda **_kwargs: None
        connector._flush_layer_groups = lambda: None
        connector._flush_prefill_forward_events = lambda: None
        connector._trace_writeback_resource_state = lambda: None
        checks = []
        connector._check_send_futures = lambda done_only: checks.append(done_only)

        connector.wait_for_save()

        self.assertEqual(checks, [True])
        connector._spool_short_fast_path = True
        connector.wait_for_save()
        self.assertEqual(checks, [True, False])

    def test_request_spool_preserves_request_fifo_for_interleaved_layers(self):
        connector = self._spool_connector()
        connector._spool_layer_block_bytes = [10, 20]
        connector.bind_connector_metadata(
            PipelineMetadata(
                [Session("tx-a", "a", [1]), Session("tx-b", "b", [2])]
            )
        )
        spools = connector._active_request_spools
        sent = []
        connector._send_compressed_native_layer_blocks = (
            lambda _tx, request, layer, *_args: sent.append((request, layer))
        )
        for spool, layer, data in (
            (spools[0], 0, b"a0"),
            (spools[1], 0, b"b0"),
            (spools[0], 1, b"a1"),
            (spools[1], 1, b"b1"),
        ):
            connector._submit_spooled_layer(
                spool, data, layer, 1, 1, 2, "splitzip_bf16"
            )

        connector.wait_for_save()
        connector._check_send_futures(done_only=False)

        self.assertEqual(sent, [("a", 0), ("a", 1), ("b", 0), ("b", 1)])
        self.assertEqual(connector._spool_reserved_bytes, 0)
        connector.spool_executor.shutdown(wait=True)
        connector.send_executor.shutdown(wait=True)

    def test_request_spool_admission_is_batch_atomic(self):
        connector = self._spool_connector(cap=63)
        connector._spool_layer_block_bytes = [10, 20]

        with self.assertRaisesRegex(RuntimeError, "batch exceeds cap"):
            connector.bind_connector_metadata(
                PipelineMetadata(
                    [Session("tx-a", "a", [1]), Session("tx-b", "b", [2])]
                )
            )

        self.assertEqual(connector._spool_reserved_bytes, 0)
        self.assertEqual(connector._request_spools, {})
        connector.spool_executor.shutdown(wait=True)
        connector.send_executor.shutdown(wait=True)

    def test_request_spool_reservation_uses_prefix_suppressed_suffix(self):
        connector = self._spool_connector()
        connector._spool_layer_block_bytes = [10, 20]
        session = Session("tx", "req", [1, 2, 3])

        connector._apply_prefix_transfer_plan(session, 16, "producer")

        self.assertEqual(session.block_ids, [2, 3])
        self.assertEqual(connector._request_spool_bound(session), 62)
        connector.spool_executor.shutdown(wait=True)
        connector.send_executor.shutdown(wait=True)

    def test_request_spool_stage_failure_releases_all_resources(self):
        connector = self._spool_connector()
        connector._spool_layer_block_bytes = [10, 20]
        connector.bind_connector_metadata(
            PipelineMetadata([Session("tx", "req", [1])])
        )
        spool = connector._active_request_spools[0]
        connector._ready_payload = lambda _data: (_ for _ in ()).throw(
            RuntimeError("stage failed")
        )
        connector._submit_spooled_layer(
            spool, b"bad", 0, 1, 1, 2, "splitzip_bf16"
        )

        with self.assertRaisesRegex(RuntimeError, "stage failed"):
            connector.wait_for_save()
        try:
            connector._check_send_futures(done_only=False)
        except RuntimeError as exc:
            self.assertIn("stage failed", str(exc))

        self.assertEqual(connector._spool_reserved_bytes, 0)
        self.assertEqual(connector._spool_resident_bytes, 0)
        self.assertEqual(connector._request_spools, {})
        connector.spool_executor.shutdown(wait=True)
        connector.send_executor.shutdown(wait=True)

    def test_request_spool_send_failure_releases_pageable_bytes(self):
        connector = self._spool_connector()
        connector._spool_layer_block_bytes = [10, 20]
        connector.bind_connector_metadata(
            PipelineMetadata([Session("tx", "req", [1])])
        )
        spool = connector._active_request_spools[0]
        connector._send_compressed_native_layer_blocks = lambda *_args: (
            _ for _ in ()
        ).throw(RuntimeError("send failed"))
        connector._submit_spooled_layer(
            spool, b"payload", 0, 1, 1, 2, "splitzip_bf16"
        )

        error = None
        try:
            connector.wait_for_save()
        except RuntimeError as exc:
            error = exc
        try:
            connector._check_send_futures(done_only=False)
        except RuntimeError as exc:
            error = error or exc

        self.assertIsNotNone(error)
        self.assertIn("send failed", str(error))
        self.assertEqual(connector._spool_reserved_bytes, 0)
        self.assertEqual(connector._spool_resident_bytes, 0)
        self.assertEqual(connector._request_spools, {})
        connector.spool_executor.shutdown(wait=True)
        connector.send_executor.shutdown(wait=True)

    def test_request_spool_metadata_abort_unblocks_sender(self):
        connector = self._spool_connector()
        connector._spool_layer_block_bytes = [10, 20]
        connector.bind_connector_metadata(
            PipelineMetadata([Session("tx", "req", [1])])
        )

        connector.clear_connector_metadata()
        with self.assertRaisesRegex(RuntimeError, "metadata cleared"):
            connector._check_send_futures(done_only=False)

        self.assertEqual(connector._spool_reserved_bytes, 0)
        self.assertEqual(connector._request_spools, {})
        connector.spool_executor.shutdown(wait=True)
        connector.send_executor.shutdown(wait=True)

    def test_ready_payload_waits_for_staged_events(self):
        class Event:
            def __init__(self):
                self.waited = False

            def synchronize(self):
                self.waited = True

        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                )
                connector = UnchainKVConnector(config, "worker")
                event = Event()
                view = memoryview(bytearray(b"abc"))
                payload = vllm_connector._StagedPayload(view, [event])

                self.assertIs(connector._ready_payload(payload), view)
                self.assertTrue(event.waited)
        finally:
            if old_trace is None:
                os.environ.pop("UNCHAIN_KV_TRACE", None)
            else:
                os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_block_bytes_reads_standard_kv_page_dimension(self):
        class FakeBlock:
            dtype = "int16"

            def detach(self):
                return self

            def contiguous(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self

            def tobytes(self):
                return b"page"

        class FakeKVLayer:
            shape = (2, 3, 2, 1)

            def __init__(self):
                self.key = None

            def dim(self):
                return len(self.shape)

            def __getitem__(self, key):
                self.key = key
                return FakeBlock()

        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")
            kv_layer = FakeKVLayer()

            self.assertEqual(connector._block_bytes(kv_layer, 1), b"page")
            self.assertIsInstance(kv_layer.key, tuple)
            self.assertIsInstance(kv_layer.key[0], slice)
            self.assertEqual(kv_layer.key[1], 1)
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_blocks_bytes_single_block_uses_direct_block_view_without_torch(self):
        class FakeBlock:
            dtype = "int16"

            def __init__(self, data):
                self.data = bytearray(data)

            def detach(self):
                return self

            def contiguous(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.data

        class FakeKVLayer:
            shape = (2, 3, 2)
            device = "cuda:0"

            def __init__(self):
                self.keys = []

            def dim(self):
                return len(self.shape)

            def __getitem__(self, key):
                self.keys.append(key)
                return FakeBlock(b"kv")

            def index_select(self, *_args):
                raise AssertionError("single block should not gather")

        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")
            kv_layer = FakeKVLayer()

            data, block_size = connector._blocks_bytes(kv_layer, [1])

            self.assertEqual(bytes(data), b"kv")
            self.assertEqual(block_size, 2)
            self.assertEqual(kv_layer.keys, [(slice(None, None, None), 1)])
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_blocks_bytes_bfloat16_contiguous_span_imports_torch(self):
        class FakeBlock:
            dtype = "torch.bfloat16"

            def __init__(self):
                self.data = bytearray(b"abcd")
                self.view_dtype = None

            def detach(self):
                return self

            def contiguous(self):
                return self

            def cpu(self):
                return self

            def view(self, dtype):
                self.view_dtype = dtype
                return self

            def numpy(self):
                return self.data

        class FakeSlice:
            def __init__(self, block):
                self.block = block

            def transpose(self, *_args):
                return self.block

        class FakeKVLayer:
            shape = (2, 3, 2)
            device = "cuda:0"

            def __init__(self):
                self.block = FakeBlock()

            def dim(self):
                return len(self.shape)

            def __getitem__(self, _key):
                return FakeSlice(self.block)

            def index_select(self, *_args):
                raise AssertionError("contiguous span should not gather")

        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with patch.dict("sys.modules", {"torch": SimpleNamespace(uint16="uint16")}):
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                )
                connector = UnchainKVConnector(config, "worker")
                kv_layer = FakeKVLayer()

                data, block_size = connector._blocks_bytes(kv_layer, [1, 2])

                self.assertEqual(bytes(data), b"abcd")
                self.assertEqual(block_size, 2)
                self.assertEqual(kv_layer.block.view_dtype, "uint16")
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_blocks_bytes_returns_memoryview_for_torch_layer(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")
            kv_layer = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)

            data, block_size = connector._blocks_bytes(kv_layer, [1, 2])

            expected = kv_layer[:, 1:3].transpose(0, 1).contiguous().numpy().tobytes()
            self.assertIsInstance(data, memoryview)
            self.assertEqual(block_size, len(expected) // 2)
            self.assertEqual(bytes(data), expected)
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_kv_major_blocks_bytes_returns_separate_key_value_views(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")
            kv_layer = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)

            key_data, value_data, part_size = connector._kv_major_blocks_bytes(
                kv_layer, [1, 2]
            )

            expected_key = kv_layer[0, 1:3].contiguous().numpy().tobytes()
            expected_value = kv_layer[1, 1:3].contiguous().numpy().tobytes()
            self.assertIsInstance(key_data, memoryview)
            self.assertIsInstance(value_data, memoryview)
            self.assertEqual(part_size, len(expected_key) // 2)
            self.assertEqual(bytes(key_data), expected_key)
            self.assertEqual(bytes(value_data), expected_value)
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_kv_major_blocks_bytes_skips_non_contiguous_blocks_without_torch(self):
        class FakeKVPart:
            def index_select(self, *_args):
                raise AssertionError("kv-major fallback should avoid gather")

        class FakeKVLayer:
            shape = (2, 3, 2)
            device = "cuda:0"

            def dim(self):
                return len(self.shape)

            def __getitem__(self, _key):
                return FakeKVPart()

            def index_select(self, *_args):
                raise AssertionError("kv-major fallback should avoid gather")

        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")

            self.assertIsNone(connector._kv_major_blocks_bytes(FakeKVLayer(), [0, 2]))
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_native_blocks_bytes_returns_kv_first_view_for_contiguous_blocks(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")
            kv_layer = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)

            native = connector._native_blocks_bytes(kv_layer, [1, 2])

            self.assertIsNotNone(native)
            data, block_size = native
            expected = kv_layer[:, 1:3].contiguous().numpy().tobytes()
            self.assertIsInstance(data, memoryview)
            self.assertEqual(block_size, len(expected) // 2)
            self.assertEqual(bytes(data), expected)
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_native_blocks_bytes_skips_non_contiguous_blocks(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")
            kv_layer = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)

            self.assertIsNone(connector._native_blocks_bytes(kv_layer, [0, 2]))
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_native_blocks_bytes_skips_non_contiguous_blocks_without_torch(self):
        class FakeKVLayer:
            shape = (2, 3, 2)
            device = "cuda:0"

            def dim(self):
                return len(self.shape)

            def index_select(self, *_args):
                raise AssertionError("native fallback should avoid gather")

        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
            )
            connector = UnchainKVConnector(config, "worker")

            self.assertIsNone(connector._native_blocks_bytes(FakeKVLayer(), [0, 2]))
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_restore_layer_writes_registered_kv_cache(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector.sessions["tx"] = Session("tx", "req", [1])
            kv_cache = torch.zeros((2, 3, 2), dtype=torch.float32)
            connector.register_kv_caches({"model.layers.0.self_attn": kv_cache})
            source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
            payload = source.numpy().tobytes()
            connector.store.add(
                Chunk(
                    ChunkHeader(
                        "tx",
                        "req",
                        0,
                        0,
                        0,
                        1,
                        0,
                        len(payload),
                        crc32(payload) & 0xFFFFFFFF,
                    ),
                    payload,
                )
            )

            connector._restore_layer("model.layers.0.self_attn", "tx")

            self.assertTrue(torch.equal(kv_cache[:, 1], source))
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_restore_layer_accepts_native_layout_payload(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_max_blocks = os.environ.get("UNCHAIN_KV_MAX_BLOCKS")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_MAX_BLOCKS"] = "0"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector.sessions["tx"] = Session("tx", "req", [1, 2])
            kv_cache = torch.zeros((2, 4, 2), dtype=torch.float32)
            connector.register_kv_caches({"model.layers.0.self_attn": kv_cache})
            source = torch.tensor(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                dtype=torch.float32,
            )
            payload = source.numpy().tobytes()
            connector.store.add(
                Chunk(
                    ChunkHeader(
                        "tx",
                        "req",
                        0,
                        0,
                        0,
                        1,
                        0,
                        len(payload),
                        crc32(payload) & 0xFFFFFFFF,
                    ),
                    payload,
                ),
                layout="native",
            )

            connector._restore_layer("model.layers.0.self_attn", "tx")

            self.assertTrue(torch.equal(kv_cache[:, 1:3], source))
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_max_blocks is None:
            os.environ.pop("UNCHAIN_KV_MAX_BLOCKS", None)
        else:
            os.environ["UNCHAIN_KV_MAX_BLOCKS"] = old_max_blocks

    def test_restore_payload_keeps_single_memoryview(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            view = memoryview(bytearray(b"payload"))

            self.assertIs(connector._restore_payload([view]), view)
            self.assertEqual(connector._restore_payload([b"a", b"b"]), b"ab")
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_copy_blocks_bytes_reads_memoryview_without_bytearray(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            kv_cache = torch.zeros((2, 3, 2), dtype=torch.float32)
            source = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
            payload = memoryview(bytearray(source.numpy().tobytes()))

            with patch(
                "builtins.bytearray",
                side_effect=AssertionError("restore copied payload through bytearray"),
            ):
                connector._copy_blocks_bytes(kv_cache, [1], payload)

            self.assertTrue(torch.equal(kv_cache[:, 1], source[0]))
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace

    def test_restore_layer_accepts_kv_major_payloads(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_max_blocks = os.environ.get("UNCHAIN_KV_MAX_BLOCKS")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_MAX_BLOCKS"] = "0"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector.sessions["tx"] = Session("tx", "req", [1, 2])
            kv_cache = torch.zeros((2, 4, 2), dtype=torch.float32)
            connector.register_kv_caches({"model.layers.0.self_attn": kv_cache})
            key = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
            value = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
            key_payload = key.numpy().tobytes()
            value_payload = value.numpy().tobytes()
            for index, payload in enumerate([key_payload, value_payload]):
                connector.store.add(
                    Chunk(
                        ChunkHeader(
                            "tx",
                            "req",
                            0,
                            index,
                            index,
                            2,
                            0,
                            len(payload),
                            crc32(payload) & 0xFFFFFFFF,
                        ),
                        payload,
                    ),
                    layout="kv_major",
                )

            connector._restore_layer("model.layers.0.self_attn", "tx")

            self.assertTrue(torch.equal(kv_cache[0, 1:3], key))
            self.assertTrue(torch.equal(kv_cache[1, 1:3], value))
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_max_blocks is None:
            os.environ.pop("UNCHAIN_KV_MAX_BLOCKS", None)
        else:
            os.environ["UNCHAIN_KV_MAX_BLOCKS"] = old_max_blocks


    def test_restore_layer_rejects_short_payload_and_releases_transfer(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_max_blocks = os.environ.get("UNCHAIN_KV_MAX_BLOCKS")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_MAX_BLOCKS"] = "8"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector.sessions["tx"] = Session("tx", "req", [1, 2])
            kv_cache = torch.zeros((2, 4, 2), dtype=torch.float32)
            connector.register_kv_caches({"model.layers.0.self_attn": kv_cache})
            source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
            payload = source.numpy().tobytes()
            connector.store.add(
                Chunk(
                    ChunkHeader(
                        "tx",
                        "req",
                        0,
                        0,
                        0,
                        1,
                        0,
                        len(payload),
                        crc32(payload) & 0xFFFFFFFF,
                    ),
                    payload,
                )
            )

            with self.assertRaisesRegex(RuntimeError, "restore incomplete"):
                connector._restore_layer("model.layers.0.self_attn", "tx")

            self.assertNotIn(("tx", 0), connector._restored_layers)
            self.assertEqual(connector.store.payload_bytes("tx"), 0)
            with self.assertRaisesRegex(RuntimeError, "restore incomplete"):
                connector.store.wait("tx", 0, 0)
        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_max_blocks is None:
            os.environ.pop("UNCHAIN_KV_MAX_BLOCKS", None)
        else:
            os.environ["UNCHAIN_KV_MAX_BLOCKS"] = old_max_blocks

    def test_bulk_decode_waits_for_all_layers_before_admission(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_layers = os.environ.get("UNCHAIN_KV_EXPECTED_LAYERS")
        old_bulk = os.environ.get("UNCHAIN_KV_BULK_DECODE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = "2"
            os.environ["UNCHAIN_KV_BULK_DECODE"] = "1"
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(kv_role="kv_consumer")
            )
            connector = UnchainKVConnector(config, "worker")
            connector._active_sessions["tx"] = Session("tx", "req", [])
            connector._active_transfer_id = "tx"
            connector._active_request_id = "req"

            payload = b"x"
            connector.store.add(
                Chunk(
                    ChunkHeader(
                        "tx",
                        "req",
                        0,
                        0,
                        0,
                        1,
                        0,
                        len(payload),
                        crc32(payload) & 0xFFFFFFFF,
                    ),
                    payload,
                )
            )
            self.assertEqual(connector.get_finished(), (None, None))

            connector.store.add(
                Chunk(
                    ChunkHeader(
                        "tx",
                        "req",
                        1,
                        0,
                        0,
                        1,
                        0,
                        len(payload),
                        crc32(payload) & 0xFFFFFFFF,
                    ),
                    payload,
                )
            )
            self.assertEqual(connector.get_finished(), (set(), {"req"}))

        if old_trace is None:
            os.environ.pop("UNCHAIN_KV_TRACE", None)
        else:
            os.environ["UNCHAIN_KV_TRACE"] = old_trace
        if old_layers is None:
            os.environ.pop("UNCHAIN_KV_EXPECTED_LAYERS", None)
        else:
            os.environ["UNCHAIN_KV_EXPECTED_LAYERS"] = old_layers
        if old_bulk is None:
            os.environ.pop("UNCHAIN_KV_BULK_DECODE", None)
        else:
            os.environ["UNCHAIN_KV_BULK_DECODE"] = old_bulk



    def test_producer_traces_request_block_layout(self):
        """Producer worker emits producer_block_layout event per request."""
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_extent = os.environ.get("UNCHAIN_KV_EXTENT_ALLOC")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_EXTENT_ALLOC"] = "normalize"
            try:
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
                    cache_config=SimpleNamespace(enable_prefix_caching=False),
                    scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
                )
                connector = UnchainKVConnector(config, "worker")
                metadata = PipelineMetadata([
                    Session("tx", "req", [1, 2, 8, 9, 10]),
                ])
                connector.bind_connector_metadata(metadata)
                trace_path = Path(tmp) / "trace.jsonl"
                self.assertTrue(trace_path.exists(), "trace file should exist")
                lines = [
                    json.loads(line)
                    for line in trace_path.read_text().strip().split("\n")
                    if line.strip()
                ]
                layouts = [
                    e for e in lines if e.get("event") == "producer_block_layout"
                ]
                self.assertEqual(len(layouts), 1)
                layout = layouts[0]
                self.assertEqual(layout["transfer"], "tx")
                self.assertEqual(layout["request"], "req")
                self.assertEqual(layout["blocks"], 5)
                self.assertEqual(layout["runs"], 2)
                self.assertEqual(layout["longest_run"], 3)
                self.assertFalse(layout["contiguous"])
                self.assertEqual(layout["extent_alloc"], "normalize")
            finally:
                if old_trace is None:
                    os.environ.pop("UNCHAIN_KV_TRACE", None)
                else:
                    os.environ["UNCHAIN_KV_TRACE"] = old_trace
                if old_extent is None:
                    os.environ.pop("UNCHAIN_KV_EXTENT_ALLOC", None)
                else:
                    os.environ["UNCHAIN_KV_EXTENT_ALLOC"] = old_extent

    def test_producer_block_layout_handles_empty_blocks(self):
        old_trace = os.environ.get("UNCHAIN_KV_TRACE")
        old_extent = os.environ.get("UNCHAIN_KV_EXTENT_ALLOC")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["UNCHAIN_KV_TRACE"] = str(Path(tmp) / "trace.jsonl")
            os.environ["UNCHAIN_KV_EXTENT_ALLOC"] = "normalize"
            try:
                config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
                    cache_config=SimpleNamespace(enable_prefix_caching=False),
                    scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
                )
                connector = UnchainKVConnector(config, "worker")
                metadata = PipelineMetadata([
                    Session("tx", "req", []),
                ])
                connector.bind_connector_metadata(metadata)
                trace_path = Path(tmp) / "trace.jsonl"
                lines = [
                    json.loads(line)
                    for line in trace_path.read_text().strip().split("\n")
                    if line.strip()
                ]
                layouts = [
                    e for e in lines if e.get("event") == "producer_block_layout"
                ]
                self.assertEqual(len(layouts), 1)
                layout = layouts[0]
                self.assertEqual(layout["blocks"], 0)
                self.assertEqual(layout["runs"], 0)
                self.assertEqual(layout["longest_run"], 0)
            finally:
                if old_trace is None:
                    os.environ.pop("UNCHAIN_KV_TRACE", None)
                else:
                    os.environ["UNCHAIN_KV_TRACE"] = old_trace
                if old_extent is None:
                    os.environ.pop("UNCHAIN_KV_EXTENT_ALLOC", None)
                else:
                    os.environ["UNCHAIN_KV_EXTENT_ALLOC"] = old_extent

    def test_extent_eligibility_allows_prefix_cache_but_rejects_unsupported_layouts(self):
        connector = object.__new__(UnchainKVConnector)
        config = SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=True)
        )
        full = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=SimpleNamespace(attention_type="full")
                )
            ]
        )
        connector._check_extent_eligibility(full, config)

        with self.assertRaises(RuntimeError):
            connector._check_extent_eligibility(
                SimpleNamespace(kv_cache_groups=[full, full]), config
            )
        with self.assertRaises(RuntimeError):
            connector._check_extent_eligibility(
                SimpleNamespace(
                    kv_cache_groups=[
                        SimpleNamespace(
                            kv_cache_spec=SimpleNamespace(
                                attention_type="sliding"
                            )
                        )
                    ]
                ),
                config,
            )

    def test_extent_prefer_mode_is_accepted(self):
        with patch.dict(
            os.environ,
            {"UNCHAIN_KV_TRACE_ENABLED": "0", "UNCHAIN_KV_EXTENT_ALLOC": "prefer"},
            clear=True,
        ):
            connector = UnchainKVConnector(
                SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
                "worker",
            )
        self.assertEqual(connector.extent_alloc_mode, "prefer")

if __name__ == "__main__":
    unittest.main()
