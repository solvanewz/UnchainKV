from pathlib import Path
import unittest


class RunOvsLimitedMatrixTest(unittest.TestCase):
    def test_passes_adaptive_spool_limits_into_containers(self):
        script = Path("scripts/run_ovs_limited_matrix.sh").read_text()

        self.assertIn("UNCHAIN_KV_REQUEST_SPOOL_AUTO=", script)
        self.assertIn("UNCHAIN_KV_HOST_GUARD_BYTES=", script)
        self.assertIn("UNCHAIN_KV_GPU_GUARD_BYTES=", script)

    def test_derives_expected_layers_from_the_selected_model(self):
        script = Path("scripts/run_ovs_limited_matrix.sh").read_text()

        self.assertIn('["num_hidden_layers"]', script)
        self.assertIn('-e UNCHAIN_KV_EXPECTED_LAYERS="${EXPECTED_LAYERS}"', script)

    def test_passes_native_splitzip_decode_flag_to_containers(self):
        script = Path("scripts/run_ovs_limited_matrix.sh").read_text()
        self.assertIn(
            '-e UNCHAIN_KV_SPLITZIP_NATIVE_DECODE="${UNCHAIN_KV_SPLITZIP_NATIVE_DECODE:-0}"',
            script,
        )
        self.assertIn(
            '-e UNCHAIN_KV_SPLITZIP_TOP16="${UNCHAIN_KV_SPLITZIP_TOP16:-0}"',
            script,
        )
        self.assertIn(
            '-e UNCHAIN_KV_CODEC_WRITEBACK="${UNCHAIN_KV_CODEC_WRITEBACK:-}"',
            script,
        )
        self.assertIn(
            '-e UNCHAIN_KV_CODEC_WRITEBACK_STRICT="${UNCHAIN_KV_CODEC_WRITEBACK_STRICT:-0}"',
            script,
        )
        self.assertIn(
            '-e UNCHAIN_KV_HOST_MIRROR_BYTES="${UNCHAIN_KV_HOST_MIRROR_BYTES:-0}"',
            script,
        )
        self.assertIn(
            '-e UNCHAIN_KV_SPLITZIP_CHUNKS="${UNCHAIN_KV_SPLITZIP_CHUNKS:-1}"',
            script,
        )
        self.assertIn(
            '--decode-slots "${UNCHAIN_KV_DECODE_SLOTS:-0}"',
            script,
        )
        self.assertIn(
            '--metrics "${RUN_ROOT}/proxy-metrics.jsonl"',
            script,
        )
        self.assertIn(
            'UNCHAIN_KV_HOST_MIRROR_LAYERS="${HOST_MIRROR_LAYERS}"',
            script,
        )
        self.assertIn(
            '-e UNCHAIN_KV_WAIT_TIMEOUT_S="${UNCHAIN_KV_WAIT_TIMEOUT_S:-300}"',
            script,
        )
        self.assertIn(
            '-e UNCHAIN_KV_GPU_PACK_STRICT="${UNCHAIN_KV_GPU_PACK_STRICT:-0}"',
            script,
        )
        self.assertIn(
            '-e UNCHAIN_KV_GPU_PACK_BYTES="${UNCHAIN_KV_GPU_PACK_BYTES:-0}"',
            script,
        )
        self.assertIn(
            '-e UNCHAIN_KV_PAYLOAD_READY="${UNCHAIN_KV_PAYLOAD_READY:-1}"',
            script,
        )
        self.assertIn("UNCHAIN_KV_EXTENT_RESERVE_BLOCKS=", script)
        self.assertIn('warmup = int(sys.argv[7])', script)
        self.assertIn('warmup_per_length = int(sys.argv[8])', script)
        self.assertIn('warmup-by-length.jsonl', script)
        self.assertIn('-e PYTHONSAFEPATH=1', script)

    def test_mooncake_matrix_supports_the_same_sample_and_warmup_controls(self):
        script = Path("scripts/run_ovs_limited_mooncake_matrix.sh").read_text()
        self.assertIn('SAMPLES="${UNCHAIN_KV_SAMPLES:-10}"', script)
        self.assertIn('SAMPLE_OFFSET="${UNCHAIN_KV_SAMPLE_OFFSET:-0}"', script)
        self.assertIn(
            'WARMUP_PER_LENGTH="${UNCHAIN_KV_WARMUP_PER_LENGTH:-0}"', script
        )
        self.assertIn('for index in range(samples):', script)
        self.assertIn('sample_index = sample_offset + index', script)

    def test_mooncake_containers_reuse_ephemeral_ports(self):
        script = Path("scripts/run_ovs_limited_mooncake_matrix.sh").read_text()
        self.assertIn('--sysctl net.ipv4.ip_local_port_range="1024 65535"', script)
        self.assertIn("--sysctl net.ipv4.tcp_tw_reuse=1", script)

    def test_sharegpt_runner_supports_open_loop_random_32k(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()
        self.assertIn('BENCH_DATASET="${UNCHAIN_KV_BENCH_DATASET:-sharegpt}"', script)
        self.assertIn('REQUEST_RATE="${UNCHAIN_KV_REQUEST_RATE:-inf}"', script)
        self.assertIn('--random-input-len "${RANDOM_INPUT_LEN}"', script)
        self.assertIn('--request-rate "${REQUEST_RATE}"', script)
        self.assertIn('UNCHAIN_KV_CODEC_GPU_BYTES="${CODEC_GPU_BYTES}"', script)
        self.assertIn('UNCHAIN_KV_HOST_MIRROR_BYTES="${HOST_MIRROR_BYTES}"', script)
        self.assertIn('cp "${run_root}/trace/prefill.jsonl"', script)

    def test_sharegpt_runner_decouples_writeback_scheduling_knobs(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()
        self.assertIn('BULK_DECODE="${UNCHAIN_KV_BULK_DECODE:-0}"', script)
        self.assertIn('RESTORE_AHEAD="${UNCHAIN_KV_RESTORE_AHEAD:-0}"', script)
        self.assertIn(
            'HOST_MIRROR_LAYERS="${UNCHAIN_KV_HOST_MIRROR_LAYERS:-32}"', script
        )
        self.assertIn('bulk_decode="${BULK_DECODE}"', script)
        self.assertIn('restore_ahead="${RESTORE_AHEAD}"', script)
        self.assertIn('host_mirror_layers="${HOST_MIRROR_LAYERS}"', script)
        self.assertIn('echo "bulk_decode=${bulk_decode}"', script)
        self.assertIn('echo "restore_ahead=${restore_ahead}"', script)
        self.assertIn('echo "host_mirror_layers=${host_mirror_layers}"', script)
        self.assertIn(
            'SHAREGPT_OUTPUT_LEN="${UNCHAIN_KV_SHAREGPT_OUTPUT_LEN:-}"', script
        )
        self.assertIn('sharegpt_args=(--sharegpt-output-len "${SHAREGPT_OUTPUT_LEN}")', script)



    def test_extent_alloc_passed_to_start_vllm_by_role(self):
        script = Path("scripts/run_ovs_limited_matrix.sh").read_text()
        self.assertIn("extent_alloc", script)
        self.assertIn("kv_consumer", script)
        self.assertIn("kv_producer", script)
        # Consumer must be forced off, producer uses outer mode
        self.assertIn('${EXTENT_ALLOC:-off}', script)

    def test_sharegpt_runner_passes_extent_alloc(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()
        self.assertIn('EXTENT_ALLOC="${UNCHAIN_KV_EXTENT_ALLOC:-off}"', script)
        self.assertIn("UNCHAIN_KV_EXTENT_ALLOC=", script)
        self.assertIn("extent_alloc=", script)
        self.assertIn("UNCHAIN_KV_GPU_PACK_BYTES=", script)
        self.assertIn("gpu_pack_bytes=", script)

    def test_zero_sample_setup_skips_trace_summary(self):
        script = Path("scripts/run_ovs_limited_matrix.sh").read_text()
        self.assertIn(
            "(( SAMPLES > 0 || WARMUP > 0 || WARMUP_PER_LENGTH > 0 ))", script
        )

if __name__ == "__main__":
    unittest.main()
