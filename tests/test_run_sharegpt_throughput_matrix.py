from pathlib import Path
import os
import subprocess
import unittest


class RunSharegptThroughputMatrixTest(unittest.TestCase):
    def test_canonical_profiles_change_only_cumulative_methods(self):
        script = "scripts/run_sharegpt_throughput_matrix.sh"

        def profile(path):
            output = subprocess.run(
                ["bash", script, "profile", path],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            return dict(line.split("=", 1) for line in output.splitlines())

        r1, m1, m12 = profile("R1"), profile("M1"), profile("M12")
        m123, fixed, auto = (
            profile("M123"),
            profile("M1234-F"),
            profile("M1234-A"),
        )
        self.assertEqual(r1["codec"], "")
        self.assertEqual(m1["codec"], "splitzip_bf16")
        self.assertEqual(m1["codec_writeback"], "0")
        self.assertEqual(m12["codec_writeback"], "1")
        self.assertEqual(m12["extent_alloc"], "off")
        self.assertEqual(m123["extent_alloc"], "prefer")
        self.assertNotEqual(fixed["request_spool_bytes"], "0")
        self.assertEqual(fixed["request_spool_auto"], "0")
        self.assertEqual(auto["request_spool_bytes"], "0")
        self.assertEqual(auto["request_spool_auto"], "1")
        for current in (m1, m12, m123, fixed, auto):
            self.assertEqual(current["bulk_decode"], "0")
            self.assertEqual(current["restore_ahead"], "0")
            self.assertEqual(current["host_mirror_layers"], "32")

    def test_adaptive_matrix_has_four_interleaved_cells(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()

        self.assertIn("adaptive)", script)
        self.assertIn("run_order 1 r0 r1 r2 r3", script)
        self.assertIn("run_order 4 r3 r0 r1 r2", script)

    def test_release_normalization_is_shared_and_manifested(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()
        mooncake = Path("scripts/run_ovs_limited_mooncake_matrix.sh").read_text()
        native = Path("scripts/run_ovs_limited_matrix.sh").read_text()
        for text in (script, mooncake, native):
            self.assertIn("UNCHAIN_KV_NORMALIZE_RELEASE", text)
        self.assertIn("python3 -m unchain_kv.patch_vllm", script)
        self.assertIn("python3 -m unchain_kv.patch_vllm", mooncake)
        self.assertIn('echo "normalize_release=', script)

    def test_adaptive_cells_select_off_observe_auto_and_fixed(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()

        self.assertIn("r1) request_spool_auto=observe", script)
        self.assertIn("r2) request_spool_auto=1", script)
        self.assertIn('r3) request_spool_bytes="${FIXED_SPOOL_BYTES}"', script)

    def test_adaptive_cells_force_method_three_and_one_pack(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()

        self.assertIn("extent_alloc=prefer", script)
        self.assertIn("gpu_pack_layers=1", script)
        self.assertIn("gpu_pack_bytes=67108864", script)
        self.assertIn("gpu_pack_strict=1", script)

    def test_canonical_pack_cap_can_follow_model_geometry(self):
        result = subprocess.run(
            ["bash", "scripts/run_sharegpt_throughput_matrix.sh", "profile", "M1234-A"],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "UNCHAIN_KV_CANONICAL_GPU_PACK_BYTES": "134217728"},
        )

        self.assertIn("gpu_pack_bytes=134217728", result.stdout)

    def test_failed_benchmark_is_not_resumed_or_reported_as_success(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()

        self.assertIn("bench_complete()", script)
        self.assertIn('bench_complete "${run_root}/bench.json" "${PROMPTS}"', script)
        self.assertIn("benchmark incomplete:", script)
        self.assertIn('timeout --signal=TERM --kill-after=30', script)

    def test_formal_cell_has_fail_closed_preflight_before_launch(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()

        self.assertIn("write_static_preflight()", script)
        self.assertLess(
            script.index('write_static_preflight "${run_root}/preflight-static.txt"'),
            script.index('case "${cell}" in', script.index("run_cell()")),
        )
        self.assertIn(
            'cp "${run_root}/preflight.txt" "${phase_root}/preflight.txt"', script
        )
        self.assertIn('grep -q " mtu ${MTU} "', script)
        self.assertIn('grep -qi "rate ${RATE}"', script)
        self.assertIn("disk_available_kib=", script)
        self.assertIn("grep -c 'NVIDIA A40'", script)
        self.assertIn('grep -q "via ${GATEWAY_IP} "', script)
        self.assertIn('if ! write_preflight "${run_root}/preflight.txt"', script)

    def test_gpu_containers_are_checked_before_vllm_start(self):
        native = Path("scripts/run_ovs_limited_matrix.sh").read_text()
        mooncake = Path("scripts/run_ovs_limited_mooncake_matrix.sh").read_text()
        parent = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()
        for script in (native, mooncake, parent):
            self.assertIn("container-gpu-preflight.txt", script)
            self.assertIn("torch.cuda.device_count() == 1", script)
            self.assertLess(
                script.index("container-gpu-preflight.txt"),
                script.index("start_vllm", script.index("container-gpu-preflight.txt"))
                if "start_vllm" in script[script.index("container-gpu-preflight.txt") :]
                else script.index("start_nixl_vllm", script.index("container-gpu-preflight.txt")),
            )

    def test_static_preflight_can_freeze_objects_without_launching_services(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()

        self.assertIn('if [[ "${mode}" == "static-preflight" ]]', script)
        self.assertIn('write_static_preflight "$2"', script)
        self.assertIn('UNCHAIN_KV_HASH_MODEL_WEIGHTS:-0', script)
        self.assertIn("-name '*.safetensors'", script)
        self.assertIn("missing controller-provided git provenance", script)
        self.assertIn("scope=scripts,src,native", script)
        self.assertIn("source_manifest_sha256=", script)

    def test_launch_failure_cleans_up_and_returns_status(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()

        self.assertIn("local launch_status=0", script)
        self.assertIn('launch_native "${cell}" "${run_id}" "${vllm_args}" || launch_status="$?"', script)
        self.assertIn('if (( launch_status != 0 )); then', script)
        self.assertIn('return "${launch_status}"', script)

    def test_manifest_records_trace_and_monitor_configuration(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()

        self.assertIn('echo "trace_enabled=${TRACE_ENABLED}"', script)
        self.assertIn('echo "resource_monitor=${UNCHAIN_KV_RESOURCE_MONITOR:-0}"', script)

    def test_live_cap_drop_is_explicit_and_manifested(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()

        self.assertIn("UNCHAIN_KV_SPOOL_LIVE_CAP_DROP_BYTES", script)
        self.assertIn("live-cap drop requires a writable cap file", script)
        self.assertIn('echo "spool_live_cap_drop_bytes=${SPOOL_LIVE_CAP_DROP_BYTES}"', script)
        self.assertIn("resource_monitor_interval_s=", script)

    def test_fault_injection_is_opt_in_manifested_and_waited(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()

        self.assertIn('FAULT_ACTION="${UNCHAIN_KV_FAULT_ACTION:-}"', script)
        self.assertIn('echo "fault_action=${FAULT_ACTION}"', script)
        self.assertIn('if [[ -n "${FAULT_ACTION}" ]]; then', script)
        self.assertIn('wait "${FAULT_PID}" || benchmark_status=1', script)

    def test_latest_aliases_separate_pack_off_pack_on_and_all_methods(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()

        self.assertIn("raw|raw_layer_wise|raw_no_runs|layer_wise|writeback", script)
        self.assertIn("raw_layer_wise|raw_no_runs)", script)
        self.assertIn("gpu_pack_layers=0", script)
        self.assertIn('[[ "${cell}" == "raw_no_runs" ]] && block_runs=0', script)
        self.assertIn('UNCHAIN_KV_BLOCK_RUNS="${block_runs}"', script)
        self.assertIn("all_methods)", script)
        self.assertIn('if [[ "${REQUEST_SPOOL_AUTO}" != "0" ]]', script)
        self.assertIn('request_spool_auto="${REQUEST_SPOOL_AUTO}"', script)
        self.assertIn('request_spool_bytes="${FIXED_SPOOL_BYTES}"', script)

    def test_manifest_records_effective_codec_threshold(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()

        self.assertIn('CODEC_MIN_BLOCKS="${UNCHAIN_KV_CODEC_MIN_BLOCKS:-0}"', script)
        self.assertIn("UNCHAIN_KV_CANONICAL_CODEC_MIN_BLOCKS", script)
        self.assertIn('echo "codec_min_blocks=${codec_min_blocks}"', script)
        self.assertIn('echo "prefix_fast_wait_s=${PREFIX_FAST_WAIT_S}"', script)

    def test_context_raw_rows_do_not_shadow_finalized_requests(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()

        self.assertIn('--output "${run_root}/context-requests.jsonl"', script)
        self.assertIn("missing context prompt:", script)
        self.assertIn('echo "prefill_vllm_args=${prefill_vllm_args}"', script)
        self.assertIn('echo "decode_vllm_args=${decode_vllm_args}"', script)
        self.assertIn('echo "prompt_manifest_sha256=$(sha256sum', script)

    def test_mixed_context_benchmarks_can_import_scripts_package(self):
        script = Path("scripts/run_sharegpt_throughput_matrix.sh").read_text()

        self.assertEqual(
            script.count('-e PYTHONPATH="${ROOT}:${ROOT}/src"'),
            2,
        )


if __name__ == "__main__":
    unittest.main()
