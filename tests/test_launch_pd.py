from pathlib import Path
import unittest


class LaunchPdTest(unittest.TestCase):
    def test_passes_connector_tuning_env_vars(self):
        script = Path("scripts/launch_pd.sh").read_text()
        for name in [
            "UNCHAIN_KV_KV_MAJOR_PAYLOAD",
            "UNCHAIN_KV_NATIVE_LAYOUT_PAYLOAD",
            "UNCHAIN_KV_TCP_LIB",
            "UNCHAIN_KV_PINNED_STAGING",
            "UNCHAIN_KV_HOST_MIRROR_LAYERS",
            "UNCHAIN_KV_REQUEST_SPOOL_AUTO",
            "UNCHAIN_KV_HOST_GUARD_BYTES",
            "UNCHAIN_KV_GPU_GUARD_BYTES",
            "UNCHAIN_KV_SEND_WORKERS",
            "UNCHAIN_KV_SEND_INFLIGHT",
            "UNCHAIN_KV_LAYER_GROUP_SIZE",
        ]:
            self.assertGreaterEqual(script.count(f"-e {name}="), 2, name)

    def test_checks_configured_gpus_before_launch(self):
        script = Path("scripts/launch_pd.sh").read_text()

        self.assertIn("require_gpu", script)
        self.assertIn("nvidia-smi -L", script)
        self.assertIn('grep -q "^GPU ${gpu_id}:"', script)
        self.assertIn('require_gpu "${UNCHAIN_KV_DECODE_GPU}"', script)
        self.assertIn('require_gpu "${UNCHAIN_KV_PREFILL_GPU}"', script)

    def test_can_launch_prefill_after_decode_is_ready(self):
        script = Path("scripts/launch_pd.sh").read_text()

        self.assertIn("UNCHAIN_KV_SEQUENTIAL_LAUNCH", script)
        self.assertIn("wait_api_ready", script)
        self.assertIn("${UNCHAIN_KV_DECODE_API_PORT}", script)



    def test_decode_extent_alloc_is_explicitly_off(self):
        script = Path("scripts/launch_pd.sh").read_text()
        # Must appear only in the decode container, not shared with prefill
        self.assertIn('-e UNCHAIN_KV_EXTENT_ALLOC=off \\', script)

    def test_prefill_extent_alloc_is_parametrized(self):
        script = Path("scripts/launch_pd.sh").read_text()
        self.assertIn('-e UNCHAIN_KV_EXTENT_ALLOC="${UNCHAIN_KV_EXTENT_ALLOC:-off}" \\', script)
        self.assertIn('-e UNCHAIN_KV_EXTENT_RESERVE_BLOCKS=0 \\', script)
        self.assertIn(
            '-e UNCHAIN_KV_EXTENT_RESERVE_BLOCKS="${UNCHAIN_KV_EXTENT_RESERVE_BLOCKS:-0}" \\',
            script,
        )

    def test_extent_alloc_env_example_defaults_to_bounded_prefer(self):
        text = Path("configs/unchain-kv.env.example").read_text()
        self.assertIn("UNCHAIN_KV_EXTENT_ALLOC=prefer", text)
        self.assertIn("UNCHAIN_KV_GPU_PACK_LAYERS=1", text)
        self.assertIn("UNCHAIN_KV_GPU_PACK_BYTES=67108864", text)
        self.assertIn("UNCHAIN_KV_GPU_PACK_STRICT=1", text)
        self.assertIn("UNCHAIN_KV_EXTENT_RESERVE_BLOCKS=0", text)
        self.assertIn("UNCHAIN_KV_PAYLOAD_READY=1", text)
        self.assertIn("UNCHAIN_KV_REQUEST_SPOOL_AUTO=0", text)

    def test_gpu_pack_byte_cap_reaches_both_services(self):
        script = Path("scripts/launch_pd.sh").read_text()
        self.assertEqual(script.count("UNCHAIN_KV_GPU_PACK_BYTES="), 2)

if __name__ == "__main__":
    unittest.main()
