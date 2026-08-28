import importlib.util
import os
from pathlib import Path
import socket
import json
import tempfile
import threading
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_bandwidth_replay.py"
SPEC = importlib.util.spec_from_file_location("bench_bandwidth_replay", SCRIPT)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class BenchBandwidthReplayTest(unittest.TestCase):
    def test_recv_checksum_detects_corruption(self):
        left, right = socket.socketpair()
        try:
            left.sendall(b"payload")
            checksum = replay._recv_checksum(right, 7)
        finally:
            left.close()
            right.close()

        self.assertNotEqual(checksum, 0)
        self.assertNotEqual(checksum, replay.crc32(b"payloax"))

    def test_payload_sizes_match_registered_32k_values(self):
        self.assertEqual(replay.payload_sizes(32000), (65_536_000, 49_971_205))

    def test_tcp_replay_drains_frames_without_storing_payloads(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.close()
        server_result = {}

        def serve():
            server_result.update(replay.serve_tcp(("127.0.0.1", port), 8, 5))

        old = os.environ.pop("UNCHAIN_KV_TCP_LIB", None)
        thread = threading.Thread(target=serve)
        thread.start()
        try:
            client = replay.replay_tcp(
                ("127.0.0.1", port),
                "writeback",
                1024,
                768,
                4,
                2,
                3,
                1,
                2,
                100,
            )
        finally:
            thread.join(5)
            if old is not None:
                os.environ["UNCHAIN_KV_TCP_LIB"] = old

        self.assertTrue(client["ok"])
        self.assertEqual(client["application_bytes"], 3 * 2 * 768)
        self.assertTrue(server_result["ok"])
        self.assertEqual(server_result["frames"], 8)

    def test_summarizes_network_and_pcie_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            network = root / "bw-net-formal-r1-raw"
            network.mkdir()
            (network / "client.json").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "path": "raw",
                        "offered_rps": 0.75,
                        "completed_rps": 0.62,
                        "application_gbps": 9.2,
                        "queue_wait_s": {"p99": 2.8},
                        "service_s": {"p50": 1.59},
                    }
                ),
                encoding="utf-8",
            )
            (network / "tc-after.txt").write_text(
                " Sent 10 bytes 1 pkt (dropped 0, overlimits 0 requeues 0)\n",
                encoding="utf-8",
            )
            pcie = root / "bw-pcie-formal-r1-wb.json"
            pcie.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "path": "writeback",
                        "direction": "d2h",
                        "layout": "fragmented",
                        "offered_rps": 14,
                        "completed_rps": 13.9,
                        "effective_gbps": 19.5,
                        "queue_wait_s": {"p99": 0.0002},
                        "operation_ms": {"p50": 90},
                    }
                ),
                encoding="utf-8",
            )

            result = replay.summarize_replays([network, pcie])

        self.assertTrue(result["validation"]["all_ok"])
        self.assertEqual(result["validation"]["network_dropped"], 0)
        self.assertEqual(
            result["network_by_path"]["raw"]["completed_rps"]["median"],
            0.62,
        )
        self.assertEqual(
            result["pcie_by_path"]["writeback"]["effective_gbytes_s"][
                "median"
            ],
            19.5,
        )


if __name__ == "__main__":
    unittest.main()
