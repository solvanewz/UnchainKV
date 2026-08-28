#!/usr/bin/env python3
"""Probe whether this host can support GPU/registered-memory KV transfer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def run(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        return result.returncode, result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def collect() -> dict[str, object]:
    _, nvidia = run(["nvidia-smi", "-L"])
    _, modules = run(["lsmod"])
    _, libs = run(["ldconfig", "-p"])
    _, pci = run(["lspci", "-nn"])
    _, ibv = run(["ibv_devinfo"])
    _, topo = run(["nvidia-smi", "topo", "-m"])
    infiniband = sorted(str(path) for path in Path("/dev/infiniband").glob("*"))
    devices = [path for path in ["/dev/gdrdrv", "/dev/nvidia0"] if os.path.exists(path)]
    return {
        "gpu_count": len(re.findall(r"^GPU \d+:", nvidia, flags=re.MULTILINE)),
        "nvidia_smi": lines(nvidia),
        "topology": lines(topo),
        "infiniband_devices": infiniband,
        "ibv_devinfo": lines(ibv),
        "modules": lines(modules),
        "libs": lines(libs),
        "devices": devices,
        "network_pci": [
            line
            for line in lines(pci)
            if re.search(r"ethernet|network|infiniband|mellanox", line, re.I)
        ],
    }


def has_any(values: list[str], needles: tuple[str, ...]) -> bool:
    text = "\n".join(values).lower()
    return any(needle.lower() in text for needle in needles)


def summarize(data: dict[str, object]) -> dict[str, object]:
    libs = list(data.get("libs", []))
    modules = list(data.get("modules", []))
    devices = list(data.get("devices", []))
    infiniband = list(data.get("infiniband_devices", []))
    network_pci = list(data.get("network_pci", []))
    has_ibverbs = has_any(libs, ("libibverbs",))
    has_rdmacm = has_any(libs, ("librdmacm",))
    has_gdrapi = has_any(libs, ("libgdrapi",))
    has_cuda = has_any(libs, ("libcuda",))
    has_rdma_dev = bool(infiniband)
    has_gdrdrv = "/dev/gdrdrv" in devices
    has_peermem = has_any(modules, ("nvidia_peermem", "nv_peer_mem"))
    has_rdma_nic = has_any(network_pci, ("mellanox", "infiniband"))
    missing = []
    if not has_rdma_dev:
        missing.append("/dev/infiniband")
    if not has_rdma_nic:
        missing.append("RDMA-capable NIC")
    if not has_ibverbs:
        missing.append("libibverbs")
    if not has_rdmacm:
        missing.append("librdmacm")
    if not has_gdrapi:
        missing.append("libgdrapi")
    if not has_gdrdrv:
        missing.append("/dev/gdrdrv")
    if not has_peermem:
        missing.append("nvidia_peermem")
    if not has_cuda:
        missing.append("libcuda")
    return {
        "rdma_ready": has_rdma_dev and has_rdma_nic and has_ibverbs and has_rdmacm,
        "gdr_ready": has_cuda and has_gdrapi and has_gdrdrv,
        "gpudirect_rdma_ready": (
            has_rdma_dev
            and has_rdma_nic
            and has_ibverbs
            and has_rdmacm
            and has_peermem
        ),
        "missing": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()
    data = collect()
    data["summary"] = summarize(data)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    summary = data["summary"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
