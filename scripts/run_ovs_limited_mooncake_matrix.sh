#!/usr/bin/env bash
set -euo pipefail

ROOT="${UNCHAIN_KV_REMOTE_ROOT:-/workspace/unchain-kv}"
IMAGE="${UNCHAIN_KV_IMAGE:-unchain-kv-runtime:latest}"
MODEL="${UNCHAIN_KV_MODEL:-/models/Qwen2.5-7B-Instruct}"
BRIDGE="${UNCHAIN_KV_OVS_BRIDGE:-exp-br}"
RATE="${UNCHAIN_KV_NET_RATE:-10gbit}"
RUN_ID="${UNCHAIN_KV_RUN_ID:-ovs10g-mooncake-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${ROOT}/runs/${RUN_ID}"
PROMPT_DIR="${UNCHAIN_KV_PROMPT_DIR:-${ROOT}/runs/prompts}"
VLLM_ARGS="${UNCHAIN_KV_VLLM_ARGS:---max-model-len 32768 --max-num-batched-tokens 32768 --max-num-seqs 1 --gpu-memory-utilization 0.95 --enforce-eager}"
PREFILL_VLLM_ARGS="${UNCHAIN_KV_PREFILL_VLLM_ARGS:-${VLLM_ARGS}}"
DECODE_VLLM_ARGS="${UNCHAIN_KV_DECODE_VLLM_ARGS:-${VLLM_ARGS}}"
LENGTHS="${UNCHAIN_KV_LENGTHS:-1024 2048 4096 8192 16384 32000}"
SAMPLES="${UNCHAIN_KV_SAMPLES:-10}"
SAMPLE_OFFSET="${UNCHAIN_KV_SAMPLE_OFFSET:-0}"
WARMUP_PER_LENGTH="${UNCHAIN_KV_WARMUP_PER_LENGTH:-0}"
BETWEEN_REQUEST_SLEEP_S="${UNCHAIN_KV_BETWEEN_REQUEST_SLEEP_S:-0}"

PROXY_IP="${UNCHAIN_KV_PROXY_IP:-172.16.0.40}"
PREFILL_IP="${UNCHAIN_KV_PREFILL_IP:-172.16.0.41}"
DECODE_IP="${UNCHAIN_KV_DECODE_IP:-172.16.0.42}"
GATEWAY_IP="${UNCHAIN_KV_GATEWAY_IP:-172.16.0.1}"

PREFILL_GPU="${UNCHAIN_KV_PREFILL_GPU:-1}"
DECODE_GPU="${UNCHAIN_KV_DECODE_GPU:-0}"
BUFFER_SIZE="${UNCHAIN_KV_MOONCAKE_BUFFER_SIZE:-1000000000}"

mkdir -p "${RUN_ROOT}/logs"

need_root() {
  if [[ "$(id -u)" != "0" ]]; then
    echo "run with sudo; docker/ip/tc/ovs need root" >&2
    exit 2
  fi
}

cleanup_one() {
  local name="$1"
  local host_if="$2"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  ovs-vsctl --if-exists del-port "${BRIDGE}" "${host_if}" >/dev/null 2>&1 || true
  ip link del "${host_if}" >/dev/null 2>&1 || true
  rm -f "/var/run/netns/${name}"
}

attach_ovs() {
  local name="$1"
  local host_if="$2"
  local peer_if="$3"
  local ip_addr="$4"
  local pid
  pid="$(docker inspect -f '{{.State.Pid}}' "${name}")"
  ln -sf "/proc/${pid}/ns/net" "/var/run/netns/${name}"
  ip link add "${host_if}" type veth peer name "${peer_if}"
  ip link set "${peer_if}" netns "${name}"
  ovs-vsctl --may-exist add-port "${BRIDGE}" "${host_if}"
  ip link set "${host_if}" up
  ip netns exec "${name}" ip link set lo up
  ip netns exec "${name}" ip link set "${peer_if}" name eth0
  ip netns exec "${name}" ip addr add "${ip_addr}/24" dev eth0
  ip netns exec "${name}" ip link set eth0 up
  ip netns exec "${name}" ip route replace default via "${GATEWAY_IP}" dev eth0
  ip netns exec "${name}" tc qdisc replace dev eth0 root handle 1: htb default 10
  ip netns exec "${name}" tc class replace dev eth0 parent 1: classid 1:10 htb rate "${RATE}" ceil "${RATE}"
}

start_container() {
  local name="$1"
  local gpu="$2"
  local ip_addr="$3"
  local host_if="$4"
  local peer_if="$5"
  cleanup_one "${name}" "${host_if}"
  local gpu_args=()
  if [[ "${gpu}" != "none" ]]; then
    gpu_args=(--gpus "device=${gpu}")
  fi
  docker run -d --name "${name}" \
    --hostname "${name}" \
    --add-host "${name}:${ip_addr}" \
    --sysctl net.ipv4.ip_local_port_range="1024 65535" \
    --sysctl net.ipv4.tcp_tw_reuse=1 \
    "${gpu_args[@]}" \
    --network none \
    -v "${ROOT}:${ROOT}" \
    -v /models:/models:ro \
    -w "${ROOT}" \
    -e PYTHONPATH="${ROOT}/src" \
    -e GLOO_SOCKET_IFNAME=eth0 \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -e VLLM_HOST_IP="${ip_addr}" \
    -e VLLM_MOONCAKE_PROTOCOL=tcp \
    "${IMAGE}" sleep infinity >/dev/null
  attach_ovs "${name}" "${host_if}" "${peer_if}" "${ip_addr}"
}

mooncake_config() {
  local role="$1"
  local rank="$2"
  printf '{"kv_connector":"MooncakeConnector","kv_role":"%s","kv_rank":%s,"kv_parallel_size":2,"kv_buffer_device":"cuda","kv_buffer_size":%s,"kv_connector_extra_config":{"mooncake_protocol":"tcp"}}' \
    "${role}" "${rank}" "${BUFFER_SIZE}"
}

start_vllm() {
  local name="$1"
  local role="$2"
  local rank="$3"
  local host="$4"
  local port="$5"
  local log="$6"
  local config
  local role_vllm_args="${DECODE_VLLM_ARGS}"
  [[ "${role}" == "kv_producer" ]] && role_vllm_args="${PREFILL_VLLM_ARGS}"
  config="$(mooncake_config "${role}" "${rank}")"
  docker exec -d \
    -e UNCHAIN_KV_NORMALIZE_RELEASE="${UNCHAIN_KV_NORMALIZE_RELEASE:-0}" \
    "${name}" bash -lc "
    python3 -m unchain_kv.patch_vllm \$(python3 -c 'from pathlib import Path; import vllm; print(Path(vllm.__file__).resolve().parents[1])') &&
    vllm serve ${MODEL} --host ${host} --port ${port} \
      --kv-transfer-config '${config}' \
      ${role_vllm_args} > ${log} 2>&1
  "
}

wait_ready() {
  local host="$1"
  local port="$2"
  for _ in $(seq 1 "${UNCHAIN_KV_READY_ATTEMPTS:-300}"); do
    if curl -fsS "http://${host}:${port}/v1/models" >/dev/null 2>&1; then
      return
    fi
    sleep 2
  done
  echo "${host}:${port} not ready" >&2
  exit 3
}

write_tc_state() {
  {
    echo "rate=${RATE}"
    for name in kvp-moon-proxy kvp-moon-prefill kvp-moon-decode; do
      echo "=== ${name}"
      ip netns exec "${name}" tc qdisc show dev eth0
      ip netns exec "${name}" tc class show dev eth0
    done
  } >"${RUN_ROOT}/tc-state.txt"
}

measure_matrix() {
  python3 - "$RUN_ROOT" "$PROMPT_DIR" "$MODEL" "http://${PROXY_IP}:18082/v1/completions" "$LENGTHS" "$SAMPLES" "$SAMPLE_OFFSET" "$WARMUP_PER_LENGTH" "$BETWEEN_REQUEST_SLEEP_S" <<'PY'
from __future__ import annotations
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

run_root = Path(sys.argv[1])
prompt_dir = Path(sys.argv[2])
model = sys.argv[3]
url = sys.argv[4]
lengths = [int(x) for x in sys.argv[5].split()]
samples = int(sys.argv[6])
sample_offset = int(sys.argv[7])
warmup_per_length = int(sys.argv[8])
between_request_sleep_s = float(sys.argv[9])

def first_text(event):
    choices = event.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if choice.get("text"):
        return str(choice["text"])
    delta = choice.get("delta") or {}
    return str(delta.get("content") or "")

def post_stream(prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            body = line[5:].strip()
            if body == b"[DONE]":
                break
            event = json.loads(body.decode())
            if first_text(event) and first is None:
                first = time.perf_counter()
            chunks += 1
    end = time.perf_counter()
    return {
        "ok": first is not None,
        "ttft_s": None if first is None else first - start,
        "e2e_s": end - start,
        "chunks": chunks,
    }

def pct(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    return values[min(len(values) - 1, max(0, round((len(values) - 1) * q)))]

def summarize(rows):
    ok = [r for r in rows if r.get("ok") and r.get("ttft_s") is not None]
    ttft = [float(r["ttft_s"]) for r in ok]
    e2e = [float(r["e2e_s"]) for r in ok]
    return {
        "count": len(rows),
        "ok": len(ok),
        "ttft_median_s": statistics.median(ttft) if ttft else 0.0,
        "ttft_p90_s": pct(ttft, 0.90),
        "ttft_p99_s": pct(ttft, 0.99),
        "ttft_min_s": min(ttft, default=0.0),
        "ttft_max_s": max(ttft, default=0.0),
        "e2e_median_s": statistics.median(e2e) if e2e else 0.0,
        "e2e_p90_s": pct(e2e, 0.90),
        "e2e_p99_s": pct(e2e, 0.99),
        "e2e_min_s": min(e2e, default=0.0),
        "e2e_max_s": max(e2e, default=0.0),
    }

summary = {}
warmup_out = run_root / "warmup-by-length.jsonl"
if warmup_per_length:
    warmup_out.write_text("")
for length in lengths:
    for warmup_index in range(warmup_per_length):
        prompt = (prompt_dir / f"prompt-{length}-0.txt").read_text()
        row = post_stream(prompt)
        row.update({"index": warmup_index, "prompt_tokens_target": length})
        with warmup_out.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print("warmup", length, json.dumps(row, sort_keys=True), flush=True)
    rows = []
    out = run_root / f"ttft-{length}.jsonl"
    out.write_text("")
    for index in range(samples):
        sample_index = sample_offset + index
        prompt = (prompt_dir / f"prompt-{length}-{sample_index}.txt").read_text()
        row = post_stream(prompt)
        row.update({"index": sample_index, "prompt_tokens_target": length, "prompt_chars": len(prompt)})
        rows.append(row)
        with out.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(length, index, json.dumps(row, sort_keys=True), flush=True)
        if between_request_sleep_s:
            time.sleep(between_request_sleep_s)
    summary[str(length)] = summarize(rows)
(run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

need_root
cleanup_one kvp-moon-proxy moon-proxy-h
cleanup_one kvp-moon-prefill moon-pref-h
cleanup_one kvp-moon-decode moon-dec-h

start_container kvp-moon-decode "${DECODE_GPU}" "${DECODE_IP}" moon-dec-h moon-dec-c
start_container kvp-moon-prefill "${PREFILL_GPU}" "${PREFILL_IP}" moon-pref-h moon-pref-c
start_container kvp-moon-proxy none "${PROXY_IP}" moon-proxy-h moon-proxy-c
write_tc_state

for name in kvp-moon-decode kvp-moon-prefill; do
  echo "=== ${name}"
  docker exec "${name}" nvidia-smi -L
  docker exec "${name}" python3 -c \
    'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1'
done >"${RUN_ROOT}/container-gpu-preflight.txt" 2>&1

start_vllm kvp-moon-decode kv_consumer 1 "${DECODE_IP}" 18081 "${RUN_ROOT}/logs/decode.docker.log"
start_vllm kvp-moon-prefill kv_producer 0 "${PREFILL_IP}" 18080 "${RUN_ROOT}/logs/prefill.docker.log"
wait_ready "${DECODE_IP}" 18081
wait_ready "${PREFILL_IP}" 18080

docker exec -d kvp-moon-proxy bash -lc "
  python3 scripts/mooncake_proxy.py \
    --listen ${PROXY_IP}:18082 \
    --prefill-url http://${PREFILL_IP}:18080 \
    --prefill-bootstrap-url http://${PREFILL_IP}:8998 \
    --decode-url http://${DECODE_IP}:18081 \
    --timeout-s 900 \
    > ${RUN_ROOT}/logs/proxy.docker.log 2>&1
"
for _ in $(seq 1 60); do
  if curl -fsS "http://${PROXY_IP}:18082/healthcheck" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

measure_matrix
echo "${RUN_ID}"
