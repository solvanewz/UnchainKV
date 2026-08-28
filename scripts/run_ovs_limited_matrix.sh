#!/usr/bin/env bash
set -euo pipefail

ROOT="${UNCHAIN_KV_REMOTE_ROOT:-/workspace/unchain-kv}"
IMAGE="${UNCHAIN_KV_IMAGE:-unchain-kv-runtime:latest}"
MODEL="${UNCHAIN_KV_MODEL:-/models/Qwen2.5-7B-Instruct}"
EXPECTED_LAYERS="${UNCHAIN_KV_EXPECTED_LAYERS:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["num_hidden_layers"])' "${MODEL}/config.json")}"
TCP_LIB="${UNCHAIN_KV_TCP_LIB:-${ROOT}/native/unchain_kv/build/libunchain_kv_tcp.so}"
BRIDGE="${UNCHAIN_KV_OVS_BRIDGE:-exp-br}"
RATE="${UNCHAIN_KV_NET_RATE:-10gbit}"
RUN_ID="${UNCHAIN_KV_RUN_ID:-ovs-limited-native-hostmirror-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${ROOT}/runs/${RUN_ID}"
PROMPT_DIR="${UNCHAIN_KV_PROMPT_DIR:-${ROOT}/runs/prompts}"
VLLM_ARGS="${UNCHAIN_KV_VLLM_ARGS:---max-model-len 32768 --max-num-batched-tokens 32768 --max-num-seqs 1 --gpu-memory-utilization 0.80 --enforce-eager}"
PREFILL_VLLM_ARGS="${UNCHAIN_KV_PREFILL_VLLM_ARGS:-${VLLM_ARGS}}"
DECODE_VLLM_ARGS="${UNCHAIN_KV_DECODE_VLLM_ARGS:-${VLLM_ARGS}}"
LENGTHS="${UNCHAIN_KV_LENGTHS:-1024 2048 4096 8192 16384 32000}"
SAMPLES="${UNCHAIN_KV_SAMPLES:-10}"
WARMUP="${UNCHAIN_KV_WARMUP:-0}"
WARMUP_PER_LENGTH="${UNCHAIN_KV_WARMUP_PER_LENGTH:-0}"
HOST_MIRROR_LAYERS="${UNCHAIN_KV_HOST_MIRROR_LAYERS:-28}"
PINNED_STAGING="${UNCHAIN_KV_PINNED_STAGING:-0}"
EXTENT_ALLOC="${UNCHAIN_KV_EXTENT_ALLOC:-off}"

PROXY_IP="${UNCHAIN_KV_PROXY_IP:-172.16.0.30}"
PREFILL_IP="${UNCHAIN_KV_PREFILL_IP:-172.16.0.31}"
DECODE_IP="${UNCHAIN_KV_DECODE_IP:-172.16.0.32}"
GATEWAY_IP="${UNCHAIN_KV_GATEWAY_IP:-172.16.0.1}"

PREFILL_GPU="${UNCHAIN_KV_PREFILL_GPU:-1}"
DECODE_GPU="${UNCHAIN_KV_DECODE_GPU:-0}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/trace"

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

apply_rate() {
  local ns="$1"
  ip netns exec "${ns}" tc qdisc replace dev eth0 root handle 1: htb default 10
  ip netns exec "${ns}" tc class replace dev eth0 parent 1: classid 1:10 htb rate "${RATE}" ceil "${RATE}"
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
  apply_rate "${name}"
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
    "${gpu_args[@]}" \
    --network none \
    -v "${ROOT}:${ROOT}" \
    -v /models:/models:ro \
    -w "${ROOT}" \
    -e PYTHONSAFEPATH=1 \
    -e PYTHONPATH="${ROOT}/src" \
    -e GLOO_SOCKET_IFNAME=eth0 \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -e VLLM_HOST_IP="${ip_addr}" \
    -e UNCHAIN_KV_TRANSPORT=tcp \
    -e UNCHAIN_KV_TCP_LIB="${TCP_LIB}" \
    -e UNCHAIN_KV_CHUNK_BYTES=32768 \
    -e UNCHAIN_KV_MAX_BLOCKS=0 \
    -e UNCHAIN_KV_BULK_DECODE="${UNCHAIN_KV_BULK_DECODE:-0}" \
    -e UNCHAIN_KV_WAIT_TIMEOUT_S="${UNCHAIN_KV_WAIT_TIMEOUT_S:-300}" \
    -e UNCHAIN_KV_EXPECTED_LAYERS="${EXPECTED_LAYERS}" \
    -e UNCHAIN_KV_RECV_BUFFER_BYTES=268435456 \
    -e UNCHAIN_KV_GRANT_WINDOW="${UNCHAIN_KV_GRANT_WINDOW:-0}" \
    -e UNCHAIN_KV_RESTORE_AHEAD="${UNCHAIN_KV_RESTORE_AHEAD:-0}" \
    -e UNCHAIN_KV_HOST_MIRROR_LAYERS="${HOST_MIRROR_LAYERS}" \
    -e UNCHAIN_KV_HOST_MIRROR_BYTES="${UNCHAIN_KV_HOST_MIRROR_BYTES:-0}" \
    -e UNCHAIN_KV_REQUEST_SPOOL_BYTES="${UNCHAIN_KV_REQUEST_SPOOL_BYTES:-0}" \
    -e UNCHAIN_KV_REQUEST_SPOOL_AUTO="${UNCHAIN_KV_REQUEST_SPOOL_AUTO:-0}" \
    -e UNCHAIN_KV_HOST_GUARD_BYTES="${UNCHAIN_KV_HOST_GUARD_BYTES:-2147483648}" \
    -e UNCHAIN_KV_GPU_GUARD_BYTES="${UNCHAIN_KV_GPU_GUARD_BYTES:-536870912}" \
    -e UNCHAIN_KV_AUTO_SPOOL_HARD_BYTES="${UNCHAIN_KV_AUTO_SPOOL_HARD_BYTES:-0}" \
    -e UNCHAIN_KV_SPOOL_PRESSURE_RATIO="${UNCHAIN_KV_SPOOL_PRESSURE_RATIO:-1.15}" \
    -e UNCHAIN_KV_SPOOL_LIVE_CAP_FILE="${UNCHAIN_KV_SPOOL_LIVE_CAP_FILE:-}" \
    -e UNCHAIN_KV_NATIVE_LAYOUT_PAYLOAD=1 \
    -e UNCHAIN_KV_BLOCK_RUNS="${UNCHAIN_KV_BLOCK_RUNS:-1}" \
    -e UNCHAIN_KV_KV_MAJOR_PAYLOAD=0 \
    -e UNCHAIN_KV_PINNED_STAGING="${PINNED_STAGING}" \
    -e UNCHAIN_KV_SEND_WORKERS="${UNCHAIN_KV_SEND_WORKERS:-1}" \
    -e UNCHAIN_KV_SEND_INFLIGHT="${UNCHAIN_KV_SEND_INFLIGHT:-0}" \
    -e UNCHAIN_KV_LAYER_GROUP_SIZE=1 \
    -e UNCHAIN_KV_EARLY_STAGE="${UNCHAIN_KV_EARLY_STAGE:-0}" \
    -e UNCHAIN_KV_CODEC="${UNCHAIN_KV_CODEC:-}" \
    -e UNCHAIN_KV_CODEC_MIN_BLOCKS="${UNCHAIN_KV_CODEC_MIN_BLOCKS:-0}" \
    -e UNCHAIN_KV_CODEC_GPU_BYTES="${UNCHAIN_KV_CODEC_GPU_BYTES:-1073741824}" \
    -e UNCHAIN_KV_SPLITZIP_FIXED5="${UNCHAIN_KV_SPLITZIP_FIXED5:-0}" \
    -e UNCHAIN_KV_SPLITZIP_FIXED5_LAYERS="${UNCHAIN_KV_SPLITZIP_FIXED5_LAYERS:-}" \
    -e UNCHAIN_KV_SPLITZIP_TOP16="${UNCHAIN_KV_SPLITZIP_TOP16:-0}" \
    -e UNCHAIN_KV_SPLITZIP_TOP16_MAX_COOLDOWN="${UNCHAIN_KV_SPLITZIP_TOP16_MAX_COOLDOWN:-32}" \
    -e UNCHAIN_KV_CODEC_WRITEBACK="${UNCHAIN_KV_CODEC_WRITEBACK:-}" \
    -e UNCHAIN_KV_CODEC_WRITEBACK_STRICT="${UNCHAIN_KV_CODEC_WRITEBACK_STRICT:-0}" \
    -e UNCHAIN_KV_SPLITZIP_CHUNKS="${UNCHAIN_KV_SPLITZIP_CHUNKS:-1}" \
    -e UNCHAIN_KV_SPLITZIP_NATIVE_DECODE="${UNCHAIN_KV_SPLITZIP_NATIVE_DECODE:-0}" \
    -e UNCHAIN_KV_SPLITZIP_LIB="${UNCHAIN_KV_SPLITZIP_LIB:-${ROOT}/native/unchain_kv/build/libunchain_kv_splitzip_cuda.so}" \
    -e UNCHAIN_KV_GPU_PACK_LAYERS="${UNCHAIN_KV_GPU_PACK_LAYERS:-0}" \
    -e UNCHAIN_KV_GPU_PACK_BYTES="${UNCHAIN_KV_GPU_PACK_BYTES:-0}" \
    -e UNCHAIN_KV_GPU_PACK_STRICT="${UNCHAIN_KV_GPU_PACK_STRICT:-0}" \
    -e UNCHAIN_KV_PAYLOAD_READY="${UNCHAIN_KV_PAYLOAD_READY:-1}" \
    -e UNCHAIN_KV_EARLY_STAGE_PACK_ONLY="${UNCHAIN_KV_EARLY_STAGE_PACK_ONLY:-0}" \
    -e UNCHAIN_KV_PERMUTE_BLOCK_IDS="${UNCHAIN_KV_PERMUTE_BLOCK_IDS:-}" \
    -e UNCHAIN_KV_TRACE_ENABLED="${UNCHAIN_KV_TRACE_ENABLED:-1}" \
    -e UNCHAIN_KV_TRACE_CUDA="${UNCHAIN_KV_TRACE_CUDA:-0}" \
    -e UNCHAIN_KV_TRACE_BF16_EXPONENTS="${UNCHAIN_KV_TRACE_BF16_EXPONENTS:-0}" \
    -e UNCHAIN_KV_TRACE_PREFILL_WINDOW="${UNCHAIN_KV_TRACE_PREFILL_WINDOW:-0}" \
    -e UNCHAIN_KV_VLLM_ARGS="${VLLM_ARGS}" \
    "${IMAGE}" sleep infinity >/dev/null
  attach_ovs "${name}" "${host_if}" "${peer_if}" "${ip_addr}"
}

start_vllm() {
  local name="$1"
  local role="$2"
  local host="$3"
  local port="$4"
  local bind="$5"
  local peer="$6"
  local log="$7"
  local trace="$8"
  local extent_alloc=off
  local role_vllm_args="${DECODE_VLLM_ARGS}"
  if [[ "${role}" == "kv_producer" ]]; then
    extent_alloc="${EXTENT_ALLOC:-off}"
    role_vllm_args="${PREFILL_VLLM_ARGS}"
  fi
  docker exec -d \
    -e UNCHAIN_KV_BIND="${bind}" \
    -e UNCHAIN_KV_PEER="${peer}" \
    -e UNCHAIN_KV_TRACE="${trace}" \
    -e UNCHAIN_KV_EXTENT_ALLOC="${extent_alloc}" \
    -e UNCHAIN_KV_NORMALIZE_RELEASE="${UNCHAIN_KV_NORMALIZE_RELEASE:-0}" \
    -e UNCHAIN_KV_EXTENT_RESERVE_BLOCKS="${UNCHAIN_KV_EXTENT_RESERVE_BLOCKS:-0}" \
    "${name}" bash -lc "
      {
        python3 -m unchain_kv.patch_vllm \$(python3 -c 'from pathlib import Path; import vllm; print(Path(vllm.__file__).resolve().parents[1])') &&
        vllm serve ${MODEL} --host ${host} --port ${port} \
          --kv-transfer-config '{\"kv_connector\":\"UnchainKVConnector\",\"kv_role\":\"${role}\"}' \
          ${role_vllm_args}
      } > ${log} 2>&1
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
    for name in kvp-proxy kvp-prefill kvp-decode; do
      echo "=== ${name}"
      ip netns exec "${name}" tc qdisc show dev eth0
      ip netns exec "${name}" tc class show dev eth0
    done
  } >"${RUN_ROOT}/tc-state.txt"
}

measure_matrix() {
  python3 - "$RUN_ROOT" "$PROMPT_DIR" "$MODEL" "http://${PROXY_IP}:18082/v1/completions" "$LENGTHS" "$SAMPLES" "$WARMUP" "$WARMUP_PER_LENGTH" <<'PY'
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
lengths = [int(item) for item in sys.argv[5].split()]
samples = int(sys.argv[6])
warmup = int(sys.argv[7])
warmup_per_length = int(sys.argv[8])

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

if warmup > 0:
    prompt = (prompt_dir / "prompt-1024-0.txt").read_text()
    warmup_rows = [post_stream(prompt) for _ in range(warmup)]
    (run_root / "warmup.json").write_text(
        json.dumps(warmup_rows, indent=2, sort_keys=True)
    )
    print("warmup", json.dumps(warmup_rows, sort_keys=True), flush=True)

summary = {}
warmup_by_length_out = run_root / "warmup-by-length.jsonl"
if warmup_per_length > 0:
    warmup_by_length_out.write_text("")
for length in lengths:
    for warmup_index in range(warmup_per_length):
        prompt = (prompt_dir / f"prompt-{length}-0.txt").read_text()
        row = post_stream(prompt)
        row.update({
            "index": warmup_index,
            "prompt_tokens_target": length,
            "prompt_chars": len(prompt),
        })
        with warmup_by_length_out.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print("warmup", length, json.dumps(row, sort_keys=True), flush=True)
    rows = []
    out = run_root / f"ttft-{length}.jsonl"
    out.write_text("")
    for index in range(samples):
        prompt = (prompt_dir / f"prompt-{length}-{index}.txt").read_text()
        row = post_stream(prompt)
        row.update({"index": index, "prompt_tokens_target": length, "prompt_chars": len(prompt)})
        rows.append(row)
        with out.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(length, index, json.dumps(row, sort_keys=True), flush=True)
    summary[str(length)] = summarize(rows)
(run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

need_root
cleanup_one kvp-proxy kvp-proxy-h
cleanup_one kvp-prefill kvp-pref-h
cleanup_one kvp-decode kvp-dec-h

start_container kvp-decode "${DECODE_GPU}" "${DECODE_IP}" kvp-dec-h kvp-dec-c
start_container kvp-prefill "${PREFILL_GPU}" "${PREFILL_IP}" kvp-pref-h kvp-pref-c
start_container kvp-proxy none "${PROXY_IP}" kvp-proxy-h kvp-proxy-c
write_tc_state

for name in kvp-decode kvp-prefill; do
  echo "=== ${name}"
  docker exec "${name}" nvidia-smi -L
  docker exec "${name}" python3 -c \
    'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1'
done >"${RUN_ROOT}/container-gpu-preflight.txt" 2>&1

start_vllm kvp-decode kv_consumer "${DECODE_IP}" 18081 "${DECODE_IP}:29101" "${PREFILL_IP}:29100" "${RUN_ROOT}/logs/decode.docker.log" "${RUN_ROOT}/trace/decode.jsonl"
start_vllm kvp-prefill kv_producer "${PREFILL_IP}" 18080 "${PREFILL_IP}:29100" "${DECODE_IP}:29101" "${RUN_ROOT}/logs/prefill.docker.log" "${RUN_ROOT}/trace/prefill.jsonl"
wait_ready "${DECODE_IP}" 18081
wait_ready "${PREFILL_IP}" 18080

docker exec -d kvp-proxy bash -lc "
  python3 scripts/unchain_kv_proxy.py \
    --listen ${PROXY_IP}:18082 \
    --prefill-url http://${PREFILL_IP}:18080 \
    --decode-url http://${DECODE_IP}:18081 \
    --decode-lead-s 0 \
    --decode-slots "${UNCHAIN_KV_DECODE_SLOTS:-0}" \
    --metrics "${RUN_ROOT}/proxy-metrics.jsonl" \
    --timeout-s 900 \
    > ${RUN_ROOT}/logs/proxy.docker.log 2>&1
"
for _ in $(seq 1 60); do
  if curl -fsS "http://${PROXY_IP}:18082/v1/models" >/dev/null 2>&1; then
    break
  fi
  if timeout 1 bash -c "cat < /dev/null > /dev/tcp/${PROXY_IP}/18082" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

measure_matrix
if [[ "${UNCHAIN_KV_TRACE_ENABLED:-1}" != "0" ]] &&
   (( SAMPLES > 0 || WARMUP > 0 || WARMUP_PER_LENGTH > 0 )); then
  cp "${RUN_ROOT}/trace/prefill.jsonl" "${RUN_ROOT}/kv_producer-trace.jsonl"
  cp "${RUN_ROOT}/trace/decode.jsonl" "${RUN_ROOT}/kv_consumer-trace.jsonl"
fi
echo "${RUN_ID}"
