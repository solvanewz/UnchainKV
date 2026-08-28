#!/usr/bin/env bash
set -euo pipefail

ROOT="${UNCHAIN_KV_REMOTE_ROOT:-/workspace/unchain-kv}"
IMAGE="${UNCHAIN_KV_IMAGE:-unchain-kv-runtime:latest}"
MODEL="${UNCHAIN_KV_MODEL:-/models/Qwen2.5-7B-Instruct}"
DATASET="${UNCHAIN_KV_SHAREGPT_DATASET:-/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json}"
PROMPT_DIR="${UNCHAIN_KV_PROMPT_DIR:-${ROOT}/runs/prompts}"
PROMPT_MANIFEST="${UNCHAIN_KV_PROMPT_MANIFEST:-}"
BRIDGE="${UNCHAIN_KV_OVS_BRIDGE:-exp-br}"
RATE="${UNCHAIN_KV_NET_RATE:-10gbit}"
MTU="${UNCHAIN_KV_MTU:-1500}"
MIN_DISK_KIB="${UNCHAIN_KV_MIN_DISK_KIB:-52428800}"
MATRIX_ID="${UNCHAIN_KV_MATRIX_ID:-sharegpt-c4-$(date -u +%Y%m%dT%H%M%SZ)}"
SEED="${UNCHAIN_KV_BENCH_SEED:-20260718}"
RESUME="${UNCHAIN_KV_RESUME:-0}"
TRACE_ENABLED="${UNCHAIN_KV_TRACE_ENABLED:-0}"
BENCH_DATASET="${UNCHAIN_KV_BENCH_DATASET:-sharegpt}"
REQUEST_RATE="${UNCHAIN_KV_REQUEST_RATE:-inf}"
BENCH_MAX_DURATION_S="${UNCHAIN_KV_BENCH_MAX_DURATION_S:-0}"
BURSTINESS="${UNCHAIN_KV_BURSTINESS:-1.0}"
RANDOM_INPUT_LEN="${UNCHAIN_KV_RANDOM_INPUT_LEN:-32000}"
RANDOM_OUTPUT_LEN="${UNCHAIN_KV_RANDOM_OUTPUT_LEN:-1}"
RANDOM_PREFIX_LEN="${UNCHAIN_KV_RANDOM_PREFIX_LEN:-0}"
CONTEXT_SAMPLE_OFFSET="${UNCHAIN_KV_CONTEXT_SAMPLE_OFFSET:-0}"
CONTEXT_PROMPT_CYCLE="${UNCHAIN_KV_CONTEXT_PROMPT_CYCLE:-0}"
HOST_MIRROR_BYTES="${UNCHAIN_KV_HOST_MIRROR_BYTES:-268435456}"
CODEC_GPU_BYTES="${UNCHAIN_KV_CODEC_GPU_BYTES:-268435456}"
CODEC_MIN_BLOCKS="${UNCHAIN_KV_CODEC_MIN_BLOCKS:-0}"
CANONICAL_CODEC_MIN_BLOCKS="${UNCHAIN_KV_CANONICAL_CODEC_MIN_BLOCKS:-128}"
PREFIX_FAST_WAIT_S="${UNCHAIN_KV_PREFIX_FAST_WAIT_S:-1.0}"
GPU_PACK_LAYERS="${UNCHAIN_KV_GPU_PACK_LAYERS:-6}"
GPU_PACK_BYTES="${UNCHAIN_KV_GPU_PACK_BYTES:-0}"
CANONICAL_GPU_PACK_BYTES="${UNCHAIN_KV_CANONICAL_GPU_PACK_BYTES:-67108864}"
GPU_PACK_STRICT="${UNCHAIN_KV_GPU_PACK_STRICT:-0}"
PAYLOAD_READY="${UNCHAIN_KV_PAYLOAD_READY:-1}"
REQUEST_SPOOL_BYTES="${UNCHAIN_KV_REQUEST_SPOOL_BYTES:-0}"
FIXED_SPOOL_BYTES="${UNCHAIN_KV_FIXED_SPOOL_BYTES:-4294967296}"
REQUEST_SPOOL_AUTO="${UNCHAIN_KV_REQUEST_SPOOL_AUTO:-0}"
HOST_GUARD_BYTES="${UNCHAIN_KV_HOST_GUARD_BYTES:-2147483648}"
GPU_GUARD_BYTES="${UNCHAIN_KV_GPU_GUARD_BYTES:-536870912}"
AUTO_SPOOL_HARD_BYTES="${UNCHAIN_KV_AUTO_SPOOL_HARD_BYTES:-0}"
SPOOL_PRESSURE_RATIO="${UNCHAIN_KV_SPOOL_PRESSURE_RATIO:-1.15}"
SPOOL_LIVE_CAP_FILE="${UNCHAIN_KV_SPOOL_LIVE_CAP_FILE:-}"
SPOOL_LIVE_CAP_DROP_BYTES="${UNCHAIN_KV_SPOOL_LIVE_CAP_DROP_BYTES:-0}"
SPOOL_LIVE_CAP_DROP_DELAY_S="${UNCHAIN_KV_SPOOL_LIVE_CAP_DROP_DELAY_S:-30}"
BULK_DECODE="${UNCHAIN_KV_BULK_DECODE:-0}"
RESTORE_AHEAD="${UNCHAIN_KV_RESTORE_AHEAD:-0}"
HOST_MIRROR_LAYERS="${UNCHAIN_KV_HOST_MIRROR_LAYERS:-32}"
SHAREGPT_OUTPUT_LEN="${UNCHAIN_KV_SHAREGPT_OUTPUT_LEN:-}"
EXTENT_ALLOC="${UNCHAIN_KV_EXTENT_ALLOC:-off}"
EXTENT_RESERVE_BLOCKS="${UNCHAIN_KV_EXTENT_RESERVE_BLOCKS:-0}"
AGING_REQUESTS="${UNCHAIN_KV_AGING_REQUESTS:-0}"
AGING_LENGTHS="${UNCHAIN_KV_AGING_LENGTHS:-8192 16384 32000}"
AGING_SEED="${UNCHAIN_KV_AGING_SEED:-20260723}"
MIXED_LENGTHS="${UNCHAIN_KV_MIXED_LENGTHS:-1024 4096 8192 16384 32000}"
RUN_ID_OVERRIDE="${UNCHAIN_KV_RUN_ID_OVERRIDE:-}"
DYNAMIC_RATES="${UNCHAIN_KV_DYNAMIC_RATES:-}"
DYNAMIC_MIN_SECONDS="${UNCHAIN_KV_DYNAMIC_MIN_SECONDS:-0}"
FAULT_ACTION="${UNCHAIN_KV_FAULT_ACTION:-}"

NIXL_PROXY_IP="${UNCHAIN_KV_NIXL_PROXY_IP:-172.16.0.50}"
NIXL_PREFILL_IP="${UNCHAIN_KV_NIXL_PREFILL_IP:-172.16.0.51}"
NIXL_DECODE_IP="${UNCHAIN_KV_NIXL_DECODE_IP:-172.16.0.52}"
GATEWAY_IP="${UNCHAIN_KV_GATEWAY_IP:-172.16.0.1}"

DMON_PID=""
RESOURCE_MONITOR_PID=""
LIVE_CAP_PID=""
FAULT_PID=""

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

cleanup_all() {
  cleanup_one kvp-proxy kvp-proxy-h
  cleanup_one kvp-prefill kvp-pref-h
  cleanup_one kvp-decode kvp-dec-h
  cleanup_one kvp-moon-proxy moon-proxy-h
  cleanup_one kvp-moon-prefill moon-pref-h
  cleanup_one kvp-moon-decode moon-dec-h
  cleanup_one kvp-nixl-proxy nixl-proxy-h
  cleanup_one kvp-nixl-prefill nixl-pref-h
  cleanup_one kvp-nixl-decode nixl-dec-h
}

cleanup() {
  stop_resource_monitor
  if [[ -n "${LIVE_CAP_PID}" ]]; then
    kill "${LIVE_CAP_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${DMON_PID}" ]]; then
    kill "${DMON_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${FAULT_PID}" ]]; then
    kill "${FAULT_PID}" >/dev/null 2>&1 || true
  fi
  cleanup_all
}

stop_resource_monitor() {
  if [[ -n "${RESOURCE_MONITOR_PID}" ]]; then
    kill "${RESOURCE_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${RESOURCE_MONITOR_PID}" >/dev/null 2>&1 || true
    RESOURCE_MONITOR_PID=""
  fi
}

start_resource_monitor() {
  local output="$1"
  shift
  local interval="${UNCHAIN_KV_RESOURCE_MONITOR_INTERVAL_S:-5}"
  (
    echo "epoch,container,rss_kib,cpu_percent"
    while true; do
      local now name stats
      now="$(date +%s)"
      for name in "$@"; do
        stats="$(docker exec "${name}" sh -c \
          "ps -e -o rss=,pcpu= | awk '{rss += \$1; cpu += \$2} END {print rss + 0 \",\" cpu + 0}'" \
          2>/dev/null || true)"
        [[ -n "${stats}" ]] && echo "${now},${name},${stats}"
      done
      sleep "${interval}"
    done
  ) >"${output}" &
  RESOURCE_MONITOR_PID="$!"
}

apply_rate() {
  local ns="$1"
  ip netns exec "${ns}" tc qdisc replace dev eth0 root handle 1: htb default 10
  ip netns exec "${ns}" tc class replace dev eth0 parent 1: classid 1:10 htb rate "${RATE}" ceil "${RATE}"
}

change_rate() {
  local ns="$1"
  ip netns exec "${ns}" tc class replace dev eth0 parent 1: \
    classid 1:10 htb rate "${RATE}" ceil "${RATE}"
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

start_nixl_container() {
  local name="$1"
  local gpu="$2"
  local ip_addr="$3"
  local side_port="$4"
  local host_if="$5"
  local peer_if="$6"
  local gpu_args=()
  if [[ "${gpu}" != "none" ]]; then
    gpu_args=(--gpus "device=${gpu}")
  fi
  docker run -d --name "${name}" \
    --hostname "${name}" \
    "${gpu_args[@]}" \
    --network none \
    -v "${ROOT}:${ROOT}" \
    -v /models:/models:ro \
    -w "${ROOT}" \
    -e PYTHONSAFEPATH=1 \
    -e PYTHONPATH="${ROOT}/src" \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e GLOO_SOCKET_IFNAME=eth0 \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -e VLLM_HOST_IP="${ip_addr}" \
    -e VLLM_NIXL_SIDE_CHANNEL_HOST="${ip_addr}" \
    -e VLLM_NIXL_SIDE_CHANNEL_PORT="${side_port}" \
    -e UCX_TLS="${UNCHAIN_KV_NIXL_UCX_TLS:-tcp,cuda_copy}" \
    -e UCX_NET_DEVICES=eth0 \
    -e UCX_SOCKADDR_TLS_PRIORITY=tcp \
    -e UCX_LOG_LEVEL="${UNCHAIN_KV_NIXL_UCX_LOG_LEVEL:-warn}" \
    "${IMAGE}" sleep infinity >/dev/null
  attach_ovs "${name}" "${host_if}" "${peer_if}" "${ip_addr}"
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
  return 1
}

start_nixl_vllm() {
  local name="$1"
  local role="$2"
  local host="$3"
  local port="$4"
  local log="$5"
  local vllm_args="$6"
  local config
  config="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"${role}\",\"kv_load_failure_policy\":\"fail\"}"
  docker exec -d \
    -e UNCHAIN_KV_NORMALIZE_RELEASE="${UNCHAIN_KV_NORMALIZE_RELEASE:-0}" \
    "${name}" bash -lc "
    python3 -m unchain_kv.patch_vllm \$(python3 -c 'from pathlib import Path; import vllm; print(Path(vllm.__file__).resolve().parents[1])') &&
    vllm serve ${MODEL} --host ${host} --port ${port} \\
      --kv-transfer-config '${config}' \\
      ${vllm_args} > ${log} 2>&1
  "
}

launch_nixl() {
  local run_root="$1"
  local vllm_args="$2"
  local prefill_vllm_args="${UNCHAIN_KV_PREFILL_VLLM_ARGS:-${vllm_args}}"
  local decode_vllm_args="${UNCHAIN_KV_DECODE_VLLM_ARGS:-${vllm_args}}"
  mkdir -p "${run_root}/logs"
  start_nixl_container kvp-nixl-decode 0 "${NIXL_DECODE_IP}" 5601 nixl-dec-h nixl-dec-c
  start_nixl_container kvp-nixl-prefill 1 "${NIXL_PREFILL_IP}" 5600 nixl-pref-h nixl-pref-c
  start_nixl_container kvp-nixl-proxy none "${NIXL_PROXY_IP}" 5699 nixl-proxy-h nixl-proxy-c
  for name in kvp-nixl-decode kvp-nixl-prefill; do
    echo "=== ${name}"
    docker exec "${name}" nvidia-smi -L
    docker exec "${name}" python3 -c \
      'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1'
  done >"${run_root}/container-gpu-preflight.txt" 2>&1
  start_nixl_vllm kvp-nixl-decode kv_consumer "${NIXL_DECODE_IP}" 18081 "${run_root}/logs/decode.docker.log" "${decode_vllm_args}"
  start_nixl_vllm kvp-nixl-prefill kv_producer "${NIXL_PREFILL_IP}" 18080 "${run_root}/logs/prefill.docker.log" "${prefill_vllm_args}"
  wait_ready "${NIXL_DECODE_IP}" 18081
  wait_ready "${NIXL_PREFILL_IP}" 18080
  docker exec -d kvp-nixl-proxy bash -lc "
    python3 scripts/nixl_proxy.py \\
      --listen ${NIXL_PROXY_IP}:18082 \\
      --prefill-url http://${NIXL_PREFILL_IP}:18080 \\
      --decode-url http://${NIXL_DECODE_IP}:18081 \\
      --timeout-s 900 > ${run_root}/logs/proxy.docker.log 2>&1
  "
  for _ in $(seq 1 60); do
    if curl -fsS "http://${NIXL_PROXY_IP}:18082/healthcheck" >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  echo "NIXL proxy not ready" >&2
  return 1
}

apply_canonical_profile() {
  local cell="$1"
  connector=native
  codec=""
  top16=0
  native_decode=0
  early=0
  writeback=0
  strict=0
  bulk_decode=0
  restore_ahead=0
  request_spool_bytes=0
  request_spool_auto=0
  host_mirror_layers=32
  extent_alloc=off
  extent_reserve_blocks=0
  gpu_pack_layers=0
  gpu_pack_bytes=0
  gpu_pack_strict=0
  block_runs=1
  payload_ready=0
  codec_min_blocks=0

  case "${cell}" in
    MC) connector=mooncake; return ;;
    NX) connector=nixl; return ;;
    R0) block_runs=0; return ;;
    R1) return ;;
  esac
  [[ "${cell}" =~ ^(M1|M12|M123|M1234-F|M1234-A)$ ]] || return 2

  codec=splitzip_bf16
  top16=1
  native_decode=1
  early=1
  gpu_pack_layers=1
  gpu_pack_bytes="${CANONICAL_GPU_PACK_BYTES}"
  gpu_pack_strict=1
  if [[ "${cell}" =~ ^(M12|M123|M1234-F|M1234-A)$ ]]; then
    writeback=1
    strict=1
    payload_ready=1
  fi
  if [[ "${cell}" =~ ^(M123|M1234-F|M1234-A)$ ]]; then
    extent_alloc=prefer
    extent_reserve_blocks="${EXTENT_RESERVE_BLOCKS}"
  fi
  case "${cell}" in
    M1234-F)
      codec_min_blocks="${CANONICAL_CODEC_MIN_BLOCKS}"
      request_spool_bytes="${FIXED_SPOOL_BYTES}"
      ;;
    M1234-A)
      codec_min_blocks="${CANONICAL_CODEC_MIN_BLOCKS}"
      request_spool_auto=1
      ;;
  esac
}

print_canonical_profile() {
  local connector codec top16 native_decode early writeback strict bulk_decode
  local restore_ahead request_spool_bytes request_spool_auto host_mirror_layers
  local extent_alloc extent_reserve_blocks gpu_pack_layers gpu_pack_bytes
  local gpu_pack_strict block_runs payload_ready codec_min_blocks
  apply_canonical_profile "$1"
  printf '%s\n' \
    "path=$1" "connector=${connector}" "codec=${codec}" "top16=${top16}" \
    "native_decode=${native_decode}" "early_stage=${early}" \
    "codec_writeback=${writeback}" "codec_writeback_strict=${strict}" \
    "block_runs=${block_runs}" "gpu_pack_layers=${gpu_pack_layers}" \
    "gpu_pack_bytes=${gpu_pack_bytes}" "gpu_pack_strict=${gpu_pack_strict}" \
    "payload_ready=${payload_ready}" "extent_alloc=${extent_alloc}" \
    "extent_reserve_blocks=${extent_reserve_blocks}" \
    "codec_min_blocks=${codec_min_blocks}" \
    "request_spool_bytes=${request_spool_bytes}" \
    "request_spool_auto=${request_spool_auto}" \
    "bulk_decode=${bulk_decode}" "restore_ahead=${restore_ahead}" \
    "host_mirror_layers=${host_mirror_layers}"
}

launch_native() {
  local cell="$1"
  local run_id="$2"
  local vllm_args="$3"
  local codec=""
  local top16=0
  local native_decode=0
  local early=0
  local writeback=0
  local strict=0
  local bulk_decode=0
  local restore_ahead=0
  local request_spool_bytes=0
  local request_spool_auto=0
  local host_mirror_layers=32
  local extent_alloc="${EXTENT_ALLOC}"
  local gpu_pack_layers="${GPU_PACK_LAYERS}"
  local gpu_pack_bytes="${GPU_PACK_BYTES}"
  local gpu_pack_strict="${GPU_PACK_STRICT}"
  local block_runs=1
  local payload_ready="${PAYLOAD_READY}"
  local codec_min_blocks="${CODEC_MIN_BLOCKS}"
  local extent_reserve_blocks="${EXTENT_RESERVE_BLOCKS}"
  if [[ "${cell}" =~ ^(R0|R1|M1|M12|M123|M1234-F|M1234-A)$ ]]; then
    local connector
    apply_canonical_profile "${cell}"
  elif [[ "${cell}" =~ ^(writeback|r0|r1|r2|r3|all_methods)$ ]]; then
    codec=splitzip_bf16
    top16=1
    native_decode=1
    early=1
    writeback=1
    strict=1
    request_spool_bytes="${REQUEST_SPOOL_BYTES}"
    bulk_decode="${BULK_DECODE}"
    restore_ahead="${RESTORE_AHEAD}"
    host_mirror_layers="${HOST_MIRROR_LAYERS}"
  fi
  if [[ "${cell}" =~ ^(r[0-3]|all_methods)$ ]]; then
    extent_alloc=prefer
    request_spool_bytes=0
    gpu_pack_layers=1
    gpu_pack_bytes=67108864
    gpu_pack_strict=1
  fi
  case "${cell}" in
    r1) request_spool_auto=observe ;;
    r2) request_spool_auto=1 ;;
    r3) request_spool_bytes="${FIXED_SPOOL_BYTES}" ;;
    all_methods)
      if [[ "${REQUEST_SPOOL_AUTO}" != "0" ]]; then
        request_spool_auto="${REQUEST_SPOOL_AUTO}"
      else
        request_spool_bytes="${FIXED_SPOOL_BYTES}"
      fi
      ;;
    raw_layer_wise|raw_no_runs)
      gpu_pack_layers=0
      gpu_pack_bytes=0
      gpu_pack_strict=0
      ;;
  esac
  [[ "${cell}" == "raw_no_runs" ]] && block_runs=0
  env \
    UNCHAIN_KV_REMOTE_ROOT="${ROOT}" \
    UNCHAIN_KV_IMAGE="${IMAGE}" \
    UNCHAIN_KV_MODEL="${MODEL}" \
    UNCHAIN_KV_RUN_ID="${run_id}" \
    UNCHAIN_KV_NET_RATE="${RATE}" \
    UNCHAIN_KV_VLLM_ARGS="${vllm_args}" \
    UNCHAIN_KV_LENGTHS=1024 \
    UNCHAIN_KV_SAMPLES=0 \
    UNCHAIN_KV_WARMUP=0 \
    UNCHAIN_KV_WARMUP_PER_LENGTH=0 \
    UNCHAIN_KV_DECODE_SLOTS="${CONCURRENCY}" \
    UNCHAIN_KV_PINNED_STAGING=1 \
    UNCHAIN_KV_HOST_MIRROR_BYTES="${HOST_MIRROR_BYTES}" \
    UNCHAIN_KV_REQUEST_SPOOL_BYTES="${request_spool_bytes}" \
    UNCHAIN_KV_REQUEST_SPOOL_AUTO="${request_spool_auto}" \
    UNCHAIN_KV_HOST_GUARD_BYTES="${HOST_GUARD_BYTES}" \
    UNCHAIN_KV_GPU_GUARD_BYTES="${GPU_GUARD_BYTES}" \
    UNCHAIN_KV_AUTO_SPOOL_HARD_BYTES="${AUTO_SPOOL_HARD_BYTES}" \
    UNCHAIN_KV_SPOOL_PRESSURE_RATIO="${SPOOL_PRESSURE_RATIO}" \
    UNCHAIN_KV_SPOOL_LIVE_CAP_FILE="${SPOOL_LIVE_CAP_FILE}" \
    UNCHAIN_KV_BULK_DECODE="${bulk_decode}" \
    UNCHAIN_KV_RESTORE_AHEAD="${restore_ahead}" \
    UNCHAIN_KV_HOST_MIRROR_LAYERS="${host_mirror_layers}" \
    UNCHAIN_KV_SEND_INFLIGHT=8 \
    UNCHAIN_KV_SEND_WORKERS=1 \
    UNCHAIN_KV_GPU_PACK_LAYERS="${gpu_pack_layers}" \
    UNCHAIN_KV_GPU_PACK_BYTES="${gpu_pack_bytes}" \
    UNCHAIN_KV_GPU_PACK_STRICT="${gpu_pack_strict}" \
    UNCHAIN_KV_BLOCK_RUNS="${block_runs}" \
    UNCHAIN_KV_PAYLOAD_READY="${payload_ready}" \
    UNCHAIN_KV_CODEC_GPU_BYTES="${CODEC_GPU_BYTES}" \
    UNCHAIN_KV_CODEC="${codec}" \
    UNCHAIN_KV_CODEC_MIN_BLOCKS="${codec_min_blocks}" \
    UNCHAIN_KV_SPLITZIP_TOP16="${top16}" \
    UNCHAIN_KV_SPLITZIP_NATIVE_DECODE="${native_decode}" \
    UNCHAIN_KV_SPLITZIP_CHUNKS=1 \
    UNCHAIN_KV_EARLY_STAGE="${early}" \
    UNCHAIN_KV_CODEC_WRITEBACK="${writeback}" \
    UNCHAIN_KV_CODEC_WRITEBACK_STRICT="${strict}" \
    UNCHAIN_KV_TRACE_ENABLED="${TRACE_ENABLED}" \
    UNCHAIN_KV_EXTENT_ALLOC="${extent_alloc}" \
    UNCHAIN_KV_EXTENT_RESERVE_BLOCKS="${extent_reserve_blocks}" \
    bash "${ROOT}/scripts/run_ovs_limited_matrix.sh" >/dev/null
}

launch_mooncake() {
  local run_id="$1"
  local vllm_args="$2"
  env \
    UNCHAIN_KV_REMOTE_ROOT="${ROOT}" \
    UNCHAIN_KV_IMAGE="${IMAGE}" \
    UNCHAIN_KV_MODEL="${MODEL}" \
    UNCHAIN_KV_RUN_ID="${run_id}" \
    UNCHAIN_KV_NET_RATE="${RATE}" \
    UNCHAIN_KV_VLLM_ARGS="${vllm_args}" \
    UNCHAIN_KV_LENGTHS=1024 \
    UNCHAIN_KV_SAMPLES=0 \
    UNCHAIN_KV_WARMUP_PER_LENGTH=0 \
    UNCHAIN_KV_BETWEEN_REQUEST_SLEEP_S=0 \
    bash "${ROOT}/scripts/run_ovs_limited_mooncake_matrix.sh" >/dev/null
}

tc_snapshot() {
  local output="$1"
  shift
  {
    echo "rate=${RATE}"
    for name in "$@"; do
      echo "=== ${name}"
      ip netns exec "${name}" tc -s qdisc show dev eth0
      ip netns exec "${name}" tc -s class show dev eth0
    done
  } >"${output}"
}

write_static_preflight() (
  local output="$1"
  exec >"${output}" 2>&1
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  [[ -n "${UNCHAIN_KV_FREEZE_ID:-}" ]] || { echo "missing freeze_id"; return 1; }
  echo "freeze_id=${UNCHAIN_KV_FREEZE_ID}"
  if git -C "${ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "git_head=$(git -C "${ROOT}" rev-parse HEAD)"
    git -C "${ROOT}" status --short
    echo "git_diff_sha256=$(git -C "${ROOT}" diff --binary -- scripts src native | sha256sum | cut -d' ' -f1)"
    echo "git_cached_diff_sha256=$(git -C "${ROOT}" diff --cached --binary -- scripts src native | sha256sum | cut -d' ' -f1)"
  else
    [[ -n "${UNCHAIN_KV_GIT_HEAD:-}" && -n "${UNCHAIN_KV_GIT_DIFF_SHA256:-}" && -n "${UNCHAIN_KV_GIT_CACHED_DIFF_SHA256:-}" ]] || {
      echo "missing controller-provided git provenance"
      return 1
    }
    echo "git_head=${UNCHAIN_KV_GIT_HEAD}"
    echo "git_diff_sha256=${UNCHAIN_KV_GIT_DIFF_SHA256}"
    echo "git_cached_diff_sha256=${UNCHAIN_KV_GIT_CACHED_DIFF_SHA256}"
    echo "git_status=controller-provided scope=scripts,src,native"
  fi
  echo "source_manifest_sha256=$(cd "${ROOT}" && find scripts src native -type f ! -path '*/__pycache__/*' ! -name '*.pyc' ! -path 'native/unchain_kv/build/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
  docker image inspect "${IMAGE}" --format 'image_id={{.Id}} created={{.Created}}' || return
  sha256sum \
    "${ROOT}/scripts/run_sharegpt_throughput_matrix.sh" \
    "${ROOT}/scripts/inject_fault.sh" \
    "${ROOT}/src/unchain_kv/vllm_connector.py" \
    "${ROOT}/src/unchain_kv/patch_vllm.py" \
    "${ROOT}/native/unchain_kv/build/libunchain_kv_tcp.so" \
    "${ROOT}/native/unchain_kv/build/libunchain_kv_splitzip_cuda.so" \
    "${MODEL}/config.json" "${MODEL}/tokenizer.json" \
    "${MODEL}/model.safetensors.index.json" || return
  if [[ "${UNCHAIN_KV_HASH_MODEL_WEIGHTS:-0}" == "1" ]]; then
    echo "model_weight_shards_expected=$(python3 -c 'import json,sys; print(len(set(json.load(open(sys.argv[1], encoding="utf-8"))["weight_map"].values())))' "${MODEL}/model.safetensors.index.json")"
    find "${MODEL}" -maxdepth 1 -type f -name '*.safetensors' -print0 \
      | sort -z | xargs -0 -r sha256sum || return
  fi
  if [[ "${BENCH_DATASET}" == "sharegpt" ]]; then
    sha256sum "${DATASET}" || return
  elif [[ "${BENCH_DATASET}" == "context" || "${BENCH_DATASET}" == "mixed-context" ]]; then
    sha256sum "${PROMPT_DIR}/manifest.json" || return
  elif [[ "${BENCH_DATASET}" == "prompt-manifest" ]]; then
    sha256sum "${PROMPT_MANIFEST}" || return
  fi
  local gpu_inventory available_kib
  gpu_inventory="$(nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader)" || return
  echo "${gpu_inventory}"
  [[ "$(grep -c '^' <<<"${gpu_inventory}")" == "2" ]] || return 1
  [[ "$(grep -c 'NVIDIA A40' <<<"${gpu_inventory}")" == "2" ]] || return 1
  nvidia-smi topo -m || return
  echo "ntp_synchronized=$(timedatectl show -p NTPSynchronized --value)"
  [[ "$(timedatectl show -p NTPSynchronized --value)" == "yes" ]] || return 1
  df -h "${ROOT}" || return
  available_kib="$(df -Pk "${ROOT}" | awk 'NR == 2 {print $4}')"
  echo "disk_available_kib=${available_kib} minimum_kib=${MIN_DISK_KIB}"
  (( available_kib >= MIN_DISK_KIB )) || return 1
  ovs-vsctl br-exists "${BRIDGE}" || return
  local bridge_link
  bridge_link="$(ip -o link show dev "${BRIDGE}")" || return
  echo "${bridge_link}"
  grep -q " mtu ${MTU} " <<<"${bridge_link}" || return 1
  if ! ss -ltnH | awk '$4 ~ /:1808[0-2]$/ {exit 1}'; then
    echo "benchmark port already in use"
    return 1
  fi
)

write_preflight() {
  local output="$1"
  shift
  {
    date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
    echo "freeze_id=${UNCHAIN_KV_FREEZE_ID:-}"
    echo "root=${ROOT}"
    echo "image=${IMAGE}"
    docker image inspect "${IMAGE}" --format 'image_id={{.Id}} created={{.Created}}'
    sha256sum \
      "${ROOT}/scripts/run_sharegpt_throughput_matrix.sh" \
      "${ROOT}/scripts/inject_fault.sh" \
      "${ROOT}/src/unchain_kv/vllm_connector.py" \
      "${ROOT}/src/unchain_kv/patch_vllm.py" \
      "${ROOT}/native/unchain_kv/build/libunchain_kv_tcp.so" \
      "${ROOT}/native/unchain_kv/build/libunchain_kv_splitzip_cuda.so" \
      "${MODEL}/config.json"
    nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
      --format=csv,noheader
    df -h "${ROOT}"
    ovs-vsctl br-exists "${BRIDGE}"
    local name link route tc_state
    for name in "$@"; do
      link="$(ip netns exec "${name}" ip -o link show dev eth0)"
      tc_state="$(ip netns exec "${name}" tc class show dev eth0)"
      route="$(ip netns exec "${name}" ip route show default)"
      echo "${link}"
      echo "${tc_state}"
      echo "${route}"
      grep -q " mtu ${MTU} " <<<"${link}" || return 1
      grep -qi "rate ${RATE}" <<<"${tc_state}" || return 1
      grep -q "via ${GATEWAY_IP} " <<<"${route}" || return 1
    done
  } >"${output}"
}

write_manifest() {
  local output="$1"
  local cell="$2"
  local round="$3"
  local vllm_args="$4"
  local prefill_vllm_args="${UNCHAIN_KV_PREFILL_VLLM_ARGS:-${vllm_args}}"
  local decode_vllm_args="${UNCHAIN_KV_DECODE_VLLM_ARGS:-${vllm_args}}"
  local request_spool_bytes="${REQUEST_SPOOL_BYTES}"
  local request_spool_auto="${REQUEST_SPOOL_AUTO}"
  local bulk_decode="${BULK_DECODE}"
  local restore_ahead="${RESTORE_AHEAD}"
  local host_mirror_layers="${HOST_MIRROR_LAYERS}"
  local extent_alloc="${EXTENT_ALLOC}"
  local gpu_pack_layers="${GPU_PACK_LAYERS}"
  local gpu_pack_bytes="${GPU_PACK_BYTES}"
  local gpu_pack_strict="${GPU_PACK_STRICT}"
  local codec_min_blocks="${CODEC_MIN_BLOCKS}"
  local block_runs=1
  local connector=native
  local codec=""
  local top16=0
  local native_decode=0
  local early=0
  local writeback=0
  local strict=0
  local payload_ready="${PAYLOAD_READY}"
  local extent_reserve_blocks="${EXTENT_RESERVE_BLOCKS}"
  if [[ "${cell}" =~ ^(R0|R1|M1|M12|M123|M1234-F|M1234-A|MC|NX)$ ]]; then
    apply_canonical_profile "${cell}"
  elif [[ "${cell}" =~ ^(raw|raw_layer_wise|raw_no_runs|layer_wise)$ ]]; then
    request_spool_bytes=0
    bulk_decode=0
    restore_ahead=0
    host_mirror_layers=32
    codec_min_blocks=0
  fi
  case "${cell}" in
    r0) request_spool_bytes=0; request_spool_auto=0 ;;
    r1) request_spool_bytes=0; request_spool_auto=observe ;;
    r2) request_spool_bytes=0; request_spool_auto=1 ;;
    r3) request_spool_bytes="${FIXED_SPOOL_BYTES}"; request_spool_auto=0 ;;
    all_methods)
      if [[ "${REQUEST_SPOOL_AUTO}" != "0" ]]; then
        request_spool_bytes=0
        request_spool_auto="${REQUEST_SPOOL_AUTO}"
      else
        request_spool_bytes="${FIXED_SPOOL_BYTES}"
        request_spool_auto=0
      fi
      ;;
    raw_layer_wise|raw_no_runs)
      gpu_pack_layers=0
      gpu_pack_bytes=0
      gpu_pack_strict=0
      ;;
  esac
  [[ "${cell}" == "raw_no_runs" ]] && block_runs=0
  if [[ "${cell}" =~ ^(r[0-3]|all_methods)$ ]]; then
    codec=splitzip_bf16
    top16=1
    native_decode=1
    early=1
    writeback=1
    strict=1
    extent_alloc=prefer
    gpu_pack_layers=1
    gpu_pack_bytes=67108864
    gpu_pack_strict=1
  fi
  {
    echo "cell=${cell}"
    echo "round=${round}"
    echo "image=${IMAGE}"
    docker image inspect "${IMAGE}" --format 'image_id={{.Id}} created={{.Created}}'
    echo "model=${MODEL}"
    if [[ "${BENCH_DATASET}" == "context" || "${BENCH_DATASET}" == "mixed-context" ]]; then
      echo "prompt_dir=${PROMPT_DIR}"
      echo "prompt_manifest_sha256=$(sha256sum "${PROMPT_DIR}/manifest.json" | cut -d' ' -f1)"
    elif [[ "${BENCH_DATASET}" == "prompt-manifest" ]]; then
      echo "prompt_manifest=${PROMPT_MANIFEST}"
      echo "prompt_manifest_sha256=$(sha256sum "${PROMPT_MANIFEST}" | cut -d' ' -f1)"
    elif [[ "${BENCH_DATASET}" == "sharegpt" ]]; then
      echo "dataset=${DATASET}"
      sha256sum "${DATASET}"
    else
      echo "dataset=synthetic-random"
    fi
    echo "seed=${SEED}"
    echo "prompts=${PROMPTS}"
    echo "warmups=${WARMUPS}"
    echo "concurrency=${CONCURRENCY}"
    echo "rate=${RATE}"
    echo "bench_dataset=${BENCH_DATASET}"
    echo "request_rate=${REQUEST_RATE}"
    echo "bench_max_duration_s=${BENCH_MAX_DURATION_S}"
    echo "burstiness=${BURSTINESS}"
    echo "trace_enabled=${TRACE_ENABLED}"
    echo "resource_monitor=${UNCHAIN_KV_RESOURCE_MONITOR:-0}"
    echo "resource_monitor_interval_s=${UNCHAIN_KV_RESOURCE_MONITOR_INTERVAL_S:-5}"
    echo "fault_action=${FAULT_ACTION}"
    echo "fault_delay_s=${UNCHAIN_KV_FAULT_DELAY_S:-5}"
    echo "fault_hold_s=${UNCHAIN_KV_FAULT_HOLD_S:-1}"
    echo "fault_repeat=${UNCHAIN_KV_FAULT_REPEAT:-1}"
    echo "fault_interval_s=${UNCHAIN_KV_FAULT_INTERVAL_S:-1}"
    echo "fault_low_rate=${UNCHAIN_KV_FAULT_LOW_RATE:-2gbit}"
    echo "random_input_len=${RANDOM_INPUT_LEN}"
    echo "random_output_len=${RANDOM_OUTPUT_LEN}"
    echo "random_prefix_len=${RANDOM_PREFIX_LEN}"
    echo "context_sample_offset=${CONTEXT_SAMPLE_OFFSET}"
    echo "context_prompt_cycle=${CONTEXT_PROMPT_CYCLE}"
    echo "normalize_release=${UNCHAIN_KV_NORMALIZE_RELEASE:-0}"
    echo "host_mirror_bytes=${HOST_MIRROR_BYTES}"
    echo "connector=${connector}"
    echo "codec=${codec}"
    echo "top16=${top16}"
    echo "native_decode=${native_decode}"
    echo "early_stage=${early}"
    echo "codec_writeback=${writeback}"
    echo "codec_writeback_strict=${strict}"
    echo "request_spool_bytes=${request_spool_bytes}"
    echo "request_spool_auto=${request_spool_auto}"
    echo "host_guard_bytes=${HOST_GUARD_BYTES}"
    echo "gpu_guard_bytes=${GPU_GUARD_BYTES}"
    echo "auto_spool_hard_bytes=${AUTO_SPOOL_HARD_BYTES}"
    echo "spool_pressure_ratio=${SPOOL_PRESSURE_RATIO}"
    echo "spool_live_cap_file=${SPOOL_LIVE_CAP_FILE}"
    echo "spool_live_cap_drop_bytes=${SPOOL_LIVE_CAP_DROP_BYTES}"
    echo "spool_live_cap_drop_delay_s=${SPOOL_LIVE_CAP_DROP_DELAY_S}"
    echo "bulk_decode=${bulk_decode}"
    echo "restore_ahead=${restore_ahead}"
    echo "host_mirror_layers=${host_mirror_layers}"
    echo "sharegpt_output_len=${SHAREGPT_OUTPUT_LEN}"
    echo "request_spool_scope=writeback"
    echo "codec_gpu_bytes=${CODEC_GPU_BYTES}"
    echo "codec_min_blocks=${codec_min_blocks}"
    echo "prefix_fast_wait_s=${PREFIX_FAST_WAIT_S}"
    echo "extent_alloc=${extent_alloc}"
    echo "extent_reserve_blocks=${extent_reserve_blocks}"
    echo "gpu_pack_layers=${gpu_pack_layers}"
    echo "gpu_pack_bytes=${gpu_pack_bytes}"
    echo "gpu_pack_strict=${gpu_pack_strict}"
    echo "block_runs=${block_runs}"
    echo "payload_ready=${payload_ready}"
    echo "aging_requests=${AGING_REQUESTS}"
    echo "aging_lengths=${AGING_LENGTHS}"
    echo "aging_seed=${AGING_SEED}"
    echo "mixed_lengths=${MIXED_LENGTHS}"
    echo "vllm_args=${vllm_args}"
    echo "prefill_vllm_args=${prefill_vllm_args}"
    echo "decode_vllm_args=${decode_vllm_args}"
    echo "nixl_ucx_tls=${UNCHAIN_KV_NIXL_UCX_TLS:-tcp,cuda_copy}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  } >"${output}"
}

run_benchmark() {
  local cell="$1"
  local round="$2"
  local proxy_ip="$3"
  local run_root="$4"
  nvidia-smi dmon -s pucm -d 1 -o DT >"${run_root}/gpu-dmon.txt" &
  DMON_PID="$!"
  local status=0
  local bench_timeout=()
  if (( BENCH_MAX_DURATION_S > 0 )); then
    bench_timeout=(timeout --signal=TERM --kill-after=30 "${BENCH_MAX_DURATION_S}s")
  fi
  local dataset_args=()
  local sharegpt_args=()
  if [[ "${BENCH_DATASET}" == "context" ]]; then
    "${bench_timeout[@]}" docker run --rm --network host \
      -v "${ROOT}:${ROOT}" \
      -v /models:/models:ro \
      -w "${ROOT}" \
      "${IMAGE}" python3 scripts/measure_ttft.py \
        --url "http://${proxy_ip}:18082" \
        --model "${MODEL}" \
        --prompt-dir "${PROMPT_DIR}" \
        --prompt-length "${RANDOM_INPUT_LEN}" \
        --sample-offset "${CONTEXT_SAMPLE_OFFSET}" \
        --prompt-cycle "${CONTEXT_PROMPT_CYCLE}" \
        --output "${run_root}/context-requests.jsonl" \
        --bench-output "${run_root}/bench.json" \
        --runs "${PROMPTS}" \
        --warmup "${WARMUPS}" \
        --concurrency "${CONCURRENCY}" \
        --max-tokens "${RANDOM_OUTPUT_LEN}" \
        --temperature 0 \
        --timeout-s 900 \
        --ignore-eos \
        >"${run_root}/bench.stdout.log" 2>"${run_root}/bench.stderr.log" || status="$?"
  elif [[ "${BENCH_DATASET}" == "mixed-context" ]]; then
    local mixed_lengths=()
    read -r -a mixed_lengths <<<"${MIXED_LENGTHS}"
    docker run --rm --network host \
      -v "${ROOT}:${ROOT}" \
      -v /models:/models:ro \
      -w "${ROOT}" \
      -e PYTHONPATH="${ROOT}:${ROOT}/src" \
      "${IMAGE}" python3 scripts/measure_mixed_context.py \
        --url "http://${proxy_ip}:18082" \
        --model "${MODEL}" \
        --prompt-dir "${PROMPT_DIR}" \
        --lengths "${mixed_lengths[@]}" \
        --requests "${PROMPTS}" \
        --concurrency "${CONCURRENCY}" \
        --max-tokens "${RANDOM_OUTPUT_LEN}" \
        --seed "${SEED}" \
        --max-duration-s "${BENCH_MAX_DURATION_S}" \
        --output "${run_root}/context-requests.jsonl" \
        --bench-output "${run_root}/bench.json" \
        >"${run_root}/bench.stdout.log" 2>"${run_root}/bench.stderr.log" || status="$?"
  elif [[ "${BENCH_DATASET}" == "prompt-manifest" ]]; then
    docker run --rm --network host \
      -v "${ROOT}:${ROOT}" \
      -v /models:/models:ro \
      -v /datasets:/datasets:ro \
      -w "${ROOT}" \
      -e PYTHONPATH="${ROOT}:${ROOT}/src" \
      "${IMAGE}" python3 scripts/measure_mixed_context.py \
        --url "http://${proxy_ip}:18082" \
        --model "${MODEL}" \
        --manifest "${PROMPT_MANIFEST}" \
        --requests "${PROMPTS}" \
        --concurrency "${CONCURRENCY}" \
        --max-tokens "${RANDOM_OUTPUT_LEN}" \
        --seed "${SEED}" \
        --max-duration-s "${BENCH_MAX_DURATION_S}" \
        --output "${run_root}/context-requests.jsonl" \
        --bench-output "${run_root}/bench.json" \
        >"${run_root}/bench.stdout.log" 2>"${run_root}/bench.stderr.log" || status="$?"
  elif [[ "${BENCH_DATASET}" == "random" ]]; then
    dataset_args=(
      --dataset-name random
      --random-input-len "${RANDOM_INPUT_LEN}"
      --random-output-len "${RANDOM_OUTPUT_LEN}"
      --random-prefix-len "${RANDOM_PREFIX_LEN}"
      --random-range-ratio 0
    )
  else
    dataset_args=(--dataset-name sharegpt --dataset-path "${DATASET}")
    if [[ -n "${SHAREGPT_OUTPUT_LEN}" ]]; then
      sharegpt_args=(--sharegpt-output-len "${SHAREGPT_OUTPUT_LEN}")
    fi
  fi
  if [[ "${BENCH_DATASET}" != "context" && "${BENCH_DATASET}" != "mixed-context" && "${BENCH_DATASET}" != "prompt-manifest" ]]; then
    "${bench_timeout[@]}" docker run --rm --network host \
      -v "${ROOT}:${ROOT}" \
      -v /models:/models:ro \
      -v /datasets:/datasets:ro \
      -w "${ROOT}" \
      -e HF_HUB_OFFLINE=1 \
      -e TRANSFORMERS_OFFLINE=1 \
      "${IMAGE}" vllm bench serve \
        --backend openai \
        --base-url "http://${proxy_ip}:18082" \
        --endpoint /v1/completions \
        --model "${MODEL}" \
        --tokenizer "${MODEL}" \
        "${dataset_args[@]}" \
        "${sharegpt_args[@]}" \
        --num-prompts "${PROMPTS}" \
        --num-warmups "${WARMUPS}" \
        --request-rate "${REQUEST_RATE}" \
        --burstiness "${BURSTINESS}" \
        --max-concurrency "${CONCURRENCY}" \
        --seed "${SEED}" \
        --temperature 0 \
        --ignore-eos \
        --metric-percentiles 50,95,99 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --request-id-prefix "sharegpt-${cell}-r${round}-" \
        --metadata "path=${cell}" "round=${round}" "network=${RATE}" "concurrency=${CONCURRENCY}" \
        --save-result \
        --save-detailed \
        --result-dir "${run_root}" \
        --result-filename bench.json \
        --disable-tqdm \
        --ready-check-timeout-sec 60 \
        >"${run_root}/bench.stdout.log" 2>"${run_root}/bench.stderr.log" || status="$?"
  fi
  kill "${DMON_PID}" >/dev/null 2>&1 || true
  wait "${DMON_PID}" >/dev/null 2>&1 || true
  DMON_PID=""
  return "${status}"
}

bench_complete() {
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(not (int(d.get("completed", -1)) == int(sys.argv[2]) and int(d.get("failed", -1)) == 0))' "$1" "$2"
}

run_aging() {
  local proxy_ip="$1"
  local run_root="$2"
  (( AGING_REQUESTS > 0 )) || return 0
  local lengths=()
  read -r -a lengths <<<"${AGING_LENGTHS}"
  python3 "${ROOT}/scripts/measure_mixed_context_soak.py" \
    --url "http://${proxy_ip}:18082/v1/completions" \
    --model "${MODEL}" \
    --prompt-dir "${PROMPT_DIR}" \
    --lengths "${lengths[@]}" \
    --requests 0 \
    --warmup "${AGING_REQUESTS}" \
    --concurrency "${CONCURRENCY}" \
    --max-tokens 1 \
    --fixed-batches \
    --seed "${AGING_SEED}" \
    --output "${run_root}/aging-requests.jsonl" \
    --summary "${run_root}/aging-summary.json" \
    --timeout 900
}

run_dynamic_benchmarks() {
  local cell="$1"
  local round="$2"
  local proxy_ip="$3"
  local run_root="$4"
  shift 4
  local containers=("$@")
  local rates=()
  read -r -a rates <<<"${DYNAMIC_RATES}"
  local index=0 status=0
  for RATE in "${rates[@]}"; do
    index=$((index + 1))
    local phase_root="${run_root}/phase-${index}-${RATE}"
    mkdir -p "${phase_root}"
    cp "${run_root}/preflight-static.txt" "${phase_root}/preflight-static.txt"
    cp "${run_root}/preflight.txt" "${phase_root}/preflight.txt"
    cp "${run_root}/container-gpu-preflight.txt" \
      "${phase_root}/container-gpu-preflight.txt"
    local name
    for name in "${containers[@]}"; do
      change_rate "${name}"
    done
    write_manifest "${phase_root}/manifest.txt" "${cell}" \
      "${round}-phase-${index}" "${vllm_args}"
    tc_snapshot "${phase_root}/tc-before.txt" "${containers[@]}"
    if [[ "${UNCHAIN_KV_RESOURCE_MONITOR:-0}" == "1" ]]; then
      start_resource_monitor "${phase_root}/container-rss.csv" "${containers[@]}"
    fi
    local phase_status=0
    run_benchmark "${cell}" "${round}-phase-${index}" \
      "${proxy_ip}" "${phase_root}" || phase_status="$?"
    stop_resource_monitor
    tc_snapshot "${phase_root}/tc-after.txt" "${containers[@]}"
    if (( phase_status == 0 && DYNAMIC_MIN_SECONDS > 0 )); then
      python3 -c \
        'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(float(d["duration"]) < float(sys.argv[2]))' \
        "${phase_root}/bench.json" "${DYNAMIC_MIN_SECONDS}" || phase_status=1
    fi
    (( phase_status == 0 )) || status=1
  done
  return "${status}"
}

run_cell() {
  local cell="$1"
  local round="$2"
  local run_id="${RUN_ID_OVERRIDE:-${MATRIX_ID}-r${round}-${cell}}"
  local run_root="${ROOT}/runs/${run_id}"
  if [[ -s "${run_root}/bench.json" && "${RESUME}" == "1" ]] \
    && bench_complete "${run_root}/bench.json" "${PROMPTS}"; then
    echo "skip completed ${run_id}"
    return
  fi
  if [[ -e "${run_root}" ]]; then
    echo "refusing to overwrite existing ${run_root}" >&2
    return 2
  fi

  local vllm_args="${UNCHAIN_KV_VLLM_ARGS:---max-model-len 32768 --max-num-batched-tokens 4096 --max-num-seqs ${CONCURRENCY} --gpu-memory-utilization 0.80 --dtype bfloat16 --enable-prefix-caching --enable-chunked-prefill --enforce-eager}"
  cleanup_all
  sleep 2
  mkdir -p "${run_root}"
  if ! write_manifest "${run_root}/manifest.txt" "${cell}" "${round}" "${vllm_args}"; then
    echo "manifest preflight failed: ${run_root}/manifest.txt" >&2
    return 2
  fi
  if ! write_static_preflight "${run_root}/preflight-static.txt"; then
    echo "static preflight failed: ${run_root}/preflight-static.txt" >&2
    return 2
  fi
  local proxy_ip
  local containers=()
  local launch_status=0
  case "${cell}" in
    R0|R1|M1|M12|M123|M1234-F|M1234-A|raw|raw_layer_wise|raw_no_runs|layer_wise|writeback|r0|r1|r2|r3|all_methods)
      launch_native "${cell}" "${run_id}" "${vllm_args}" || launch_status="$?"
      proxy_ip=172.16.0.30
      containers=(kvp-proxy kvp-prefill kvp-decode)
      ;;
    MC|mooncake)
      launch_mooncake "${run_id}" "${vllm_args}" || launch_status="$?"
      proxy_ip=172.16.0.40
      containers=(kvp-moon-proxy kvp-moon-prefill kvp-moon-decode)
      ;;
    NX|nixl)
      mkdir -p "${run_root}"
      launch_nixl "${run_root}" "${vllm_args}" || launch_status="$?"
      proxy_ip="${NIXL_PROXY_IP}"
      containers=(kvp-nixl-proxy kvp-nixl-prefill kvp-nixl-decode)
      ;;
    *)
      echo "unknown cell: ${cell}" >&2
      return 2
      ;;
  esac

  if (( launch_status != 0 )); then
    cleanup_all
    return "${launch_status}"
  fi

  if ! write_preflight "${run_root}/preflight.txt" "${containers[@]}"; then
    cleanup_all
    return 2
  fi
  run_aging "${proxy_ip}" "${run_root}"
  if [[ -n "${DYNAMIC_RATES}" ]]; then
    local dynamic_status=0
    run_dynamic_benchmarks "${cell}" "${round}" "${proxy_ip}" \
      "${run_root}" "${containers[@]}" || dynamic_status="$?"
    cleanup_all
    sleep 2
    if [[ "${TRACE_ENABLED}" != "0" ]]; then
      cp "${run_root}/trace/prefill.jsonl" "${run_root}/kv_producer-trace.jsonl"
      cp "${run_root}/trace/decode.jsonl" "${run_root}/kv_consumer-trace.jsonl"
    fi
    return "${dynamic_status}"
  fi
  tc_snapshot "${run_root}/tc-before.txt" "${containers[@]}"
  echo "run ${run_id}"
  if [[ "${SPOOL_LIVE_CAP_DROP_BYTES}" != 0 ]]; then
    [[ -n "${SPOOL_LIVE_CAP_FILE}" && -w "${SPOOL_LIVE_CAP_FILE}" ]] || {
      echo "live-cap drop requires a writable cap file" >&2
      return 2
    }
    (
      sleep "${SPOOL_LIVE_CAP_DROP_DELAY_S}"
      printf '%s\n' "${SPOOL_LIVE_CAP_DROP_BYTES}" >"${SPOOL_LIVE_CAP_FILE}"
    ) &
    LIVE_CAP_PID="$!"
  fi
  if [[ "${UNCHAIN_KV_RESOURCE_MONITOR:-0}" == "1" ]]; then
    start_resource_monitor "${run_root}/container-rss.csv" "${containers[@]}"
  fi
  if [[ -n "${FAULT_ACTION}" ]]; then
    bash "${ROOT}/scripts/inject_fault.sh" "${FAULT_ACTION}" "${run_root}" \
      >"${run_root}/fault.stdout.log" 2>"${run_root}/fault.stderr.log" &
    FAULT_PID="$!"
  fi
  local benchmark_status=0
  run_benchmark "${cell}" "${round}" "${proxy_ip}" "${run_root}" || benchmark_status="$?"
  if [[ "${benchmark_status}" == "0" ]] \
    && ! bench_complete "${run_root}/bench.json" "${PROMPTS}"; then
    echo "benchmark incomplete: expected=${PROMPTS} ${run_root}/bench.json" >&2
    benchmark_status=1
  fi
  if [[ -n "${FAULT_PID}" ]]; then
    wait "${FAULT_PID}" || benchmark_status=1
    FAULT_PID=""
  fi
  stop_resource_monitor
  if [[ -n "${LIVE_CAP_PID}" ]]; then
    wait "${LIVE_CAP_PID}" >/dev/null 2>&1 || true
    LIVE_CAP_PID=""
  fi
  tc_snapshot "${run_root}/tc-after.txt" "${containers[@]}"
  cleanup_all
  sleep 2
  if [[ "${TRACE_ENABLED}" != "0" && "${cell}" =~ ^(R0|R1|M1|M12|M123|M1234-F|M1234-A|raw|raw_layer_wise|raw_no_runs|layer_wise|writeback|r0|r1|r2|r3|all_methods)$ ]]; then
    cp "${run_root}/trace/prefill.jsonl" "${run_root}/kv_producer-trace.jsonl"
    cp "${run_root}/trace/decode.jsonl" "${run_root}/kv_consumer-trace.jsonl"
  fi
  (( benchmark_status == 0 ))
}

run_order() {
  local round="$1"
  shift
  for cell in "$@"; do
    run_cell "${cell}" "${round}"
  done
}

print_plan() {
  echo "matrix_id=${MATRIX_ID}"
  echo "dataset=${BENCH_DATASET} request_rate=${REQUEST_RATE}"
  echo "correctness: mooncake nixl raw writeback; c1; 16 prompts"
  echo "pilot:       mooncake nixl raw writeback; c4; 64 prompts"
  echo "formal r1:   mooncake nixl raw writeback"
  echo "formal r2:   nixl raw writeback mooncake"
  echo "formal r3:   raw writeback mooncake nixl"
  echo "formal r4:   writeback mooncake nixl raw"
  echo "adaptive r1: r0 r1 r2 r3"
  echo "adaptive r2: r1 r2 r3 r0"
  echo "adaptive r3: r2 r3 r0 r1"
  echo "adaptive r4: r3 r0 r1 r2"
}

main() {
  local mode="${1:-plan}"
  if [[ "${mode}" == "profile" ]]; then
    [[ $# == 2 ]] || { echo "usage: $0 profile PATH" >&2; return 2; }
    print_canonical_profile "$2"
    return
  fi
  if [[ "${mode}" == "plan" ]]; then
    print_plan
    return
  fi
  need_root
  if [[ "${mode}" == "static-preflight" ]]; then
    [[ $# == 2 ]] || { echo "usage: $0 static-preflight OUTPUT" >&2; return 2; }
    mkdir -p "$(dirname "$2")"
    write_static_preflight "$2"
    return
  fi
  if [[ "${BENCH_DATASET}" == "sharegpt" ]]; then
    [[ -r "${DATASET}" ]] || { echo "missing dataset: ${DATASET}" >&2; return 2; }
  elif [[ "${BENCH_DATASET}" == "prompt-manifest" ]]; then
    [[ -r "${PROMPT_MANIFEST}" ]] || { echo "missing prompt manifest: ${PROMPT_MANIFEST}" >&2; return 2; }
  elif [[ "${BENCH_DATASET}" == "context" || "${BENCH_DATASET}" == "mixed-context" ]]; then
    [[ -r "${PROMPT_DIR}/manifest.json" ]] || { echo "missing prompts: ${PROMPT_DIR}" >&2; return 2; }
    if [[ "${BENCH_DATASET}" == "mixed-context" ]]; then
      local mixed_length prompt_index
      for mixed_length in ${MIXED_LENGTHS}; do
        for ((prompt_index = 0; prompt_index < 10; prompt_index++)); do
          [[ -r "${PROMPT_DIR}/prompt-${mixed_length}-${prompt_index}.txt" ]] || {
            echo "missing mixed context prompt: length=${mixed_length} index=${prompt_index}" >&2
            return 2
          }
        done
      done
    else
      [[ "${CONTEXT_PROMPT_CYCLE}" =~ ^[0-9]+$ ]] || { echo "invalid context prompt cycle: ${CONTEXT_PROMPT_CYCLE}" >&2; return 2; }
    fi
    if [[ "${BENCH_DATASET}" == "context" ]] && (( CONTEXT_PROMPT_CYCLE > 0 )); then
      local prompt_index
      for ((prompt_index = 0; prompt_index < CONTEXT_PROMPT_CYCLE; prompt_index++)); do
        [[ -r "${PROMPT_DIR}/prompt-${RANDOM_INPUT_LEN}-$((CONTEXT_SAMPLE_OFFSET + prompt_index)).txt" ]] || {
          echo "missing context prompt: length=${RANDOM_INPUT_LEN} index=$((CONTEXT_SAMPLE_OFFSET + prompt_index))" >&2
          return 2
        }
      done
    fi
  fi
  trap cleanup EXIT INT TERM
  case "${mode}" in
    correctness)
      PROMPTS="${UNCHAIN_KV_NUM_PROMPTS:-16}"
      WARMUPS="${UNCHAIN_KV_NUM_WARMUPS:-0}"
      CONCURRENCY="${UNCHAIN_KV_CONCURRENCY:-1}"
      run_order correctness mooncake nixl raw writeback
      ;;
    pilot)
      PROMPTS="${UNCHAIN_KV_NUM_PROMPTS:-64}"
      WARMUPS="${UNCHAIN_KV_NUM_WARMUPS:-16}"
      CONCURRENCY="${UNCHAIN_KV_CONCURRENCY:-4}"
      run_order pilot mooncake nixl raw writeback
      ;;
    formal)
      PROMPTS="${UNCHAIN_KV_NUM_PROMPTS:-256}"
      WARMUPS="${UNCHAIN_KV_NUM_WARMUPS:-16}"
      CONCURRENCY="${UNCHAIN_KV_CONCURRENCY:-4}"
      run_order 1 mooncake nixl raw writeback
      run_order 2 nixl raw writeback mooncake
      run_order 3 raw writeback mooncake nixl
      run_order 4 writeback mooncake nixl raw
      ;;
    adaptive)
      PROMPTS="${UNCHAIN_KV_NUM_PROMPTS:-16}"
      WARMUPS="${UNCHAIN_KV_NUM_WARMUPS:-2}"
      CONCURRENCY="${UNCHAIN_KV_CONCURRENCY:-2}"
      run_order 1 r0 r1 r2 r3
      run_order 2 r1 r2 r3 r0
      run_order 3 r2 r3 r0 r1
      run_order 4 r3 r0 r1 r2
      ;;
    cell)
      [[ $# == 3 ]] || { echo "usage: $0 cell PATH ROUND" >&2; return 2; }
      PROMPTS="${UNCHAIN_KV_NUM_PROMPTS:-64}"
      WARMUPS="${UNCHAIN_KV_NUM_WARMUPS:-16}"
      CONCURRENCY="${UNCHAIN_KV_CONCURRENCY:-4}"
      run_cell "$2" "$3"
      ;;
    *)
      echo "usage: $0 {plan|profile PATH|static-preflight OUTPUT|correctness|pilot|formal|adaptive|cell PATH ROUND}" >&2
      return 2
      ;;
  esac
}

main "$@"
