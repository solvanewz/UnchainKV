#!/usr/bin/env bash
set -euo pipefail

ROOT="${UNCHAIN_KV_REMOTE_ROOT:-/workspace/unchain-kv}"
BRIDGE="${UNCHAIN_KV_OVS_BRIDGE:-exp-br}"
RATE="${UNCHAIN_KV_NET_RATE:-10gbit}"
RUN_ID="${UNCHAIN_KV_RUN_ID:-bandwidth-replay-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${ROOT}/runs/${RUN_ID}"
TX_NS=kvp-bwtx
RX_NS=kvp-bwrx
TX_IP=172.16.0.61
RX_IP=172.16.0.62

cleanup_one() {
  local ns="$1"
  local host_if="$2"
  ovs-vsctl --if-exists del-port "${BRIDGE}" "${host_if}" >/dev/null 2>&1 || true
  ip netns del "${ns}" >/dev/null 2>&1 || true
  ip link del "${host_if}" >/dev/null 2>&1 || true
}

cleanup() {
  cleanup_one "${TX_NS}" kvp-bwtx-h
  cleanup_one "${RX_NS}" kvp-bwrx-h
}

attach_ns() {
  local ns="$1"
  local host_if="$2"
  local peer_if="$3"
  local ip_addr="$4"
  ip netns add "${ns}"
  ip link add "${host_if}" type veth peer name "${peer_if}"
  ip link set "${peer_if}" netns "${ns}"
  ovs-vsctl --may-exist add-port "${BRIDGE}" "${host_if}"
  ip link set "${host_if}" up
  ip netns exec "${ns}" ip link set lo up
  ip netns exec "${ns}" ip link set "${peer_if}" name eth0
  ip netns exec "${ns}" ip addr add "${ip_addr}/24" dev eth0
  ip netns exec "${ns}" ip link set eth0 up
  ip netns exec "${ns}" tc qdisc replace dev eth0 root handle 1: htb default 10
  ip netns exec "${ns}" tc class replace dev eth0 parent 1: classid 1:10 htb rate "${RATE}" ceil "${RATE}"
}

snapshot() {
  local output="$1"
  {
    echo "rate=${RATE}"
    for ns in "${TX_NS}" "${RX_NS}"; do
      echo "=== ${ns}"
      ip netns exec "${ns}" tc -s qdisc show dev eth0
      ip netns exec "${ns}" tc -s class show dev eth0
    done
  } >"${output}"
}

setup() {
  [[ "$(id -u)" == 0 ]] || { echo "run with sudo" >&2; return 2; }
  [[ ! -e "${RUN_ROOT}" ]] || { echo "refusing to overwrite ${RUN_ROOT}" >&2; return 2; }
  mkdir -p "${RUN_ROOT}"
  cleanup
  attach_ns "${TX_NS}" kvp-bwtx-h kvp-bwtx-c "${TX_IP}"
  attach_ns "${RX_NS}" kvp-bwrx-h kvp-bwrx-c "${RX_IP}"
  snapshot "${RUN_ROOT}/tc-before.txt"
}

run_iperf() {
  local duration="${UNCHAIN_KV_IPERF_DURATION_S:-30}"
  local parallel="${UNCHAIN_KV_IPERF_PARALLEL:-1}"
  ip netns exec "${RX_NS}" iperf3 -s -1 -J >"${RUN_ROOT}/iperf-server.json" &
  local server_pid=$!
  sleep 0.25
  ip netns exec "${TX_NS}" iperf3 -c "${RX_IP}" -t "${duration}" -P "${parallel}" -J >"${RUN_ROOT}/iperf-client.json"
  wait "${server_pid}"
}

run_replay() {
  local path="${UNCHAIN_KV_REPLAY_PATH:-raw}"
  local requests="${UNCHAIN_KV_REPLAY_REQUESTS:-16}"
  local warmups="${UNCHAIN_KV_REPLAY_WARMUPS:-1}"
  local layers="${UNCHAIN_KV_REPLAY_LAYERS:-28}"
  local concurrency="${UNCHAIN_KV_REPLAY_CONCURRENCY:-4}"
  local offered_rps="${UNCHAIN_KV_REPLAY_OFFERED_RPS:-0.75}"
  local expected_frames=$(((requests + warmups) * layers))
  ip netns exec "${RX_NS}" env PYTHONPATH="${ROOT}/src" \
    python3 "${ROOT}/scripts/bench_bandwidth_replay.py" tcp-server \
      --bind "${RX_IP}:29620" \
      --expected-frames "${expected_frames}" \
      --output "${RUN_ROOT}/server.json" &
  local server_pid=$!
  sleep 0.25
  ip netns exec "${TX_NS}" env \
    PYTHONPATH="${ROOT}/src" \
    UNCHAIN_KV_TCP_LIB="${ROOT}/native/unchain_kv/build/libunchain_kv_tcp.so" \
    python3 "${ROOT}/scripts/bench_bandwidth_replay.py" tcp-client \
      --peer "${RX_IP}:29620" \
      --path "${path}" \
      --requests "${requests}" \
      --warmup-requests "${warmups}" \
      --layers "${layers}" \
      --concurrency "${concurrency}" \
      --offered-rps "${offered_rps}" \
      --output "${RUN_ROOT}/client.json"
  wait "${server_pid}"
}

print_plan() {
  echo "run_id=${RUN_ID}"
  echo "rate=${RATE}"
  echo "modes: iperf, replay"
}

main() {
  local mode="${1:-plan}"
  if [[ "${mode}" == plan ]]; then
    print_plan
    return
  fi
  trap cleanup EXIT INT TERM
  setup
  case "${mode}" in
    iperf) run_iperf ;;
    replay) run_replay ;;
    *) echo "usage: $0 {plan|iperf|replay}" >&2; return 2 ;;
  esac
  snapshot "${RUN_ROOT}/tc-after.txt"
  echo "${RUN_ROOT}"
}

main "$@"
