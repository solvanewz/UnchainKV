#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:?usage: inject_fault.sh ACTION RUN_ROOT}"
RUN_ROOT="${2:?usage: inject_fault.sh ACTION RUN_ROOT}"
DELAY_S="${UNCHAIN_KV_FAULT_DELAY_S:-5}"
HOLD_S="${UNCHAIN_KV_FAULT_HOLD_S:-1}"
REPEAT="${UNCHAIN_KV_FAULT_REPEAT:-1}"
INTERVAL_S="${UNCHAIN_KV_FAULT_INTERVAL_S:-1}"
EVENTS="${RUN_ROOT}/fault-events.jsonl"

case "${ACTION}" in
  pause-producer|rate-drop|kill-prefill|kill-decode|reset-transport) ;;
  *) echo "unsupported fault action: ${ACTION}" >&2; exit 2 ;;
esac
[[ "${REPEAT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "fault repeat must be positive" >&2
  exit 2
}

record() {
  printf '{"action":"%s","cycle":%s,"event":"%s","utc":"%s"}\n' \
    "${ACTION}" "$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >>"${EVENTS}"
}

set_rate() {
  local rate="$1" name
  for name in kvp-proxy kvp-prefill kvp-decode; do
    ip netns exec "${name}" tc class replace dev eth0 parent 1: \
      classid 1:10 htb rate "${rate}" ceil "${rate}"
  done
}

wait_transport() {
  local attempt
  for ((attempt = 0; attempt < 100; attempt++)); do
    if ip netns exec kvp-prefill ss -Htn dst 172.16.0.32 \
      dport = :29620 | grep -q .; then
      return
    fi
    sleep 0.1
  done
  echo "transport socket not found" >&2
  return 1
}

sleep "${DELAY_S}"
for ((cycle = 1; cycle <= REPEAT; cycle++)); do
  record "${cycle}" start
  case "${ACTION}" in
    pause-producer)
      docker pause kvp-prefill >/dev/null
      sleep "${HOLD_S}"
      docker unpause kvp-prefill >/dev/null
      ;;
    rate-drop)
      set_rate "${UNCHAIN_KV_FAULT_LOW_RATE:-2gbit}"
      sleep "${HOLD_S}"
      set_rate "${UNCHAIN_KV_NET_RATE:-10gbit}"
      ;;
    kill-prefill)
      docker exec kvp-prefill sh -c \
        "kill -KILL \$(pgrep -f '[v]llm serve' | head -n 1)"
      ;;
    kill-decode)
      docker exec kvp-decode sh -c \
        "kill -KILL \$(pgrep -f '[v]llm serve' | head -n 1)"
      ;;
    reset-transport)
      wait_transport
      ip netns exec kvp-prefill ss -K dst 172.16.0.32 dport = :29620
      ;;
  esac
  record "${cycle}" complete
  (( cycle == REPEAT )) || sleep "${INTERVAL_S}"
done
