#!/usr/bin/env bash
set -euo pipefail

unexpected=()
while IFS= read -r name; do
  case "${name}" in
    buildx_buildkit_*|gpustack-worker) ;;
    *) unexpected+=("${name}") ;;
  esac
done < <(docker ps --format '{{.Names}}')

((${#unexpected[@]} == 0)) || {
  printf 'experiment containers still running%s: %s\n' "${1:+ $1}" "${unexpected[*]}" >&2
  exit 1
}
