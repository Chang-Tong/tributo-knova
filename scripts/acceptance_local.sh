#!/usr/bin/env bash
# Disposable local-only acceptance. No ports are exposed to the host.
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACCEPT_IMAGE="${ACCEPT_IMAGE:-knova-tributo-devbox:v2.5}"
export ACCEPT_ARTIFACTS="${ACCEPT_ARTIFACTS:-${script_dir}/../dist/acceptance-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${ACCEPT_ARTIFACTS}/shared"
ACCEPT_ARTIFACTS="$(cd "${ACCEPT_ARTIFACTS}" && pwd)"
export ACCEPT_ARTIFACTS
docker image inspect "$ACCEPT_IMAGE" > /dev/null
compose=(docker compose -p knova-local-accept -f "${script_dir}/acceptance.compose.yaml")
if [[ -n "$("${compose[@]}" ps -aq)" ]]; then
  echo 'Existing local acceptance resources found; inspect and remove them first.' >&2
  exit 2
fi
cleanup() {
  status=$?
  trap - EXIT
  "${compose[@]}" logs --no-color > "${ACCEPT_ARTIFACTS}/services.log" 2>&1 || true
  if [[ "${ACCEPT_KEEP:-0}" != 1 ]]; then
    "${compose[@]}" down --volumes > "${ACCEPT_ARTIFACTS}/cleanup.log" 2>&1 || true
  fi
  echo "Acceptance exit=${status}; evidence: ${ACCEPT_ARTIFACTS}"
  exit "$status"
}
trap cleanup EXIT
"${compose[@]}" up -d > "${ACCEPT_ARTIFACTS}/startup.log" 2>&1
"${compose[@]}" run --rm --no-deps workloads 2>&1 | tee "${ACCEPT_ARTIFACTS}/workloads.log"
