#!/usr/bin/env bash
# Isolated real Sentinel failover; only disposable local credentials are used.
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output="${ACCEPT_ARTIFACTS:-${script_dir}/../dist/sentinel-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$output"
output="$(cd "$output" && pwd)"
if [[ -e "$output/sentinel-ready" ]]; then
  echo "Use a fresh artifact directory for each Sentinel run." >&2
  exit 2
fi
prefix="knova-sentinel-accept-$$"
containers=()
cleanup() {
  status=$?
  trap - EXIT
  for container in "${containers[@]}"; do
    docker logs "$container" > "$output/${container}.log" 2>&1 || true
    docker rm -f "$container" >/dev/null 2>&1 || true
  done
  docker network rm "$prefix" >/dev/null 2>&1 || true
  echo "Sentinel exit=${status}; evidence: $output"
  exit "$status"
}
trap cleanup EXIT
docker network create "$prefix" >/dev/null
master="${prefix}-master"
containers+=("$master")
docker run -d --name "$master" --network "$prefix" redis:8-bookworm \
  redis-server --requirepass local-repro-only >/dev/null
master_ip="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$master")"
replica="${prefix}-replica"
containers+=("$replica")
docker run -d --name "$replica" --network "$prefix" redis:8-bookworm \
  redis-server --requirepass local-repro-only --masterauth local-repro-only \
  --replicaof "$master_ip" 6379 >/dev/null
for i in 1 2 3; do
  mkdir -p "$output/s$i"
  cat > "$output/s$i/sentinel.conf" <<CONF
port 26379
bind 0.0.0.0
protected-mode no
requirepass local-sentinel-only
sentinel monitor mymaster $master_ip 6379 2
sentinel auth-pass mymaster local-repro-only
sentinel down-after-milliseconds mymaster 3000
sentinel failover-timeout mymaster 10000
CONF
  container="${prefix}-s$i"
  containers+=("$container")
  docker run -d --name "$container" --network "$prefix" \
    --network-alias "accept-sentinel-$i" -v "$output/s$i:/etc/redis" \
    redis:8-bookworm redis-server /etc/redis/sentinel.conf --sentinel >/dev/null
done
for _ in $(seq 1 30); do
  replication="$(docker exec -e REDISCLI_AUTH=local-repro-only "$master" redis-cli INFO replication)"
  [[ "$replication" == *connected_slaves:1* ]] && break
  sleep 1
done
[[ "$replication" == *connected_slaves:1* ]]
probe="${prefix}-probe"
containers+=("$probe")
docker run -d --name "$probe" --network "$prefix" \
  -v "$script_dir:/scripts:ro" -v "$output:/artifacts" \
  --entrypoint python "${ACCEPT_IMAGE:-knova-tributo-devbox:v2.5}" \
  /scripts/acceptance_sentinel.py >/dev/null
for _ in $(seq 1 60); do
  [[ -f "$output/sentinel-ready" ]] && break
  sleep 1
done
[[ -f "$output/sentinel-ready" ]]
docker stop "$master" >/dev/null
touch "$output/sentinel-master-stopped"
probe_status="$(docker wait "$probe")"
[[ "$probe_status" == 0 ]]
