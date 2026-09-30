#!/usr/bin/env bash
set -Eeuo pipefail
if [[ $# != 2 ]]; then
  echo 'Usage: import-image.sh IMAGE.tar.zst docker|containerd|nerdctl' >&2
  exit 2
fi
archive="$1"
runtime="$2"
case "$runtime" in docker|containerd|nerdctl) ;; *) echo 'Unknown runtime' >&2; exit 2;; esac
archive_dir="$(cd "$(dirname "$archive")" && pwd)"
archive_name="$(basename "$archive")"
cd "$archive_dir"
if command -v sha256sum >/dev/null; then
  sha256sum -c "${archive_name}.sha256"
else
  shasum -a 256 -c "${archive_name}.sha256"
fi
zstd --test "$archive_name"
case "$runtime" in
  docker) zstd -dc "$archive_name" | docker load ;;
  containerd) zstd -dc "$archive_name" | ctr --namespace k8s.io images import - ;;
  nerdctl) zstd -dc "$archive_name" | nerdctl --namespace k8s.io load ;;
esac
echo 'Imported on this host only. All schedulable Kubernetes nodes need the image or registry access.'
