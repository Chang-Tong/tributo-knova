#!/bin/sh
set -eu

image="${1:-tributo-knova:local}"
output="${2:-dist/tributo-knova.tar.zst}"

case "$output" in
  *.tar.zst) ;;
  *) echo "output must end in .tar.zst" >&2; exit 2 ;;
esac

for command_name in docker zstd shasum; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "$command_name is required" >&2
    exit 1
  }
done

docker image inspect "$image" >/dev/null
output_dir=$(dirname "$output")
output_name=$(basename "$output")
mkdir -p "$output_dir"

docker save "$image" | zstd -19 -T0 -f -o "$output"
zstd -t "$output"
(
  cd "$output_dir"
  shasum -a 256 "$output_name" >"$output_name.sha256"
)
docker image inspect "$image" --format '{{.Os}}/{{.Architecture}}' \
  >"$output.platform"

echo "created $output, $output.sha256, and $output.platform"
