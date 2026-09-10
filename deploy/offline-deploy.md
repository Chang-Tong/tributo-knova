# Offline deployment

Build and package on an internet-connected machine with the target server's
CPU architecture:

```bash
docker compose build consumer
./scripts/package_image.sh tributo-knova:local
```

The packaging script streams `docker save` through zstd level 19, validates the
archive, writes a SHA-256 checksum, and records the image OS/architecture. Copy
all three files from `dist/` to the target.

On the target server:

```bash
cd /opt/images
shasum -a 256 -c tributo-knova.tar.zst.sha256
zstd -t tributo-knova.tar.zst
zstdcat tributo-knova.tar.zst | docker load
docker image inspect tributo-knova:local --format '{{.Os}}/{{.Architecture}}'
```

Copy `compose.yaml` and `deploy/config/knova-compose.json`, point Redis/Ray and
storage environment values at the target infrastructure, then start without a
registry pull:

```bash
docker compose up -d --no-build
docker compose ps
docker compose exec consumer tributo-knova health \
  --config /etc/tributo-knova/config.json
```

For systemd, install the virtual environment at `/opt/tributo-knova`, copy the
runtime package trees to `/opt/tributo-runtime`, and install the supplied config
and unit under `/etc/tributo-knova` and `/etc/systemd/system`. Put S3-compatible
credentials only in `/etc/tributo-knova/environment` with mode `0600`; never add
them to the JSON config.
