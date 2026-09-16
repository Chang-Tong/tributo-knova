# Deployment

## Docker Compose

The Compose stack runs three services from two images:

- `redis` owns task, event, cancellation, and consumer-group state.
- `ray-head` runs the Ray Job server and workers from the same immutable
  `tributo-knova` image as the Consumer.
- `consumer` imports `tributo`, `tributo-broker-redis`, and the official
  boosting package, then adapts KnoVa protocol-v2 messages to them.

Start the stack and verify both external dependencies from the Consumer:

```bash
docker compose up -d --build
docker compose ps
docker compose exec consumer tributo-knova health \
  --config /etc/tributo-knova/config.json
```

The example deliberately does not publish Redis or Ray ports. Add explicit
loopback-only port mappings when local development tools need host access.
ClickHouse remains a KnoVa-managed datasource supplied in each task request.

For S3-compatible Bundle storage, provide the standard `AWS_*` variables to
both `ray-head` and `consumer`. Keep them out of the JSON config and task
payload. NAS/NFS checkpoint paths must be mounted at the same absolute path on
every Ray node.

## KnoVa devbox sidecar

The repository also provides `compose.devbox.yaml` for end-to-end Web UI
development. It does not start a second KnoVa control plane or copy the legacy
closed-source runtime. Instead, it joins the existing external
`knova-network`, consumes the distributed training and inference streams from
Redis database 3, and submits the translated work to a dedicated Ray head.

Start the KnoVa devbox infrastructure and its `ai-model` profile first. Redis,
RustFS, ClickHouse, and `ai-model-control` must all be reachable on
`knova-network`. Then build the current adapter image and start the sidecar:

```bash
docker build -t tributo-knova:devbox .
docker compose -f compose.devbox.yaml up -d
docker compose -f compose.devbox.yaml ps
docker compose -f compose.devbox.yaml exec tributo-knova-consumer \
  tributo-knova health --config /etc/tributo-knova/config.json
```

To reuse an already validated local image, set `TRIBUTO_KNOVA_IMAGE` instead
of rebuilding it. The Compose file uses `pull_policy: never`, so it cannot
silently replace the locally validated image with an unrelated registry tag.

```bash
TRIBUTO_KNOVA_IMAGE=tributo-knova:validated \
  docker compose -f compose.devbox.yaml up -d
```

Stop only the sidecar without changing the devbox databases or application
containers:

```bash
docker compose -f compose.devbox.yaml down
```

This is the development integration path. Once the immutable image is pushed
to the internal registry, the same two services can be included as a devbox
`tributo` profile without changing the Redis protocol or KnoVa Web UI.

## systemd

Install the application virtual environment at `/opt/tributo-knova`, and put
the packaged Core and extension module directories under
`/opt/tributo-runtime` as shown by the supplied config. Then install:

```bash
install -d -m 0750 -o tributo-knova -g tributo-knova /etc/tributo-knova
install -m 0640 -o root -g tributo-knova \
  deploy/config/knova-systemd.json /etc/tributo-knova/config.json
install -m 0644 deploy/systemd/tributo-knova.service \
  /etc/systemd/system/tributo-knova.service
systemctl daemon-reload
systemctl enable --now tributo-knova
systemctl status tributo-knova
```

The unit runs `tributo-knova health` before consumption, restarts on failure,
and uses systemd filesystem hardening. Store S3-compatible credentials in
`/etc/tributo-knova/environment` with mode `0600` when needed.

## Configuration check

`validate` checks the full broker and runtime-path schema. Add
`--check-connectivity` to ping Redis. `health` always checks both Redis and the
Ray dashboard:

```bash
tributo-knova validate --config /etc/tributo-knova/config.json
tributo-knova health --config /etc/tributo-knova/config.json
```
