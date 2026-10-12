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

Production Sentinel Redis is configured with
`deploy/config/knova-sentinel.json`. Redis and Sentinel passwords remain in
environment variables or mounted secret files; they must never be copied into
the JSON file. The configured references must be available to both the
Consumer and the external Ray Job driver. If Consumer and Ray nodes reach
Sentinel through different addresses, keep `sentinel_nodes` and
`driver_sentinel_nodes` distinct.

Run only the Consumer against the existing Sentinel and Ray services:

```bash
cp deploy/environment.example deploy/.env
# Fill KNOVA_REDIS_PASSWORD and the existing storage settings in deploy/.env.
docker compose --env-file deploy/.env \
  -f deploy/compose.sentinel.yaml up -d
```

The Compose file intentionally does not create Redis, ClickHouse, Ray, object
storage, or any KnoVa control-plane service. The Ray cluster must receive the
same named Redis secret independently because the password is never forwarded
inside a Ray Job submission.

For KubeRay, add the following to the Consumer container and to the Ray
container in both `spec.headGroupSpec.template.spec.containers` and every
`spec.workerGroupSpecs[].template.spec.containers`. The Secret must exist in
each Pod's namespace. Update the RayCluster Pod templates and recreate/roll
the Pods so that the running Ray processes inherit the new environment.

```yaml
envFrom:
  - secretRef:
      name: tributo-knova-env
```

That Secret supplies `KNOVA_REDIS_PASSWORD`, and
`KNOVA_REDIS_SENTINEL_PASSWORD` when `sentinel_password_ref` is configured.
Keep these out of `execution.env_vars`. The Consumer's `health` command only
checks the Consumer's connectivity; it does not prove the Ray Driver has
credentials or can reach Sentinel and the Redis master discovered by Sentinel.

## Incremental provider image

`Dockerfile.hotfix` updates the KnoVa adapter and the vendored broker wheel on
the validated base. It synchronizes both installed packages and the
`/opt/tributo-runtime/extensions` copies uploaded through Ray `py_modules`.
Rebuild the broker wheel after changing its source, then build the image:

```bash
# Run from the sibling tributo-broker-redis checkout:
uv build --wheel --out-dir ../tributo-knova/vendor/wheels
# Run from the tributo-knova checkout:
docker build --platform linux/arm64 -f Dockerfile.hotfix \
  -t knova-tributo-devbox:v2.3 .
docker run --rm --entrypoint python knova-tributo-devbox:v2.3 \
  /opt/tributo-runtime/verify_image.py
```

### Test the packaged dependencies locally

Run these commands from the `tributo-knova` repository root. This creates a
separate environment under `dist`; it does not use the development `.venv`.
The four wheels in `vendor/wheels` must be the ones intended for the image.

```bash
KNOVA_PACKAGED_ENV="$PWD/dist/packaged-test-venv"

# Install the locked third-party dependencies and the current KnoVa adapter,
# leaving the four packaged dependencies for the wheel installation below.
UV_PROJECT_ENVIRONMENT="$KNOVA_PACKAGED_ENV" uv sync \
  --python 3.13 --frozen --extra dev --offline \
  --no-install-package tributo \
  --no-install-package tributo-broker-redis \
  --no-install-package tributo-algorithms-boosting \
  --no-install-package ray-clickhouse

# --no-config prevents the project's override-dependencies from replacing
# the local Tributo wheel with the Git revision in pyproject.toml.
uv pip install --no-config --offline --no-deps \
  --python "$KNOVA_PACKAGED_ENV/bin/python" vendor/wheels/*.whl

# Run the installed interpreter directly: do not re-sync after installing wheels.
"$KNOVA_PACKAGED_ENV/bin/python" -c \
  'import tributo_broker_redis.driver_transport as m; print(m.__file__)'
"$KNOVA_PACKAGED_ENV/bin/python" -m pytest -q tests
```

`--offline` on `uv sync` requires its locked dependencies and Python runtime
to be cached. On a fresh machine, omit that flag from the first command to
populate the cache; keep all four `--no-install-package` options. The wheel
installation stays offline and does not resolve additional dependencies.

The lockfile still contains the upstream Git pins. Running `uv run pytest`
or `uv run --offline pytest` after the wheel installation can synchronize
back to those older revisions, including a broker that lacks
`driver_transport`. `--offline` controls network access; it does not disable
synchronization. Also retain `--no-config` on `uv pip install`: the project's
`tool.uv.override-dependencies` otherwise overrides the supplied local
Tributo wheel. Use the explicit environment Python for all subsequent tests.

To reproduce the full local acceptance run, the same sequence can target
`dist/comprehensive-recheck/packaged-venv`; its recorded test output is
`dist/comprehensive-recheck/packaged-dependencies-pytest.log` when that local
evidence directory is present. These commands validate Python integration
against the packaged dependencies; the image verification and real Ray/Web UI
checks remain separate deployment checks.

Driver startup now logs its actual module path. Validate the deployment with
a newly submitted job and `ray job logs <submission_id>`, rather than importing
the module in a shell. The Driver entrypoint is `python -m
tributo_knova.execution_driver`, which uses `__main__` as its module name.
Do not run that entrypoint manually without the Ray Job's input environment.

The Consumer watches Ray terminal status and publishes a fallback failure if
the Driver could not publish one (for example, missing Redis credentials).
Consumer logs include `Fallback FAILED event published` with the operation,
submission ID and event stream. If publishing fails, the watcher logs the
exception and retries. On restart it also reconciles already-ended jobs without
a terminal event. A Redis outage still prevents publication until connectivity
returns.

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
The devbox config uses dedicated consumer groups starting at `$`, so bringing
up a developer sidecar cannot reclaim stale deliveries owned by an older
runtime. Production configs retain their normal recovery-oriented groups.

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
