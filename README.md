# tributo-knova

KnoVa integration package built on the public `tributo` and
`tributo-broker-redis` packages.

This repository starts as a thin integration layer. It should reuse the two
upstream packages directly and add only KnoVa protocol mapping, missing runtime
behavior, and deployment assets.

The current implementation accepts the existing KnoVa protocol v2 training and
batch-inference envelopes and adapts them to the public Redis broker runtime.
It provides distributed XGBoost training, content-addressed Tributo Bundles,
Ray-native ClickHouse input, Bundle inference, ClickHouse result output, and
exact or explicitly approximate per-row TreeSHAP output through the official
XGBoost UBJ flavor, split-aware terminal evaluation, live round metrics,
feature importance, and distributed feature correlation, plus KnoVa v2
lifecycle events. Redis Streams consumption, consumer groups, pending recovery,
cancellation, Ray Job admission, retries, and acknowledgements remain owned by
`tributo-broker-redis`.

See [the implementation plan](docs/implementation-plan.md) for the completed
vertical slices and the remaining upstream limits. See the
[verification matrix](docs/verification-matrix.md) for the tested paths. This
branch is not a 1.0 release yet: cross-restart checkpoint resume, direct
UBJ-only prediction, and compatible upstream release tags remain open.

## ClickHouse parallel reads

Training and inference share one Tributo ingestion Binding. It imports
`ray-clickhouse` for bounded Arrow streaming and disjoint physical-partition or
integer-range reads, while all execution still uses Ray's public Datasource API.
For inference, each emitted source block is capped at 200,000 rows and 64 MiB;
the measured input row count determines the requested number of range tasks.
Leading equality predicates are sent to ClickHouse as bound parameters and are
recorded as exact Tributo pushdowns.

For Tributo `auto` partitioning, KnoVa reads the table engine, active physical
partitions, sorting key, and first key type. Multiple physical partitions are
balanced by on-disk bytes. Otherwise an integer first sorting column creates
disjoint range tasks. With a composite key, only its first integer column is
needed for row coverage; the remaining columns do not need to be discarded or
invented as independent split keys.

When a simple sorting key exists but its first column is not an integer, KnoVa
retains `ray.data.read_clickhouse(order_by=...)` so existing parallel ordered
reads keep working. A key such as `tenant_id, event_time, user_id` is passed in
its original order. This fallback uses Ray's native global ORDER BY/OFFSET
implementation.

Sorting-key SQL expressions are not passed through as arbitrary SQL. If neither
physical partitions nor an integer range key is available, the Binding uses one
bounded streaming task and emits a single-worker pressure warning. If metadata
itself is unavailable, it falls back to Ray's native single-task reader. KnoVa
does not invent unsafe split predicates that could duplicate or omit rows.

## Development

The local development configuration resolves both dependencies from sibling
repositories during the smoke test. Normal installation uses immutable Git
commits for Tributo, the Redis Broker, and the official boosting package until
compatible releases are published:

```bash
uv sync --frozen --extra dev
uv run pytest
```

## Local end-to-end smoke test

With Redis, Ray, and ClickHouse already running locally, set the ClickHouse
password through the environment and run:

```bash
export TRIBUTO_KNOVA_CLICKHOUSE_PASSWORD='...'
uv run python scripts/local_smoke.py
uv run python scripts/local_smoke.py --shap-mode exact
uv run python scripts/local_smoke.py --shap-mode approximate
uv run python scripts/local_smoke.py --shap-mode exact --exercise-recovery
```

The script creates isolated ClickHouse tables, submits a real training task and
a real inference task through Redis, verifies one terminal event per task and
the exact output row count, then removes its temporary tables, Redis messages,
events, and model Bundle.
The SHAP modes additionally verify the explanation exactness and feature width
inside a real ClickHouse `pred_extra` result.
The recovery mode also stages a pending Redis delivery and an admitted Ray Job,
restarts the Consumer, and verifies that both are rediscovered and cancelled
with one terminal event each.
Sibling source checkouts can be overridden with `--core-root`,
`--broker-root`, and `--algorithms-root`.

To exercise shared Ray checkpoint storage and S3-compatible final Bundle
publication, configure the standard AWS credential chain on the Ray nodes and
the smoke-test process, then run:

```bash
export AWS_ENDPOINT_URL='http://127.0.0.1:9100'
export AWS_REGION='us-east-1'
uv run python scripts/local_smoke.py --storage-mode s3 --shap-mode exact
```

Credentials stay in the deployment environment; they are not copied into the
KnoVa request, Bundle manifest, or Ray Job metadata. The smoke test creates an
isolated object prefix, verifies the published ONNX, UBJ and manifest objects,
passes the training `artifact_manifest` directly into inference, and removes
the prefix afterward.

## Deployment

The repository includes one production Consumer image shared by the consumer
and Ray head, a minimal Compose stack, a hardened systemd unit, and dependency
readiness checks:

```bash
docker compose up -d --build
docker compose exec consumer tributo-knova health \
  --config /etc/tributo-knova/config.json
```

See [deployment instructions](deploy/README.md) and
[offline deployment](deploy/offline-deploy.md). Offline packages are created
with `./scripts/package_image.sh`; the script records platform compatibility,
generates a SHA-256 checksum, and validates the zstd archive before returning.
