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
KnoVa v2 lifecycle events. Redis Streams consumption, consumer groups, pending
recovery, cancellation, Ray Job admission, retries, and acknowledgements remain
owned by `tributo-broker-redis`.

See [the implementation plan](docs/implementation-plan.md) for the completed
vertical slices and the remaining release work. This branch is not a 1.0
release yet: S3 publication, checkpoint-resume testing, UBJ-native inference,
SHAP, and production deployment assets remain open.

## Development

The local development configuration resolves both dependencies from sibling
repositories during the smoke test. Normal installation uses immutable Git
commits for Tributo, the Redis Broker, and the official boosting package until
compatible releases are published:

```bash
uv sync --extra dev
uv run pytest
```

## Local end-to-end smoke test

With Redis, Ray, and ClickHouse already running locally, set the ClickHouse
password through the environment and run:

```bash
export TRIBUTO_KNOVA_CLICKHOUSE_PASSWORD='...'
uv run python scripts/local_smoke.py
```

The script creates isolated ClickHouse tables, submits a real training task and
a real inference task through Redis, verifies one terminal event per task and
the exact output row count, then removes its temporary tables and model Bundle.
Sibling source checkouts can be overridden with `--core-root`,
`--broker-root`, and `--algorithms-root`.
