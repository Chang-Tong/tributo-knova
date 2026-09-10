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
XGBoost UBJ flavor, plus KnoVa v2 lifecycle events. Redis Streams consumption,
consumer groups, pending recovery, cancellation, Ray Job admission, retries,
and acknowledgements remain owned by `tributo-broker-redis`.

See [the implementation plan](docs/implementation-plan.md) for the completed
vertical slices and the remaining release work. This branch is not a 1.0
release yet: checkpoint-resume support, direct UBJ-only prediction, and
production deployment assets remain open.

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
uv run python scripts/local_smoke.py --shap-mode exact
uv run python scripts/local_smoke.py --shap-mode approximate
```

The script creates isolated ClickHouse tables, submits a real training task and
a real inference task through Redis, verifies one terminal event per task and
the exact output row count, then removes its temporary tables and model Bundle.
The SHAP modes additionally verify the explanation exactness and feature width
inside a real ClickHouse `pred_extra` result.
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
loads that S3 Bundle for inference, and removes the prefix afterward.
