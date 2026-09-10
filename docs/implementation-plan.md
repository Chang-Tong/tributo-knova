# tributo-knova implementation plan

`tributo-knova` is an integration package, not a fork of Tributo. It imports
the public `tributo` and `tributo-broker-redis` packages and adds only the
KnoVa-specific protocol and deployment behavior that those packages should not
own.

## Ownership

| Capability | Owner |
| --- | --- |
| Algorithm execution, ingestion contracts, inference, Bundle and export | `tributo` |
| Redis Streams, consumer groups, pending recovery, retry, cancellation and Ray Job admission | `tributo-broker-redis` |
| KnoVa protocol v2 mapping, event compatibility, ClickHouse extensions and deployment defaults | `tributo-knova` |
| XGBoost implementation, ONNX and UBJ exporters | official `tributo-algorithms-boosting` package |

KnoVa code should call these public contracts directly. It should not copy the
Redis consumer or create a parallel training/inference API.

## Milestone 1 — protocol and broker vertical slice

Status: implemented and covered by unit tests. The imported Broker watchdog now
publishes `FAILED` when a Ray Job exits before the driver can create its
reporter; the KnoVa adapter verifies that this uses the protocol-v2 terminal
envelope.

- Accept the current training and inference v2 envelopes. Keep `job_id` on the
  training wire while using `TrainingExecutionRequest` and `execution_id`
  consistently inside Python.
- Adapt validated requests to the broker's existing `GenericRequest` and
  `PreparedOperation` contracts.
- Add only the Broker injection points needed to select the KnoVa parser,
  mapper and Ray driver.
- Align task streams, outer identity fields and cancellation keys with the
  existing KnoVa defaults.
- Add KnoVa v2 event envelopes for admission, driver progress and terminal
  events. Preserve the Broker's single-terminal and retry behavior.

Acceptance: both current example requests pass admission; a fake Redis task is
submitted once, ACKed once and carries the unchanged validated request into the
Ray driver.

## Milestone 2 — XGBoost training and artifacts

Status: the public algorithm dispatcher, two-worker XGBoost execution, NFS/NAS
storage mapping, and content-addressed Bundle publication are implemented and
covered by the local end-to-end smoke test. The same test now verifies the Ray
checkpoint artifacts on shared storage and publishes/loads the final Bundle
through a local S3-compatible RustFS service. Cross-restart resume remains open:
the pinned official boosting implementation explicitly rejects `resume_from`
until its separate recovery gate is available.

- Map the existing training request to Tributo's formal algorithm execution
  contracts and use `tributo-algorithms-boosting` for distributed XGBoost.
- Map NAS/NFS `storage_context` to Ray checkpoint storage and map final model
  storage to Tributo Bundle publication.
- Publish ONNX and UBJ model variants plus the Bundle manifest and metadata
  through the existing exporter/Bundle path; do not implement exporters in
  KnoVa. Add metrics only through an upstream exporter contract rather than
  inventing a KnoVa-only file format.
- Preserve request-digest idempotency and terminal replay behavior.

Acceptance: two-worker training produces ONNX and UBJ artifacts, resumes from a
shared checkpoint where supported, uploads the final Bundle to S3, and emits one
KnoVa terminal event.

## Milestone 3 — batch inference and ClickHouse

Status: Ray-native ClickHouse reads, ONNX Bundle inference, protocol-v2 result
mapping, ClickHouse writes, filter pushdown, and exact terminal row counts are
implemented and covered by the local end-to-end smoke test. Measured-row
adaptive batch sizing and bounded pre-write memory-pressure retries are also
implemented. Exact and explicitly marked approximate TreeSHAP use the official
Bundle native role/UBJ model after public Tributo ONNX prediction; both modes
are covered by the same end-to-end smoke test. Direct UBJ-only prediction
remains open because the current upstream native predictor drops feature names.

- Implement a KnoVa-owned Ray-native ClickHouse ingestion Binding through
  Tributo's `tributo.ingestion_bindings` entry point.
- Route inference output through Tributo's existing `BoundResultSink` extension
  point and `data-write-v1` receipt contract. The sink delegates to Ray's native
  ClickHouse writer and keeps credentials out of the public Core request and
  receipt.
- Map ONNX and UBJ model references to Tributo's existing Bundle inference
  runtime. Use exact TreeSHAP where the model flavor supports it and Tributo's
  approximate explainer otherwise.
- Keep adaptive batch sizing as a bounded KnoVa execution policy around the
  existing inference executor.

Acceptance: ClickHouse-to-ClickHouse inference works for ONNX and UBJ Bundles,
emits exact or explicitly marked approximate SHAP output, and reduces batch size
without losing or duplicating rows when memory pressure is simulated.

## Milestone 4 — operations and release

Status: a repeatable local Redis/Ray/ClickHouse smoke test exists. Container,
systemd, health/readiness, offline packaging, and recovery/failure matrices
remain open.

- Add the consumer image, Compose example, systemd unit, health/readiness
  checks, and zstd offline image packaging.
- Add Redis/Ray/ClickHouse/S3 integration tests and failure-injection tests for
  retry, cancellation, consumer restart and duplicate delivery.
- Pin compatible releases of Tributo, the Redis broker and official algorithm
  packages before tagging `tributo-knova` 1.0.

Acceptance: one documented online deployment and one offline deployment pass
the same smoke test and can recover an in-flight task after consumer restart.
