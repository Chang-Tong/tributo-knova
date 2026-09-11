# Verification matrix

The repository keeps one implementation path: KnoVa protocol v2 is adapted to
the public Tributo, Redis Broker, and official boosting contracts. Credentials
are supplied only through the process environment.

| Area | Verified path | Evidence |
| --- | --- | --- |
| Protocol | Training `job_id`; inference `execution_id`; required tenant/model/version/task identities; v2 lifecycle events | Unit suite |
| Broker | Redis Streams admission, ACK, retry, idempotency, queued cancellation, watchdog and one terminal event | Unit suite |
| Restart recovery | `XAUTOCLAIM` pending delivery recovery and active Ray Job cancellation state reconstruction | Unit suite and `--exercise-recovery` smoke |
| ClickHouse input | Ray-native ordered reads; AUTO discovery preserves simple composite sorting keys; Ray owns count/size/schema estimates and block scheduling | Unit suite and KnoVa devbox integration |
| Training | Ray-native ClickHouse input, lossless train/validation/test split, two-worker distributed XGBoost and live per-round metrics | Unit suite and KnoVa devbox integration |
| Evaluation | Full test-set metrics, confusion matrix, ROC, threshold analysis, gain importance and distributed Pearson feature correlation | Unit suite and KnoVa devbox integration |
| Checkpoint | Official `model.ubj` and `feature_names.json` checkpoint on shared NAS/NFS path | Local integration smoke |
| Bundle | Official ONNX and UBJ artifacts, content digest and Bundle identity validation | Unit suite and local/S3 smoke |
| Protocol projection | Training `artifact_manifest` is passed directly into inference with exact paths, sizes and SHA-256 hashes | Local S3 smoke |
| Inference | ONNX prediction plus exact or explicitly approximate UBJ TreeSHAP | Local integration smoke |
| Output | Ray-native ClickHouse write, dynamic measured-row batching and bounded pre-write retry | Unit suite and local integration smoke |
| Deployment | Non-root image, Redis/Ray/Consumer Compose health, hardened systemd unit | Compose smoke |
| Offline | `docker save` + `zstd -19 -T0`, archive test, SHA-256 and platform record | Packaging script and generated package verification |

## Explicit upstream limits

- The pinned official boosting package persists a checkpoint but explicitly
  rejects `resume_from`; cross-process training resume waits for its recovery
  gate.
- Direct UBJ-only prediction is not enabled because the current official native
  predictor drops feature names. The supported complete Bundle path is ONNX
  prediction plus UBJ attribution.
- Dependencies are pinned to immutable commits until compatible releases are
  tagged.
