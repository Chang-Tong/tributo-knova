# Air-gapped dependency wheels

The Docker build installs these checked-in pure-Python wheels instead of
fetching the four Git dependencies. Ordinary third-party packages still come
from the configured Python package index. The broker and boosting wheels
include local v2.5 / v2.7 / v2.8 patches; their locked Git commits alone do not reproduce
the shipped code. The ray-clickhouse wheel also contains earlier local changes.

| Wheel | Source base and local changes | SHA-256 |
| --- | --- | --- |
| `ray_clickhouse-0.1.0-py3-none-any.whl` | Locked reference: `Chang-Tong/ray-clickhouse-upstream@c1627101736463ccc94c99c6793f91360173e6ac`; earlier local changes, see limitation below | `e5668178a6baf9c8fd418a80c7a02a2114b9f57b31b62972cc36897fb76fa267` |
| `tributo-1.0.0-py3-none-any.whl` | `Chang-Tong/Tributo@440da0422a3a9c5b041b6ecdbb2e40c00bf43cd90` | `38c095e3812deabc16280281fb4ab06b5c4d53f09f94d751af945b8b76dc13f2` |
| `tributo_algorithms_boosting-0.1.0-py3-none-any.whl` | `Chang-Tong/tributo-algorithms@6f39378d4d850cfd41e9510abc313efc6d4f209f` + [boosting-v2.7.patch](../patches/boosting-v2.7.patch), then [boosting-v2.8-onnx-schema.patch](../patches/boosting-v2.8-onnx-schema.patch), build `packages/boosting` | `c8bbed6fd0c9f828b37de498113f578f93c4ccb2ba5811ff80ee8f4424d4e922` |
| `tributo_broker_redis-0.1.0-py3-none-any.whl` | `Chang-Tong/tributo-broker-redis@e2973b2a7409fc72807bbec23388d13966c5c319` + [broker-v2.8.patch](../patches/broker-v2.8.patch) | `8da5ef811ed7a4de0716ecd315e6ffafb9549803be226601c767280a2bc53178` |

## Reproducing the patched packages

Use a clean checkout of the source base in the table, apply the corresponding
patch with `git apply --check` and `git apply`, then build with Python 3.13.
For boosting, apply the cumulative v2.7 patch first, followed by the incremental
v2.8 ONNX schema patch (both paths are relative to the algorithms repository root):

```sh
uv build --wheel --no-sources --python 3.13 --out-dir /path/to/output .
```

Run that command at the broker repository root or in `packages/boosting` for
boosting. Add `--offline` when the build dependencies are already cached.
The patches contain source and regression tests, with no deployment environment
files. The boosting patch includes the earlier dataset-transform hook and
block-order randomization, the empty-shard fix, and the ONNX regression output
name alignment with the native checkpoint contract, plus binary evaluation
for either positive class encoding (training.positive_label=0 or 1). The v2.7
patch also converts binary logistic base_score probabilities to ONNX margins,
including constant zero-leaf trees. The v2.5 and v2.6 patches are retained
as historical provenance.

On 2026-09-24, applying the broker patch to a clean archive of its base and
building offline reproduced the shipped wheel byte-for-byte. All 13 broker
package Python files matched. The boosting patch also replayed cleanly; all
11 package Python files matched its shipped wheel. On 2026-09-25 (Asia/Shanghai),
the complete boosting v2.7 patch applied to a clean base archive and an offline
build reproduced the new wheel byte-for-byte; 40 targeted algorithm tests passed,
including imbalanced binary/multiclass and nonzero-mean regression intercepts.

On 2026-09-30, the broker v2.8 patch adds request receipt/deadline transport,
native Ray Jobs deadline stopping and recovery, cancellation before payload
validation, and per-delivery UTF-8 decoding. The v2.5 patch remains historical.
All 13 package Python files were compared byte-for-byte with the rebuilt wheel.

On 2026-09-30, the boosting v2.8 ONNX schema patch records the fitted input
column order in the ONNX file itself as JSON `feature_names`, together with
`task_type`, `objective`, `framework`, `num_features` and classification
`num_classes`. These facts come from the export source and trained Booster;
converter metadata and all v2.7 conversion fixes are preserved. The incremental
patch was checked and replayed against the v2.7 wheel source, and the resulting
`native.py` matched the repacked wheel byte-for-byte. The wheel was repacked with
a regenerated RECORD; every payload hash and size was checked. Its input wheel
SHA-256 was `4e6addbeca1f912e514fd73442c43e7d1551fbd2d334e0f53629a28f12f2da27`.
The 11 real binary/multiclass/regression exports in
`tests/test_boosting_wheel_metadata.py` verify opaque result-column metadata,
checkpoint output names and ONNX Runtime predictions against XGBoost, including
constant/imbalanced data. The test loads the checked-in wheel source directly,
so an older installed Boosting package cannot mask a regression.

Rebuild and update this table whenever the source base or a local patch changes.
Verify that every wheel filename still ends in `py3-none-any.whl`.

## ray-clickhouse provenance limitation

This wheel must not be described as an unmodified build of the locked
`c162710...` reference. Its 13 package Python files match the existing local
`ray-clickhouse-auto-range` checkout at
`a5d6f99159aa7bc149ff2dbc463cec24dc56c7a2`, including uncommitted changes to
`src/ray_clickhouse/_schema.py` and `src/ray_clickhouse/datasink.py`.
The locked reference was unavailable in that checkout during this audit;
the complete portable patch chain and a byte-identical rebuild of this wheel
have not been established. The wheel hash above identifies the audited bytes.

## Licenses

The other wheels carry their license file inside the wheel archive. The
boosting wheel currently omits it, so its upstream Apache-2.0 license is kept
next to the wheel as `tributo-algorithms-LICENSE`.
