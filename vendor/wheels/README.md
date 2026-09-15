# Air-gapped dependency wheels

These pure-Python wheels mirror the immutable Git sources recorded in
`uv.lock`. The Docker build skips network installation for these four packages
and installs the checked-in wheels instead; all ordinary third-party packages
still come from the configured Python package index.

| Wheel | Source commit | SHA-256 |
| --- | --- | --- |
| `ray_clickhouse-0.1.0-py3-none-any.whl` | `Chang-Tong/ray-clickhouse-upstream@c1627101736463ccc94c99c6793f91360173e6ac` | `422495c3d89555c641fafcbcbd4383513402b4d6973c0e575de51070a694c1a4` |
| `tributo-1.0.0-py3-none-any.whl` | `Chang-Tong/Tributo@440da0422a3a9c5b041b6ecdbb2e40c00bf43cd90` | `38c095e3812deabc16280281fb4ab06b5c4d53f09f94d751af945b8b76dc13f2` |
| `tributo_algorithms_boosting-0.1.0-py3-none-any.whl` | `Chang-Tong/tributo-algorithms@6f39378d4d850cfd41e9510abc313efc6d4f209f` (`packages/boosting`) | `78ecd95a05f8f33ea546fe9b9496c9d4a46eb85ed5f4f9b6a207132861cb45d7` |
| `tributo_broker_redis-0.1.0-py3-none-any.whl` | `Chang-Tong/tributo-broker-redis@05b4ef6ca557ed3b6373ec9cfc9cc1be3648330bd` | `8db95dc7e72ddf3153cdf3b670782a4bcca5865b39a359b24a2dd737bcdd9fbe` |

Rebuild the wheels with Python 3.13 whenever one of those locked commits
changes, update the hashes above, and verify that every filename still ends in
`py3-none-any.whl` before committing the result.

The other wheels carry their license file inside the wheel archive. The
boosting wheel currently omits it, so its upstream Apache-2.0 license is kept
next to the wheel as `tributo-algorithms-LICENSE`.
