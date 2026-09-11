# Air-gapped dependency wheels

These pure-Python wheels mirror the immutable Git sources recorded in
`uv.lock`. The Docker build skips network installation for these four packages
and installs the checked-in wheels instead; all ordinary third-party packages
still come from the configured Python package index.

| Wheel | Source commit | SHA-256 |
| --- | --- | --- |
| `ray_clickhouse-0.1.0-py3-none-any.whl` | `jiangxt2/ray-clickhouse@418f6c9608b61228986e6d247e604f36856307ba` | `55dc41f0699b5ee80beb871ef1718094db6faddeddbdfe42a57a8ea45327ad30` |
| `tributo-1.0.0-py3-none-any.whl` | `jiangxt2/Tributo@8cf821a22267e4972075d576eedacf3735f3d330` | `8095676906382aded3b31a9ea987d18d1cea2f5d131684a691862924321e949a` |
| `tributo_algorithms_boosting-0.1.0-py3-none-any.whl` | `jiangxt2/tributo-algorithms@71a6a16c3fd3676aba2225a0c528817634927fc7` (`packages/boosting`) | `d4ff909290fa1d2c46b92ff0a2151efcac8231d8ff01fc02f6675da1f5c17b24` |
| `tributo_broker_redis-0.1.0-py3-none-any.whl` | `Chang-Tong/tributo-broker-redis@41a193b7b9215e332d82b67abf72315e99d18b45` | `d2a6bb1a230506347ef5c85b08040597a2f082c1b13460b6f676c9d2f00ec11d` |

Rebuild the wheels with Python 3.13 whenever one of those locked commits
changes, update the hashes above, and verify that every filename still ends in
`py3-none-any.whl` before committing the result.

The other wheels carry their license file inside the wheel archive. The
boosting wheel currently omits it, so its upstream Apache-2.0 license is kept
next to the wheel as `tributo-algorithms-LICENSE`.
