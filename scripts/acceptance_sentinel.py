"""Real Sentinel authentication, endpoint fallback and failover probes."""

import json
import os
import time
from pathlib import Path

from tributo_broker_redis.config import RedisConnectionConfig
from tributo_broker_redis.redis_client import create_redis_client

os.environ["KNOVA_REDIS_PASSWORD"] = "local-repro-only"
os.environ["KNOVA_REDIS_SENTINEL_PASSWORD"] = "local-sentinel-only"
OUT = Path("/artifacts")
results = []
config = {
    "mode": "sentinel",
    "sentinel_nodes": [f"accept-sentinel-{i}:26379" for i in range(1, 4)],
    "sentinel_master": "mymaster",
    "database": 3,
    "password_ref": "env:KNOVA_REDIS_PASSWORD",
    "sentinel_password_ref": "env:KNOVA_REDIS_SENTINEL_PASSWORD",
}


def case(name, fn):
    try:
        fn()
        results.append({"case": name, "passed": True})
    except Exception as exc:
        results.append({"case": name, "passed": False, "exception": type(exc).__name__})
    (OUT / "sentinel-results.json").write_text(json.dumps(results, indent=2))


def ping(value):
    client = create_redis_client(RedisConnectionConfig.model_validate(value))
    try:
        assert client.ping()
        client.set("accept:sentinel", "ok", ex=300)
        assert client.get("accept:sentinel") == "ok"
    finally:
        client.close()


def rejects(value):
    try:
        ping(value)
    except Exception:
        return
    raise AssertionError("invalid credentials were accepted")


case("sentinel_auth_and_db3", lambda: ping(config))
case(
    "one_dead_sentinel_fallback",
    lambda: ping(
        {**config, "sentinel_nodes": ["accept-sentinel-1:1", "accept-sentinel-2:26379"]}
    ),
)
os.environ["ACCEPT_WRONG_SECRET"] = "wrong-only-test"
case(
    "sentinel_bad_password",
    lambda: rejects({**config, "sentinel_password_ref": "env:ACCEPT_WRONG_SECRET"}),
)
case(
    "redis_bad_password",
    lambda: rejects({**config, "password_ref": "env:ACCEPT_WRONG_SECRET"}),
)
case(
    "sentinel_missing_secret",
    lambda: rejects({**config, "sentinel_password_ref": "env:ACCEPT_MISSING_SECRET"}),
)
case(
    "redis_missing_secret",
    lambda: rejects({**config, "password_ref": "env:ACCEPT_MISSING_SECRET"}),
)
case(
    "wrong_master_name",
    lambda: rejects({**config, "sentinel_master": "does-not-exist"}),
)
(OUT / "sentinel-ready").touch()
# Host orchestrator stops only the isolated master after seeing this marker.
deadline = time.monotonic() + 90
while not (OUT / "sentinel-master-stopped").exists():
    if time.monotonic() > deadline:
        raise TimeoutError("host did not initiate failover")
    time.sleep(0.2)


def failover():
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        try:
            ping(config)
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("Sentinel did not recover")


case("master_failover", failover)
if any(not result["passed"] for result in results):
    raise SystemExit(1)
