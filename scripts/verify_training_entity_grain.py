"""Exercise training entity-grain enforcement on native Ray Data blocks.

Run with the same KnoVa/Core/Broker PYTHONPATH as the protocol unit suite.
No Redis, ClickHouse, model training, or business data is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import ray
from local_smoke import _training_request

from tributo_knova import training
from tributo_knova.protocol import TrainingExecutionRequest


class _Reporter:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def publish(self, event_type, payload, *, phase=None):
        self.events.append({"event_type": event_type, **payload, "phase": phase})


def verify() -> dict:
    payload = _training_request(
        SimpleNamespace(
            storage_mode="nfs",
            clickhouse_host="unused",
            clickhouse_port=8123,
            clickhouse_user="unused",
        ),
        job_id="entity-grain-check",
        table="unused",
        storage_root="/tmp/knova-entity-grain-check",
        storage_prefix="model/",
        password="unused",
    )
    payload["data_query"]["entity_key"] = {
        "origin": {
            "table_alias": "t0",
            "column_name": "entity_id",
            "column_type": "UInt64",
        },
        "result_column": "opaque_entity",
    }
    payload["data_query"]["query"]["sql"] = payload["data_query"]["query"][
        "sql"
    ].replace("SELECT ", "SELECT t0.entity_id AS opaque_entity, ")
    transform = training._build_execution(
        TrainingExecutionRequest.model_validate(payload)
    )[-1]
    reporter = _Reporter()
    transform.reporter = reporter
    ray.init(
        address="local", num_cpus=2, include_dashboard=False, logging_level="ERROR"
    )
    try:
        source = ray.data.from_items(
            [
                {
                    "opaque_entity": 1,
                    "t0__f1": 10,
                    "t0__f2": 100,
                    "t0__label": "churn",
                },
                {
                    "opaque_entity": 2,
                    "t0__f1": 20,
                    "t0__f2": 200,
                    "t0__label": "active",
                },
                {
                    "opaque_entity": 1,
                    "t0__f1": 10,
                    "t0__f2": 100,
                    "t0__label": "churn",
                },
            ],
            override_num_blocks=3,
        )
        rows = transform(source).take_all()
        assert len(rows) == 2, rows
        assert sorted(row["t0__label"] for row in rows) == [0, 1], rows
        assert all(set(row) == {"t0__f1", "t0__f2", "t0__label"} for row in rows)
        assert any(
            event["event_type"] == "LOG" and event["level"] == "warning"
            for event in reporter.events
        ), reporter.events
        return {
            "native_ray_entity_dedup": "PASS",
            "input_blocks": 3,
            "input_rows": 3,
            "output_rows": len(rows),
            "warning_count": len(reporter.events),
        }
    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify()
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
