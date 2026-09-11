from importlib import import_module


def test_public_dependencies_are_importable() -> None:
    assert import_module("boto3") is not None
    assert import_module("ray_clickhouse") is not None
    assert import_module("tributo") is not None
    assert import_module("tributo_broker_redis") is not None
