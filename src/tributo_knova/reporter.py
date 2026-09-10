"""KnoVa protocol v2 event serialization for the Redis broker runtime."""

from __future__ import annotations

import json
import time
from typing import Any

from tributo_broker_redis.reporter import RedisEventReporter, redact

_TERMINAL_EVENTS = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
_ATOMIC_EVENT_XADD_LUA = r"""
local latest = redis.call('XREVRANGE', KEYS[1], '+', '-', 'COUNT', 1)
if #latest > 0 then
  local fields = latest[1][2]
  for i = 1, #fields, 2 do
    if fields[i] == 'payload' then
      local ok, envelope = pcall(cjson.decode, fields[i + 1])
      if ok and (envelope['event_type'] == 'COMPLETED'
          or envelope['event_type'] == 'FAILED'
          or envelope['event_type'] == 'CANCELLED') then
        return false
      end
      break
    end
  end
end
return redis.call(
  'XADD', KEYS[1], 'MAXLEN', ARGV[4], '*',
  ARGV[1], ARGV[2], 'payload', ARGV[3]
)
"""

_TRAINING_PHASES = {
    "ADMISSION": "QUEUED",
    "ADMITTED": "QUEUED",
    "PREPARING": "LOADING_DATA",
    "EXECUTING": "TRAINING",
    "MATERIALIZING": "EVALUATING",
    "PUBLISHING": "EVALUATING",
    "COMPLETED": "COMPLETED",
    "FAILED": "FAILED",
    "CANCELLED": "CANCELLED",
}
_INFERENCE_PHASES = {
    "ADMISSION": "INIT",
    "ADMITTED": "INIT",
    "PREPARING": "INIT",
    "EXECUTING": "PREDICT",
    "MATERIALIZING": "WRITE_RESULT",
    "PUBLISHING": "CLEANUP",
    "COMPLETED": "CLEANUP",
    "FAILED": "INIT",
    "CANCELLED": "PREDICT",
}


class KnovaRedisEventReporter(RedisEventReporter):
    """Serialize Broker lifecycle calls as existing KnoVa v2 events."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._started_at = time.monotonic()
        self._terminal = False

    def publish(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        phase: str | None = None,
    ) -> dict[str, Any]:
        safe_payload = redact(payload or {})
        operation_type = self._identity["operation_type"]
        if operation_type == "training":
            event = self._training_event(event_type, safe_payload, phase)
        else:
            event = self._inference_event(event_type, safe_payload, phase)

        encoded = json.dumps(
            event,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(encoded.encode("utf-8")) > self._max_event_bytes:
            raise ValueError("event exceeds configured size limit")
        if self._terminal:
            return event

        terminal = event["event_type"] in _TERMINAL_EVENTS
        evaluate = getattr(self._redis, "eval", None)
        if callable(evaluate):
            result = evaluate(
                _ATOMIC_EVENT_XADD_LUA,
                1,
                self._stream,
                self._outer_identity_field,
                self._identity["operation_id"],
                encoded,
                str(self._max_stream_length),
            )
            if result is False or result is None:
                self._terminal = True
                return event
        else:
            self._redis.xadd(
                self._stream,
                {
                    self._outer_identity_field: self._identity["operation_id"],
                    "payload": encoded,
                },
                maxlen=self._max_stream_length,
                approximate=True,
            )
        if terminal:
            self._terminal = True
        return event

    def _training_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        phase: str | None,
    ) -> dict[str, Any]:
        identity = self._identity["operation_id"]
        current_phase = _TRAINING_PHASES.get(phase or "", phase or "TRAINING")
        base = {
            "protocol_version": "2.0",
            "event_type": event_type,
            "job_id": identity,
            "timestamp": int(time.time() * 1000),
            "phase": current_phase,
        }
        if event_type == "ACCEPTED":
            return {
                **base,
                "event_type": "PHASE",
                "phase": "QUEUED",
                "message": "Ray job admitted",
            }
        if event_type == "PHASE":
            return {**base, "message": str(payload.get("message", ""))}
        if event_type == "LOG":
            return {
                **base,
                "level": str(payload.get("level", "INFO")).lower(),
                "message": str(payload.get("message", ""))[:4096],
            }
        if event_type == "METRICS":
            return {
                **base,
                "current_round": int(payload.get("current_round", 0)),
                "total_rounds": int(payload.get("total_rounds", 0)),
                "progress_percent": float(payload.get("progress_percent", 0.0)),
                "metrics": list(payload.get("metrics", [])),
            }
        if event_type == "COMPLETED":
            return {
                **base,
                "phase": "COMPLETED",
                "progress_percent": 100,
                "duration_seconds": self._duration(),
                "result_summary": payload.get("result_summary", {}),
                **self._present(payload, "training_result", "artifact_manifest"),
            }
        if event_type == "CANCELLED":
            return {
                **base,
                "has_best_model": bool(payload.get("has_best_model", False)),
                "duration_seconds": self._duration(),
            }
        if event_type == "FAILED":
            return {
                **base,
                "error_code": str(payload.get("error_code", "UNKNOWN")),
                "error_message": str(
                    payload.get("sanitized_message", "operation failed")
                ),
                "duration_seconds": self._duration(),
            }
        return {**base, "event_type": "LOG", "level": "info", "message": ""}

    def _inference_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        phase: str | None,
    ) -> dict[str, Any]:
        identity = self._identity["operation_id"]
        current_phase = _INFERENCE_PHASES.get(phase or "", phase or "INIT")
        base = {
            "protocol_version": "2.0",
            "event_type": event_type,
            "execution_id": identity,
            "timestamp": int(time.time() * 1000),
            "phase": current_phase,
        }
        if event_type == "ACCEPTED":
            return {
                **base,
                "event_type": "LOG",
                "level": "info",
                "message": "Ray job admitted",
            }
        if event_type == "LOG":
            return {
                **base,
                "level": str(payload.get("level", "INFO")).lower(),
                "message": str(payload.get("message", ""))[:4096],
            }
        if event_type in {"PHASE", "PROGRESS"}:
            return {
                **base,
                "event_type": "PROGRESS",
                "processed_rows": int(payload.get("processed_rows", 0)),
                "total_rows": int(payload.get("total_rows", 0)),
                "result_rows": int(payload.get("result_rows", 0)),
                "percent": float(payload.get("percent", 0.0)),
            }
        if event_type == "COMPLETED":
            return {
                **base,
                "phase": "CLEANUP",
                "processed_rows": int(payload.get("processed_rows", 0)),
                "result_rows": int(payload.get("result_rows", 0)),
                "total_rows": int(payload.get("total_rows", 0)),
                "percent": 100.0,
                "duration_seconds": self._duration(),
                "message": str(payload.get("message", "")),
            }
        if event_type == "CANCELLED":
            return {
                **base,
                "processed_rows": int(payload.get("processed_rows", 0)),
                "result_rows": int(payload.get("result_rows", 0)),
                "total_rows": int(payload.get("total_rows", 0)),
                "percent": float(payload.get("percent", 0.0)),
                "stop_reason": str(payload.get("stop_reason", "CANCELLED")),
                "message": str(payload.get("message", "")),
                "duration_seconds": self._duration(),
            }
        if event_type == "FAILED":
            return {
                **base,
                "error_code": str(payload.get("error_code", "UNKNOWN")),
                "error_message": str(
                    payload.get("sanitized_message", "operation failed")
                ),
                "processed_rows": int(payload.get("processed_rows", 0)),
                "result_rows": int(payload.get("result_rows", 0)),
                "duration_seconds": self._duration(),
            }
        return {**base, "event_type": "LOG", "level": "info", "message": ""}

    def _duration(self) -> float:
        return round(time.monotonic() - self._started_at, 3)

    @staticmethod
    def _present(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
        return {key: payload[key] for key in keys if payload.get(key) is not None}


__all__ = ["KnovaRedisEventReporter"]
