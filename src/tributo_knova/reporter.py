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
      -- A fast driver may already have emitted a phase before admission's
      -- reporter writes. Admission must never roll that phase backwards.
      if ARGV[5] == 'ACCEPTED' then
        return -1
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
    "QUEUED": "INIT",
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


class ExecutionControlError(Exception):
    """Sanitized driver-side deadline/cancellation checkpoint outcome."""

    def __init__(self, code: str, phase: str) -> None:
        self.code = code
        self.phase = phase
        super().__init__(
            "execution deadline exceeded"
            if code != "CANCELLED"
            else "execution cancelled"
        )


_INVALID_CODES = frozenset(
    {
        "INVALID_JSON",
        "INVALID_REQUEST",
        "INVALID_ENVELOPE",
        "INVALID_OPERATION_ID",
        "IDENTITY_MISMATCH",
        "OPERATION_CHANNEL_MISMATCH",
        "PAYLOAD_TOO_LARGE",
        "DRIVER_INPUT_TOO_LARGE",
        "UNSUPPORTED_EXECUTION_PROFILE",
    }
)
_INFERENCE_CODES = frozenset(
    {
        "UNSUPPORTED_PROTOCOL_VERSION",
        "INVALID_PAYLOAD",
        "MODEL_ARTIFACT_NOT_FOUND",
        "MODEL_ARTIFACT_CORRUPTED",
        "MODEL_LOAD_FAILED",
        "MODEL_FORMAT_UNSUPPORTED",
        "MODEL_SCHEMA_MISMATCH",
        "DATA_SOURCE_UNREACHABLE",
        "INPUT_QUERY_FAILED",
        "INPUT_COLUMN_MISSING",
        "EMPTY_INPUT",
        "PREDICTION_FAILED",
        "RESULT_WRITE_FAILED",
        "EXECUTION_TIMEOUT",
        "UNKNOWN",
    }
)
_TRAINING_CODES = frozenset(
    {
        "UNSUPPORTED_PROTOCOL_VERSION",
        "INVALID_PAYLOAD",
        "DATA_SOURCE_UNREACHABLE",
        "DATA_QUERY_FAILED",
        "EMPTY_DATASET",
        "FEATURE_ENGINEERING_FAILED",
        "TRAINING_FAILED",
        "TRAINING_OOM",
        "TRAINING_TIMEOUT",
        "EVALUATION_FAILED",
        "MODEL_EXPORT_FAILED",
        "ARTIFACT_UPLOAD_FAILED",
        "UNKNOWN",
    }
)


class KnovaRedisEventReporter(RedisEventReporter):
    """Serialize Broker lifecycle calls as existing KnoVa v2 events."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._started_at = time.monotonic()
        self._terminal = False
        self._snapshot: dict[str, Any] = {}
        self._current_phase = (
            "QUEUED" if self._identity["operation_type"] == "training" else "INIT"
        )
        self._restore_snapshot()

    def _restore_snapshot(self) -> None:
        """Recover driver progress when a watcher must publish a terminal event."""
        reader = getattr(self._redis, "xrevrange", None)
        if not callable(reader):
            return
        for _entry_id, fields in reader(self._stream, count=self._max_stream_length):
            raw = fields.get("payload", fields.get(b"payload"))
            try:
                event = json.loads(raw)
            except (TypeError, ValueError, UnicodeError):
                continue
            if not isinstance(event, dict):
                continue
            identity_key = (
                "job_id"
                if self._identity["operation_type"] == "training"
                else "execution_id"
            )
            if event.get(identity_key) != self._identity["operation_id"]:
                continue
            if not self._snapshot and event.get("phase"):
                self._current_phase = event["phase"]
            for key in (
                "processed_rows",
                "result_rows",
                "total_rows",
                "percent",
                "current_round",
                "total_rounds",
                "progress_percent",
                "phase",
            ):
                if key in event:
                    self._snapshot.setdefault(key, event[key])
            if event.get("event_type") in _TERMINAL_EVENTS:
                self._terminal = True

    def check_control(self, phase: str | None = None) -> None:
        """Check driver-owned checkpoints; the Consumer also stops native Ray Jobs."""
        if phase:
            phases = (
                _TRAINING_PHASES
                if self._identity["operation_type"] == "training"
                else _INFERENCE_PHASES
            )
            self._current_phase = phases.get(phase, phase)
        if self._terminal:
            return
        if self._cancel_key and bool(self._redis.exists(self._cancel_key)):
            raise ExecutionControlError("CANCELLED", self._current_phase)
        if (
            self._deadline_at_ms is not None
            and time.time() * 1000 >= self._deadline_at_ms
        ):
            code = (
                "TRAINING_TIMEOUT"
                if self._identity["operation_type"] == "training"
                else "EXECUTION_TIMEOUT"
            )
            raise ExecutionControlError(code, self._current_phase)

    def configure_timeout(self, seconds: float | None) -> None:
        """Apply request limits for direct execution as well as broker drivers."""
        if seconds is not None:
            received = self._received_at_ms
            if received is None:
                received = int(
                    time.time() * 1000 - (time.monotonic() - self._started_at) * 1000
                )
                self._received_at_ms = received
            deadline = received + int(seconds * 1000)
            self._deadline_at_ms = (
                min(self._deadline_at_ms, deadline)
                if self._deadline_at_ms is not None
                else deadline
            )

    def _error_code(self, code: str, phase: str) -> str:
        if code in _INVALID_CODES:
            return "INVALID_PAYLOAD"
        training = self._identity["operation_type"] == "training"
        if code == "EXECUTION_TIMEOUT" and training:
            return "TRAINING_TIMEOUT"
        allowed = _TRAINING_CODES if training else _INFERENCE_CODES
        if code in allowed:
            return code
        if code in {
            "EXECUTION_FAILED",
            "RAY_JOB_FAILED",
            "RAY_JOB_STOPPED",
            "TERMINAL_EVENT_MISSING",
        }:
            return (
                {
                    "LOADING_DATA": "DATA_QUERY_FAILED",
                    "EVALUATING": "EVALUATION_FAILED",
                    "EXPORTING": "MODEL_EXPORT_FAILED",
                }.get(phase, "TRAINING_FAILED")
                if training
                else {
                    "LOAD_MODEL": "MODEL_LOAD_FAILED",
                    "READ_INPUT": "INPUT_QUERY_FAILED",
                    "WRITE_RESULT": "RESULT_WRITE_FAILED",
                }.get(phase, "PREDICTION_FAILED")
            )
        return "UNKNOWN"

    def publish(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        phase: str | None = None,
    ) -> dict[str, Any]:
        if event_type == "COMPLETED":
            self.check_control()
        safe_payload = redact({**self._snapshot, **(payload or {})})
        if event_type in {"FAILED", "CANCELLED"} and phase in {
            None,
            "FAILED",
            "CANCELLED",
            "EXECUTING",
        }:
            phase = self._current_phase
        operation_type = self._identity["operation_type"]
        if operation_type == "training":
            event = self._training_event(event_type, safe_payload, phase)
        else:
            event = self._inference_event(event_type, safe_payload, phase)
        if event_type == "FAILED":
            event["error_code"] = self._error_code(event["error_code"], event["phase"])

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
        if event_type == "ACCEPTED" and self._snapshot:
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
                event_type,
            )
            if result == -1:
                return event
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
        else:
            self._current_phase = event.get("phase", self._current_phase)
            for key in (
                "processed_rows",
                "result_rows",
                "total_rows",
                "percent",
                "current_round",
                "total_rounds",
                "progress_percent",
                "phase",
            ):
                if key in event:
                    self._snapshot[key] = event[key]
        return event

    def metrics(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Project the public training reporter callback into a KnoVa event."""
        self.check_control("TRAINING")
        return self.publish("METRICS", payload, phase="EXECUTING")

    def phase(self, phase: str) -> dict[str, Any]:
        self.check_control(phase)
        return super().phase(phase)

    def _training_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        phase: str | None,
    ) -> dict[str, Any]:
        identity = self._identity["operation_id"]
        current_phase = _TRAINING_PHASES.get(phase or "", phase or self._current_phase)
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
        current_phase = _INFERENCE_PHASES.get(phase or "", phase or self._current_phase)
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
        if event_type == "STARTED":
            return {
                **base,
                "total_rows": int(payload.get("total_rows", 0)),
                "message": str(payload.get("message", "Inference execution started")),
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
        if self._received_at_ms is not None:
            return round(max(0.0, time.time() - self._received_at_ms / 1000), 3)
        return round(time.monotonic() - self._started_at, 3)

    @staticmethod
    def _present(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
        return {key: payload[key] for key in keys if payload.get(key) is not None}


__all__ = ["ExecutionControlError", "KnovaRedisEventReporter"]
