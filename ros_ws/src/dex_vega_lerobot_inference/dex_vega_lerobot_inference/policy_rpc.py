"""Version-neutral Unix-socket protocol between ROS Python and LeRobot Python."""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ACTION_CHUNK_SIZE, ACTION_DIMENSION, STATE_DIMENSION, TASK
from .observation_adapter import ObservationSnapshot
from .policy_runtime import PolicyPrediction, PolicyTimings


_LENGTH = struct.Struct("!Q")
_MAX_HEADER_BYTES = 1024 * 1024
_RGB_BYTES = 480 * 640 * 3
_ACTION_BYTES = ACTION_CHUNK_SIZE * ACTION_DIMENSION * 8
_PROTOCOL_VERSION = 2


class PolicyRpcError(RuntimeError):
    """Raised on policy-server connection, protocol, or remote-runtime failure."""


@dataclass(frozen=True)
class ExternalRuntimeInfo:
    """Serializable subset of the policy server's pinned runtime identity."""

    lerobot_version: str
    torch_version: str
    cuda_version: str | None
    device_name: str
    model_path: str
    model_commit: str | None
    checkpoint_tag: str | None
    tokenizer_path: str
    tokenizer_commit: str | None
    load_seconds: float
    policy_type: str = "pi05"
    action_chunk_size: int = ACTION_CHUNK_SIZE
    action_dimension: int = ACTION_DIMENSION
    base_model_path: str = ""
    base_model_commit: str | None = None
    processor_path: str = ""
    processor_commit: str | None = None


class PolicyRuntimeClient:
    """Implement the inference-worker runtime protocol through a local socket."""

    def __init__(self, socket_path: str | Path, timeout_seconds: float = 120.0) -> None:
        self._socket_path = str(Path(socket_path).expanduser().resolve())
        self._timeout_seconds = float(timeout_seconds)
        if self._timeout_seconds <= 0.0:
            raise ValueError("policy server timeout must be positive")
        response, _ = self._request({"operation": "info"})
        if response.get("protocol_version") != _PROTOCOL_VERSION:
            raise PolicyRpcError(
                "policy server protocol is outdated; restart it from this workspace build"
            )
        try:
            self.info = ExternalRuntimeInfo(**response["runtime_info"])
        except (KeyError, TypeError) as exc:
            raise PolicyRpcError("policy server returned invalid runtime identity") from exc
        if self.info.action_chunk_size <= 0 or self.info.action_chunk_size > 256:
            raise PolicyRpcError("policy server returned an invalid action chunk size")
        if self.info.action_dimension != ACTION_DIMENSION:
            raise PolicyRpcError(
                "policy server action dimension differs from the 27-value ROS contract"
            )

    def predict(self, observation: ObservationSnapshot) -> PolicyPrediction:
        """Send one exact state/RGB/task observation and receive a physical chunk."""
        header = {
            "operation": "predict",
            "task": observation.task,
            "state": observation.state.astype(np.float32, copy=False).tolist(),
            "state_stamp_ns": observation.state_stamp_ns,
            "camera_stamp_ns": observation.camera_stamp_ns,
            "receive_stamp_ns": observation.receive_stamp_ns,
            "created_stamp_ns": observation.created_stamp_ns,
            "created_monotonic_ns": observation.created_monotonic_ns,
            "state_age_seconds": observation.state_age_seconds,
            "camera_capture_age_seconds": observation.camera_capture_age_seconds,
            "camera_receive_age_seconds": observation.camera_receive_age_seconds,
            "synchronization_skew_seconds": observation.synchronization_skew_seconds,
            "payload_bytes": _RGB_BYTES,
        }
        response, payload = self._request(header, observation.rgb.tobytes(order="C"))
        self._validate_response_identity(response)
        expected_action_bytes = (
            self.info.action_chunk_size * self.info.action_dimension * 8
        )
        if len(payload) != expected_action_bytes:
            raise PolicyRpcError(
                "policy server action payload has "
                f"{len(payload)} bytes, expected {expected_action_bytes}"
            )
        actions = np.frombuffer(payload, dtype="<f8").reshape(
            self.info.action_chunk_size, self.info.action_dimension
        )
        try:
            timings = PolicyTimings(**response["timings"])
            return PolicyPrediction(
                actions=actions.copy(),
                timings=timings,
                completed_monotonic_ns=int(response["completed_monotonic_ns"]),
                peak_gpu_allocated_bytes=int(response["peak_gpu_allocated_bytes"]),
                peak_gpu_reserved_bytes=int(response["peak_gpu_reserved_bytes"]),
                cold_start=bool(response["cold_start"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PolicyRpcError("policy server returned invalid prediction metadata") from exc

    def reset(self) -> None:
        """Reset model and processor queues in the policy-server process."""
        response, _ = self._request({"operation": "reset"})
        self._validate_response_identity(response)

    def _validate_response_identity(self, response: dict[str, Any]) -> None:
        if response.get("protocol_version") != _PROTOCOL_VERSION:
            raise PolicyRpcError("policy server response has an incompatible protocol")
        try:
            current = ExternalRuntimeInfo(**response["runtime_info"])
        except (KeyError, TypeError) as exc:
            raise PolicyRpcError(
                "policy server response omitted its runtime identity"
            ) from exc
        if current != self.info:
            raise PolicyRpcError(
                "policy server identity changed; restart the ROS inference node"
            )

    def _request(
        self, header: dict[str, Any], payload: bytes = b""
    ) -> tuple[dict[str, Any], bytes]:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(self._timeout_seconds)
                client.connect(self._socket_path)
                _send_message(client, header, payload)
                response, response_payload = _receive_message(client)
        except (OSError, TimeoutError) as exc:
            raise PolicyRpcError(
                f"cannot communicate with policy server at {self._socket_path}: {exc}"
            ) from exc
        if response.get("status") != "ok":
            raise PolicyRpcError(str(response.get("error", "unknown policy server error")))
        return response, response_payload


def serve_request(runtime: Any, header: dict[str, Any], payload: bytes) -> tuple[dict, bytes]:
    """Execute one validated RPC operation; used by the server and protocol tests."""
    operation = header.get("operation")
    if operation == "info":
        return {
            "status": "ok",
            "protocol_version": _PROTOCOL_VERSION,
            "runtime_info": asdict(runtime.info),
        }, b""
    if operation == "reset":
        runtime.reset()
        return {
            "status": "ok",
            "protocol_version": _PROTOCOL_VERSION,
            "runtime_info": asdict(runtime.info),
        }, b""
    if operation != "predict":
        raise PolicyRpcError(f"unknown policy RPC operation: {operation!r}")
    if header.get("task") != TASK:
        raise PolicyRpcError("policy RPC task differs from the exact training task")
    state = np.asarray(header.get("state"), dtype=np.float32)
    if state.shape != (STATE_DIMENSION,) or not np.all(np.isfinite(state)):
        raise PolicyRpcError("policy RPC state must contain 27 finite values")
    if len(payload) != _RGB_BYTES:
        raise PolicyRpcError(
            f"policy RPC RGB payload has {len(payload)} bytes, expected {_RGB_BYTES}"
        )
    rgb = np.frombuffer(payload, dtype=np.uint8).reshape(480, 640, 3).copy()
    try:
        observation = ObservationSnapshot(
            state=state,
            rgb=rgb,
            task=TASK,
            state_stamp_ns=int(header["state_stamp_ns"]),
            camera_stamp_ns=int(header["camera_stamp_ns"]),
            receive_stamp_ns=int(header["receive_stamp_ns"]),
            created_stamp_ns=int(header["created_stamp_ns"]),
            created_monotonic_ns=int(header["created_monotonic_ns"]),
            state_age_seconds=float(header["state_age_seconds"]),
            camera_capture_age_seconds=float(header["camera_capture_age_seconds"]),
            camera_receive_age_seconds=float(header["camera_receive_age_seconds"]),
            synchronization_skew_seconds=float(header["synchronization_skew_seconds"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyRpcError("policy RPC observation metadata is incomplete") from exc
    prediction = runtime.predict(observation)
    action_chunk_size = int(
        getattr(runtime.info, "action_chunk_size", ACTION_CHUNK_SIZE)
    )
    action_dimension = int(getattr(runtime.info, "action_dimension", ACTION_DIMENSION))
    if action_chunk_size <= 0 or action_chunk_size > 256:
        raise PolicyRpcError("policy runtime declared an invalid action chunk size")
    if action_dimension != ACTION_DIMENSION:
        raise PolicyRpcError("policy runtime declared a non-27-D physical action")
    actions = np.asarray(prediction.actions, dtype="<f8")
    if actions.shape != (action_chunk_size, action_dimension):
        raise PolicyRpcError(f"policy runtime returned invalid action shape {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise PolicyRpcError("policy runtime returned non-finite actions")
    response = {
        "status": "ok",
        "protocol_version": _PROTOCOL_VERSION,
        "runtime_info": asdict(runtime.info),
        "timings": asdict(prediction.timings),
        "completed_monotonic_ns": prediction.completed_monotonic_ns,
        "peak_gpu_allocated_bytes": prediction.peak_gpu_allocated_bytes,
        "peak_gpu_reserved_bytes": prediction.peak_gpu_reserved_bytes,
        "cold_start": prediction.cold_start,
        "action_chunk_size": action_chunk_size,
        "action_dimension": action_dimension,
    }
    return response, actions.tobytes(order="C")


def _send_message(connection: socket.socket, header: dict[str, Any], payload: bytes) -> None:
    try:
        encoded = json.dumps(
            header,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PolicyRpcError("policy RPC header is not finite JSON") from exc
    if len(encoded) > _MAX_HEADER_BYTES:
        raise PolicyRpcError("policy RPC header exceeds size limit")
    connection.sendall(_LENGTH.pack(len(encoded)))
    connection.sendall(encoded)
    if payload:
        connection.sendall(payload)


def _receive_message(connection: socket.socket) -> tuple[dict[str, Any], bytes]:
    header_length = _LENGTH.unpack(_receive_exact(connection, _LENGTH.size))[0]
    if header_length > _MAX_HEADER_BYTES:
        raise PolicyRpcError("policy RPC header exceeds size limit")
    try:
        header = json.loads(
            _receive_exact(connection, header_length).decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PolicyRpcError("policy RPC header is not valid JSON") from exc
    if not isinstance(header, dict):
        raise PolicyRpcError("policy RPC header must be a JSON object")
    raw_payload_length = header.get("payload_bytes", 0)
    if isinstance(raw_payload_length, bool) or not isinstance(
        raw_payload_length, int
    ):
        raise PolicyRpcError("policy RPC payload length must be an integer")
    payload_length = raw_payload_length
    if payload_length < 0 or payload_length > max(_RGB_BYTES, _ACTION_BYTES):
        raise PolicyRpcError("policy RPC payload length is invalid")
    return header, _receive_exact(connection, payload_length) if payload_length else b""


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        block = connection.recv(size - len(data))
        if not block:
            raise PolicyRpcError("policy RPC connection closed before message completed")
        data.extend(block)
    return bytes(data)


def _reject_json_constant(value: str) -> None:
    """Reject JSON's non-standard NaN and infinity extensions."""
    raise ValueError(f"non-finite JSON constant {value}")
