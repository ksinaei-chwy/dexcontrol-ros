from dataclasses import asdict, dataclass
import time

import numpy as np
import pytest

from dex_vega_lerobot_inference.policy_rpc import (
    ExternalRuntimeInfo,
    PolicyRpcError,
    PolicyRuntimeClient,
    _receive_message,
    _send_message,
    serve_request,
)
from dex_vega_lerobot_inference.policy_server import _prepare_socket_path
from dex_vega_lerobot_inference.policy_runtime import PolicyPrediction, PolicyTimings


@dataclass(frozen=True)
class _Info:
    lerobot_version: str = "0.6.0"
    torch_version: str = "test"
    cuda_version: str = "test"
    device_name: str = "cuda"
    model_path: str = "/project/model"
    model_commit: str = "a" * 40
    checkpoint_tag: str = "step-005000"
    tokenizer_path: str = "/project/tokenizer"
    tokenizer_commit: str = "b" * 40
    load_seconds: float = 1.0
    policy_type: str = "pi05"
    action_chunk_size: int = 50
    action_dimension: int = 27


class _Runtime:
    info = _Info()

    def __init__(self):
        self.reset_count = 0

    def predict(self, observation):
        return PolicyPrediction(
            actions=np.zeros((50, 27), dtype=np.float64),
            timings=PolicyTimings(0.1, 0.2, 0.3, 0.1, 0.7, 0.8),
            completed_monotonic_ns=time.monotonic_ns(),
            peak_gpu_allocated_bytes=10,
            peak_gpu_reserved_bytes=20,
            cold_start=False,
        )

    def reset(self):
        self.reset_count += 1


class _GrootRuntime(_Runtime):
    info = _Info(
        checkpoint_tag="step-034000",
        tokenizer_path="",
        tokenizer_commit=None,
        policy_type="groot",
        action_chunk_size=40,
    )

    def predict(self, observation):
        prediction = super().predict(observation)
        return PolicyPrediction(
            actions=np.zeros((40, 27), dtype=np.float64),
            timings=prediction.timings,
            completed_monotonic_ns=prediction.completed_monotonic_ns,
            peak_gpu_allocated_bytes=prediction.peak_gpu_allocated_bytes,
            peak_gpu_reserved_bytes=prediction.peak_gpu_reserved_bytes,
            cold_start=prediction.cold_start,
        )


class _MemoryConnection:
    def __init__(self):
        self.data = bytearray()
        self.offset = 0

    def sendall(self, block):
        self.data.extend(block)

    def recv(self, size):
        block = bytes(self.data[self.offset:self.offset + size])
        self.offset += len(block)
        return block


class _ExistingSocketPath:
    def __init__(self):
        self.unlinked = False

    def exists(self):
        return True

    def is_socket(self):
        return True

    def unlink(self):
        self.unlinked = True

    def __str__(self):
        return "/project/existing-policy.sock"


def _request():
    return {
        "operation": "predict",
        "task": "put the blue bird on the meeting desk",
        "state": [0.0] * 27,
        "state_stamp_ns": 1,
        "camera_stamp_ns": 1,
        "receive_stamp_ns": 1,
        "created_stamp_ns": 1,
        "created_monotonic_ns": time.monotonic_ns(),
        "state_age_seconds": 0.0,
        "camera_capture_age_seconds": 0.0,
        "camera_receive_age_seconds": 0.0,
        "synchronization_skew_seconds": 0.0,
    }


def test_rpc_preserves_50_by_27_physical_chunk():
    response, payload = serve_request(
        _Runtime(),
        _request(),
        np.zeros((480, 640, 3), dtype=np.uint8).tobytes(),
    )
    assert response["status"] == "ok"
    assert len(payload) == 50 * 27 * 8
    assert np.frombuffer(payload, dtype="<f8").reshape(50, 27).shape == (50, 27)


def test_rpc_preserves_groot_40_by_27_physical_chunk():
    response, payload = serve_request(
        _GrootRuntime(),
        _request(),
        np.zeros((480, 640, 3), dtype=np.uint8).tobytes(),
    )
    assert response["action_chunk_size"] == 40
    assert response["action_dimension"] == 27
    assert len(payload) == 40 * 27 * 8


def test_rpc_rejects_task_or_state_contract_changes():
    request = _request()
    request["task"] = "similar but wrong task"
    with pytest.raises(PolicyRpcError, match="task"):
        serve_request(_Runtime(), request, bytes(480 * 640 * 3))
    request = _request()
    request["state"] = [0.0] * 32
    with pytest.raises(PolicyRpcError, match="27"):
        serve_request(_Runtime(), request, bytes(480 * 640 * 3))


def test_rpc_reset_is_explicit():
    runtime = _Runtime()
    response, payload = serve_request(runtime, {"operation": "reset"}, b"")
    assert response == {
        "status": "ok",
        "protocol_version": 2,
        "runtime_info": asdict(runtime.info),
    }
    assert payload == b""
    assert runtime.reset_count == 1


def test_client_rejects_policy_server_identity_change():
    client = object.__new__(PolicyRuntimeClient)
    client.info = ExternalRuntimeInfo(**asdict(_Info()))
    response = {"protocol_version": 2, "runtime_info": asdict(_Info())}
    client._validate_response_identity(response)

    changed = asdict(_Info())
    changed["model_commit"] = "c" * 40
    with pytest.raises(PolicyRpcError, match="identity changed"):
        client._validate_response_identity(
            {"protocol_version": 2, "runtime_info": changed}
        )


def test_rpc_rejects_nonfinite_actions_and_headers():
    runtime = _Runtime()
    original_predict = runtime.predict

    def nonfinite_predict(observation):
        prediction = original_predict(observation)
        prediction.actions[0, 0] = np.nan
        return prediction

    runtime.predict = nonfinite_predict
    with pytest.raises(PolicyRpcError, match="non-finite actions"):
        serve_request(runtime, _request(), bytes(480 * 640 * 3))

    connection = _MemoryConnection()
    with pytest.raises(PolicyRpcError, match="finite JSON"):
        _send_message(connection, {"value": float("nan")}, b"")
    _send_message(connection, {"payload_bytes": 0.5}, b"")
    with pytest.raises(PolicyRpcError, match="must be an integer"):
        _receive_message(connection)


def test_policy_server_never_unlinks_an_existing_socket():
    path = _ExistingSocketPath()
    with pytest.raises(RuntimeError, match="refuse to unlink"):
        _prepare_socket_path(path)
    assert not path.unlinked
