import threading
import time

import numpy as np

from dex_vega_lerobot_inference.inference_worker import LatestObservationWorker
from dex_vega_lerobot_inference.observation_adapter import ObservationSnapshot


def _observation() -> ObservationSnapshot:
    return ObservationSnapshot(
        state=np.zeros(27, dtype=np.float32),
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        task="put the blue bird on the meeting desk",
        state_stamp_ns=1,
        camera_stamp_ns=1,
        receive_stamp_ns=1,
        created_stamp_ns=1,
        created_monotonic_ns=time.monotonic_ns(),
        state_age_seconds=0.0,
        camera_capture_age_seconds=0.0,
        camera_receive_age_seconds=0.0,
        synchronization_skew_seconds=0.0,
    )


class _BlockingRuntime:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.reset_count = 0

    def predict(self, observation):
        self.started.set()
        self.release.wait(timeout=2.0)
        return observation

    def reset(self):
        self.reset_count += 1


def test_reset_discards_inflight_result_and_resets_runtime():
    runtime = _BlockingRuntime()
    results = []
    errors = []
    worker = LatestObservationWorker(
        runtime,
        on_result=lambda observation, result: results.append(result),
        on_error=errors.append,
    )
    worker.submit(_observation())
    assert runtime.started.wait(timeout=1.0)
    worker.reset()
    runtime.release.set()
    deadline = time.monotonic() + 1.0
    while runtime.reset_count == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    stats = worker.stats()
    assert results == []
    assert errors == []
    assert runtime.reset_count == 1
    assert stats.discarded_after_reset == 1
    assert worker.close()


def test_pending_slot_replaces_older_observation():
    runtime = _BlockingRuntime()
    worker = LatestObservationWorker(runtime, on_result=lambda *_: None, on_error=lambda *_: None)
    worker.submit(_observation())
    assert runtime.started.wait(timeout=1.0)
    worker.submit(_observation())
    worker.submit(_observation())
    assert worker.stats().dropped_or_replaced == 1
    runtime.release.set()
    assert worker.close(timeout_seconds=1.0)
