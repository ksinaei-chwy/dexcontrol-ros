import threading
import time

import numpy as np

from dex_pico_teleop.output_worker import LatestFrameOutputWorker


def test_slow_output_replaces_waiting_frames_without_blocking_submitter():
    entered = threading.Event()
    release = threading.Event()
    published = []

    def slow_publish(frame):
        entered.set()
        release.wait(1.0)
        published.append(int(frame[0, 0, 0]))
        return True

    worker = LatestFrameOutputWorker(name="slow", publish=slow_publish)
    try:
        started = time.monotonic()
        worker.submit(np.full((1, 1, 3), 1, dtype=np.uint8), 1)
        assert entered.wait(0.2)
        worker.submit(np.full((1, 1, 3), 2, dtype=np.uint8), 2)
        worker.submit(np.full((1, 1, 3), 3, dtype=np.uint8), 3)
        assert time.monotonic() - started < 0.2
        release.set()
        deadline = time.monotonic() + 0.5
        while worker.stats().published_frames < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        stats = worker.stats()
        assert stats.enqueued_frames == 3
        assert stats.replaced_frames == 1
        assert published == [1, 3]
    finally:
        release.set()
        worker.shutdown()


def test_worker_records_transform_or_publish_failure():
    worker = LatestFrameOutputWorker(
        name="failure",
        publish=lambda frame: (_ for _ in ()).throw(RuntimeError("blocked")),
    )
    try:
        worker.submit(np.zeros((1, 1, 3), dtype=np.uint8), 1)
        deadline = time.monotonic() + 0.2
        while worker.stats().failures == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert worker.stats().last_error == "blocked"
    finally:
        worker.shutdown()
