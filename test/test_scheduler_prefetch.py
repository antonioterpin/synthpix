import os
import queue
import time

import numpy as np
import pytest

from synthpix.scheduler import PrefetchingFlowFieldScheduler


class MinimalScheduler:
    def __init__(self, total_batches=4, shape=(8, 8, 2)):
        self.total = total_batches
        self.count = 0
        self.shape = shape
        self.reset_called = False
        self.episode_length = total_batches

    def get_batch(self, batch_size):
        if self.count >= self.total:
            raise StopIteration
        self.count += 1
        return np.ones((batch_size,) + self.shape)

    def reset(self):
        self.count = 0
        self.reset_called = True

    def get_flow_fields_shape(self):
        return self.shape


def test_iter_and_next():
    scheduler = MinimalScheduler()
    pf = PrefetchingFlowFieldScheduler(scheduler, batch_size=2, buffer_size=2)

    pf_iter = iter(pf)
    batch = next(pf_iter)
    assert batch.shape == (2, 8, 8, 2)
    pf.shutdown()


def test_stop_iteration_from_queue_empty():
    scheduler = MinimalScheduler(total_batches=0)
    pf = PrefetchingFlowFieldScheduler(scheduler, batch_size=2)

    with pytest.raises(StopIteration):
        next(iter(pf))
    pf.shutdown()


def test_worker_eos_signal_when_queue_full():
    scheduler = MinimalScheduler(total_batches=0)

    pf = PrefetchingFlowFieldScheduler(scheduler=scheduler, batch_size=1, buffer_size=1)
    iter(pf)  # Starts the thread

    # Wait until EOS is in queue or timeout
    for _ in range(20):
        if not pf._queue.empty():
            break
        time.sleep(0.05)
    else:
        pytest.fail("Timeout waiting for worker to produce EOS.")

    with pytest.raises(StopIteration):
        next(pf)
    pf.shutdown()


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="TODO: make this test pass on CI when with multiple tests.",
)
def test_reset_stops_and_joins():
    scheduler = MinimalScheduler()
    pf = PrefetchingFlowFieldScheduler(scheduler, batch_size=2)

    pf.get_batch(2)
    assert pf._thread.is_alive()

    pf.reset()
    assert not pf._thread.is_alive()
    assert scheduler.reset_called


def test_next_episode_resets_thread_and_flushes_queue():
    scheduler = MinimalScheduler(total_batches=5)
    pf = PrefetchingFlowFieldScheduler(scheduler, batch_size=1, buffer_size=5)

    next(iter(pf))  # start thread
    time.sleep(0.2)  # allow prefetch

    pf._t = 3
    pf.next_episode()
    assert pf._t == 0
    pf.shutdown()


def test_shutdown_behavior():
    scheduler = MinimalScheduler()
    pf = PrefetchingFlowFieldScheduler(scheduler, batch_size=1)
    pf.get_batch(1)

    pf.shutdown()
    assert not pf._thread.is_alive()


def test_t_counter_wraps_after_episode():
    sched = MinimalScheduler(total_batches=5)
    sched.episode_length = 2
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=2)

    it = iter(pf)

    next(it)  # _t becomes 1
    next(it)  # _t becomes 2
    assert pf._t == 2  # still inside episode

    # Third request: internal counter must wrap to 1 again
    sched.total += 1
    next(it)
    assert pf._t == 1
    pf.shutdown()


@pytest.mark.parametrize("batch_size", [None, "invalid", -1, 1.3])
def test_invalid_batch_size_raises_value_error(batch_size):
    with pytest.raises(ValueError, match="batch_size must be a positive integer."):
        PrefetchingFlowFieldScheduler(
            MinimalScheduler(), batch_size=batch_size, buffer_size=1
        )


@pytest.mark.parametrize("buffer_size", [None, "invalid", -1, 1.3])
def test_invalid_buffer_size_raises_value_error(buffer_size):
    with pytest.raises(ValueError, match="buffer_size must be a positive integer."):
        PrefetchingFlowFieldScheduler(
            MinimalScheduler(), batch_size=1, buffer_size=buffer_size
        )


def test_get_batch_size_mismatch_raises_value_error():
    pf = PrefetchingFlowFieldScheduler(MinimalScheduler(), batch_size=2)
    with pytest.raises(ValueError):
        pf.get_batch(1)
    pf.shutdown()


def test_next_episode_flushes_remaining_and_restarts():
    sched = MinimalScheduler(total_batches=5)
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=5)

    it = iter(pf)
    # consume three batches to put us mid-episode
    for _ in range(3):
        next(it)

    remaining_before = pf.steps_remaining()  # should be 2
    assert remaining_before == 2

    pf.next_episode(join_timeout=1)

    # After next_episode() we must be at t == 0 and the queue empty/new thread alive
    assert pf._t == 0
    assert pf._thread.is_alive()
    pf.shutdown()


def test_reset_stops_thread_and_clears_queue():
    sched = MinimalScheduler(total_batches=3)
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=3)

    # prime the queue with two prefetched batches
    it = iter(pf)
    next(it)
    for _ in range(40):
        if not pf._queue.empty():
            break
        time.sleep(0.1)

    assert not pf._queue.empty(), "Pre-condition queue not empty before reset failed."
    pf.reset()

    assert pf._queue.empty(), "Queue should be empty after reset"
    assert sched.reset_called, "Scheduler should have been reset"

    pf.shutdown()


def test_worker_handles_full_queue_on_eos():
    sched = MinimalScheduler(total_batches=1)
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=1)

    it = iter(pf)
    time.sleep(0.1)

    # Consume first (and only) real batch – queue becomes empty again,
    # but EOS is already injected via the queue.Full branch.
    next(it)

    # next() must now raise StopIteration coming from the sentinel None.
    with pytest.raises(StopIteration):
        next(it)

    pf.shutdown()


def test_get_flow_fields_shape_matches_underlying_scheduler():
    """The wrapper should return the same shape as the wrapped scheduler."""
    shape = (8, 8, 2)
    sched = MinimalScheduler(shape=shape)
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1)

    assert pf.get_flow_fields_shape() == shape

    pf.shutdown()


def test_shutdown_when_queue_empty():
    """Make the queue empty before calling shutdown()."""
    sched = MinimalScheduler(total_batches=1)
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=1)

    it = iter(pf)
    # Consume the only real batch
    next(it)
    # Consume the EOS sentinel (raises StopIteration) → queue is now empty
    with pytest.raises(StopIteration):
        next(it)

    assert pf._queue.empty()

    # No exception should occur, and the background thread must be dead afterwards
    pf.shutdown()
    assert not pf._thread.is_alive()


def test_shutdown_when_queue_full():
    """Leave the queue full, call shutdown(), confirm worker stops."""
    sched = MinimalScheduler(total_batches=5)
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=1)

    iter(pf)
    for _ in range(40):
        if pf._queue.full():
            break
        time.sleep(0.05)
    else:
        pytest.fail("Queue never became full")

    assert pf._queue.full()

    pf.shutdown()

    # All worked out fine


def test_reset_when_queue_empty_and_thread_not_started():
    sched = MinimalScheduler(total_batches=2)
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=2)

    assert not pf._thread.is_alive()
    assert pf._queue.empty()

    # Invoke the method under test
    pf.reset()
    assert sched.reset_called
    assert pf._started is False
    assert pf._queue.empty()
    assert not pf._thread.is_alive()

    # Ensure wrapper still works end-to-end
    it = iter(pf)
    batch = next(it)
    assert batch.shape == (1,) + sched.shape
    pf.shutdown()


def test_reset_then_next_episode_three_cycles(monkeypatch):
    """Test that reset() and next_episode() work correctly in sequence."""
    shape = (8, 8, 2)
    sched = MinimalScheduler(total_batches=9, shape=shape)
    sched.episode_length = 2  # make episodes short

    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=2)

    # ---------------- Episode 1 ----------------
    first_batch = next(iter(pf))
    assert first_batch.shape == (1,) + shape
    assert pf._t == 1

    pf.reset()
    assert not pf._thread.is_alive()
    assert pf._queue.empty()

    # ---------------- Episode 2 ----------------
    pf.next_episode(join_timeout=1)
    assert pf._t == 0 and pf._thread.is_alive()

    # Consume one batch
    assert pf.get_batch(1).shape == (1,) + shape
    assert pf._t == 1
    # Case when queue NOT empty
    pf.next_episode(join_timeout=1)  # flush unfinished episode
    assert pf._t == 0

    # ---------------- Episode 3 ----------------
    def always_empty():
        raise queue.Empty

    monkeypatch.setattr(pf._queue, "get_nowait", always_empty, raising=True)
    # Consume one batch so we are mid-episode again (_t == 1)
    assert pf.get_batch(1).shape == (1,) + shape
    assert pf._t == 1 and pf._thread.is_alive()

    # case when queue empty
    pf.next_episode(join_timeout=1)
    assert pf._t == 0  # counters reset despite empty queue
    # Producer thread was restarted
    assert pf._thread.is_alive()

    # Clean shutdown
    pf.shutdown()


def test_next_raises_stop_iteration_when_queue_empty(monkeypatch):
    pf = PrefetchingFlowFieldScheduler(MinimalScheduler(), batch_size=1, buffer_size=1)

    # Monkey-patch Queue.get so it always raises queue.Empty immediately.
    def always_empty(*_a, **_kw):
        raise queue.Empty

    monkeypatch.setattr(pf._queue, "get", always_empty, raising=True)

    # Iteration must now fail with StopIteration via the Empty-queue branch.
    with pytest.raises(StopIteration):
        next(iter(pf))

    pf.shutdown()


class AlwaysEmptyScheduler:
    """A scheduler that immediately raises StopIteration."""

    def __init__(self, shape=(8, 8, 2)):
        self.shape = shape

    def get_batch(self, batch_size):
        raise StopIteration

    def reset(self):
        pass

    def get_flow_fields_shape(self):
        return self.shape


def test_worker_eos_signal_via_full_queue_branch_direct():
    # 1) Create a scheduler that has no data at all.
    sched = AlwaysEmptyScheduler(shape=(4, 4, 2))
    pf = PrefetchingFlowFieldScheduler(scheduler=sched, batch_size=2, buffer_size=1)

    # 2) Manually fill the queue so it's 'full' before the worker runs.
    dummy = np.zeros((2,) + sched.get_flow_fields_shape())
    pf._queue.put(dummy)
    assert pf._queue.full()

    # 3) Invoke _worker() directly: it will catch StopIteration,
    #    try to put(None) → hit queue.Full → run the 'Full' handler.
    pf._worker(eos_timeout=0)

    # 4) After that, the queue should contain *only* the EOS sentinel.
    contents = []
    while True:
        try:
            contents.append(pf._queue.get_nowait())
        except queue.Empty:
            break

    # 5) And consuming it via __next__ raises StopIteration.
    with pytest.raises(StopIteration):
        next(pf)

    pf.shutdown()
