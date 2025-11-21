import os
import queue
import time
import threading
import multiprocessing

import numpy as np
import pytest

from synthpix.scheduler import PrefetchingFlowFieldScheduler
from synthpix.scheduler.protocol import (
    EpisodeEnd,
    EpisodicSchedulerProtocol,
    SchedulerProtocol,
)
from synthpix.types import SchedulerData


class MinimalScheduler(SchedulerProtocol):
    def __init__(self, total_batches=4, shape=(8, 8, 2)):
        self.total = total_batches
        self.count = 0
        self.shape = shape
        self.reset_called = False

    def get_batch(self, batch_size):
        if self.count >= self.total:
            raise StopIteration
        self.count += 1
        return SchedulerData(flow_fields=np.ones((batch_size,) + self.shape))

    def reset(self):
        self.count = 0
        self.reset_called = True

    def get_flow_fields_shape(self):
        return self.shape

    @property
    def file_list(self) -> list[str]:
        """Returns the list of files used by the underlying scheduler.

        Returns:
            The list of files.
        """
        return ["dummy_file.npy"]

    @file_list.setter
    def file_list(self, new_file_list: list[str]) -> None:
        """Sets a new list of files for the underlying scheduler.

        Args:
            new_file_list: The new list of files to set.
        """
        pass


class MinimalEpisodic(EpisodicSchedulerProtocol):
    def __init__(self, total_batches=4, shape=(8, 8, 2)):
        self._episode_length = total_batches
        self.shape = shape
        self._t = 0

    def get_batch(self, batch_size):
        # never raise StopIteration at episode boundary
        if self._t >= self.episode_length:
            # if the prefetcher keeps calling this without next_episode,
            # you can either cycle or clamp; for tests, cycle is fine:
            self._t = 0
        self._t += 1
        return SchedulerData(flow_fields=np.ones((batch_size,) + self.shape))

    def next_episode(self):
        self._t = 0

    @property
    def episode_length(self) -> int:
        return self._episode_length

    def reset(self):
        self._t = 0

    def get_flow_fields_shape(self):
        return self.shape

    def steps_remaining(self) -> int:
        return self.episode_length - self._t

    @property
    def file_list(self) -> list[str]:
        return ["dummy_file.npy"]

    @file_list.setter
    def file_list(self, new_file_list: list[str]) -> None:
        pass


def test_single_producer_thread_across_episodes():

    shape = (8, 8, 2)
    sched = MinimalEpisodic(total_batches=50, shape=shape)
    sched._episode_length = 3  # short episodes to exercise next_episode() often

    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=4)

    pf.get_batch(1)  # start the thread
    time.sleep(0.1)  # let it prefetch a bit

    first_ident = pf._thread.ident
    assert pf._thread.is_alive() and first_ident is not None

    # Run several cycles: consume a bit, jump to next episode, repeat.
    for _ in range(5):
        # consume up to two steps in the episode
        for __ in range(2):
            try:
                pf.get_batch(1)
            except StopIteration:
                break

        pf.next_episode(join_timeout=0.2)

        # The same producer thread should still be alive (no restart)
        assert pf._thread.is_alive()
        assert pf._thread.ident == first_ident

        # Ensure only one worker is alive globally
        workers = [
            t for t in threading.enumerate() if t.is_alive() and "(_worker)" in t.name
        ]
        assert len(workers) == 1

    pf.shutdown()


def test_stop_iteration_from_queue_empty():
    scheduler = MinimalScheduler(total_batches=0)
    pf = PrefetchingFlowFieldScheduler(scheduler, batch_size=2)

    with pytest.raises(StopIteration):
        pf.get_batch(2)
    pf.shutdown()


def test_worker_eos_signal_when_queue_full():
    scheduler = MinimalScheduler(total_batches=1)

    pf = PrefetchingFlowFieldScheduler(scheduler=scheduler, batch_size=1, buffer_size=1)
    pf.get_batch(1)  # Starts the thread

    # Wait until EOS is in queue or timeout
    for _ in range(20):
        if not pf._queue.empty():
            break
        time.sleep(0.05)
    else:
        pytest.fail("Timeout waiting for worker to produce EOS.")

    with pytest.raises(StopIteration):
        pf.get_batch(1)
    pf.shutdown()


def test_reset_stops_and_joins():
    scheduler = MinimalScheduler(total_batches=5)
    pf = PrefetchingFlowFieldScheduler(scheduler, batch_size=1, buffer_size=2)

    pf.get_batch(1)
    assert pf._thread.is_alive()

    pf.reset()
    assert not pf._thread.is_alive()
    assert scheduler.reset_called


def test_next_episode_resets_thread_and_flushes_queue():
    scheduler = MinimalEpisodic(total_batches=5)
    pf = PrefetchingFlowFieldScheduler(scheduler, batch_size=1, buffer_size=5)

    pf.get_batch(1)  # start the thread
    time.sleep(0.2)  # allow prefetch

    pf._t = 3  # type: ignore[attr-defined]
    pf.next_episode()
    assert pf._t == 0  # type: ignore[attr-defined]
    pf.shutdown()


def test_shutdown_behavior():
    scheduler = MinimalScheduler(total_batches=5)
    pf = PrefetchingFlowFieldScheduler(scheduler, batch_size=1)
    pf.get_batch(1)

    pf.shutdown()
    assert not pf._thread.is_alive()


def test_t_counter_wraps_after_episode():
    sched = MinimalEpisodic(total_batches=2)
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=2)

    pf.get_batch(1)  # _t becomes 1
    assert pf._t == 1
    pf.get_batch(1)  # _t becomes 2
    assert pf._t == 2  # still inside episode

    with pytest.raises(EpisodeEnd):
        pf.get_batch(1)  # _t would wrap to 0 here
    pf.reset()
    assert pf._t == 0
    pf.get_batch(1)  # _t wraps to 0
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
    TOTAL_BATCHES = 20
    scheduler = MinimalEpisodic(total_batches=TOTAL_BATCHES)
    pf = PrefetchingFlowFieldScheduler(scheduler, batch_size=1, buffer_size=5)

    for _ in range(3):
        pf.get_batch(1)

    assert pf.steps_remaining() == TOTAL_BATCHES - 3

    if os.getenv("CI") == "true":
        # CI environments can be slow; give more time to flush
        pf.next_episode(join_timeout=10)
    else:
        pf.next_episode(join_timeout=1)

    # After next_episode() we must be at t == 0 and the queue empty/new thread alive
    assert pf._t == 0
    deadline = time.time() + 0.5
    while time.time() < deadline and not pf.is_running():
        time.sleep(0.01)
    assert pf.is_running()


def test_reset_stops_thread_and_clears_queue():
    sched = MinimalScheduler(total_batches=3)
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=3)

    # prime the queue with two prefetched batches
    _ = pf.get_batch(1)
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

    time.sleep(0.1)

    # Consume first (and only) real batch – queue becomes empty again,
    # but EOS is already injected via the queue.Full branch.
    pf.get_batch(1)

    with pytest.raises(StopIteration):
        pf.get_batch(1)

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

    # Consume the only real batch
    pf.get_batch(1)
    # Consume the EOS sentinel (raises StopIteration) → queue is now empty
    with pytest.raises(StopIteration):
        pf.get_batch(1)

    assert pf._queue.empty()

    # No exception should occur, and the background thread must be dead afterwards
    pf.shutdown()
    assert not pf._thread.is_alive()


def test_shutdown_when_queue_full():
    """Leave the queue full, call shutdown(), confirm worker stops."""
    sched = MinimalScheduler(total_batches=5)
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=1)

    pf.get_batch(1)  # start consuming to let producer fill the queue
    for _ in range(40):
        if pf._queue.full():
            break
        time.sleep(0.05)
    else:
        pytest.fail("Queue never became full")

    assert pf._queue.full()

    pf.shutdown()


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
    batch = pf.get_batch(1)
    assert batch.flow_fields.shape == (1,) + sched.shape
    pf.shutdown()


def test_reset_then_next_episode_three_cycles(monkeypatch):
    """Test that reset() and next_episode() work correctly in sequence."""
    shape = (8, 8, 2)
    sched = MinimalEpisodic(total_batches=10, shape=shape)
    sched._episode_length = 2  # make episodes short

    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=2)

    # ---------------- Episode 1 ----------------
    first_batch = pf.get_batch(1)
    assert first_batch.flow_fields.shape == (1,) + shape
    assert pf._t == 1  # type: ignore[attr-defined]

    pf.reset()
    assert not pf._thread.is_alive()
    assert pf._queue.empty()

    # ---------------- Episode 2 ----------------
    pf.next_episode(join_timeout=1)
    time.sleep(1)  # allow thread to start
    assert pf._t == 0 and pf._thread.is_alive()

    # Consume one batch
    assert pf.get_batch(1).flow_fields.shape == (1,) + shape
    assert pf._t == 1  # type: ignore[attr-defined]
    # Case when queue NOT empty
    pf.next_episode(join_timeout=1)  # flush unfinished episode
    assert pf._t == 0  # type: ignore[attr-defined]

    # ---------------- Episode 3 ----------------
    def always_empty():
        raise queue.Empty

    monkeypatch.setattr(pf._queue, "get_nowait", always_empty, raising=True)
    # Consume one batch so we are mid-episode again (_t == 1)
    assert pf.get_batch(1).flow_fields.shape == (1,) + shape
    assert pf._t == 1  # type: ignore[attr-defined]
    assert pf._thread.is_alive()

    # case when queue empty
    pf.next_episode(join_timeout=1)
    assert pf._t == 0  # type: ignore[attr-defined]
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
        pf.get_batch(1)

    pf.shutdown()


def test_next_episode_immediate_timeout_break():
    """If join_timeout is 0 the loop should hit the immediate timeout break."""
    sched = MinimalEpisodic(total_batches=10)
    sched._episode_length = 5
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=2)

    # Start the iterator so _started becomes True
    pf.get_batch(1)

    # Call next_episode with zero timeout to hit the `remaining_time <= 0` branch
    pf.next_episode(join_timeout=0)

    # After calling next_episode the internal counter must be reset
    assert pf._t == 0  # type: ignore[attr-defined]
    pf.shutdown()


def test_next_episode_handles_queue_empty(monkeypatch):
    """Simulate queue.get raising queue.Empty to exercise the except branch."""
    sched = MinimalEpisodic(total_batches=10)
    sched._episode_length = 5
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=2)

    # Ensure the wrapper is marked as started so the discard loop runs
    pf._started = True

    # Make get() raise queue.Empty to trigger the except/continue branch
    def always_empty(*_a, **_kw):
        raise queue.Empty

    monkeypatch.setattr(pf._queue, "get", always_empty, raising=True)

    # This should complete without raising and reset the internal counter
    pf.next_episode(join_timeout=0.05)
    assert pf._t == 0  # type: ignore[attr-defined]
    pf.shutdown()


def test_next_episode_breaks_on_eos_sentinel():
    """If an EOS (None) is found in the queue, the flush loop must break."""
    sched = MinimalEpisodic(total_batches=5)
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=2)

    # Mark started so the discard loop activates and push an EOS sentinel
    pf._started = True
    pf._queue.put(None)

    # This should notice the EOS and break out, resetting _t
    pf.next_episode(join_timeout=1)
    assert pf._t == 0  # type: ignore[attr-defined]
    pf.shutdown()


class AlwaysEmptyScheduler(SchedulerProtocol):
    """A scheduler that immediately raises StopIteration."""

    def __init__(self, shape=(8, 8, 2)):
        self.shape = shape

    def get_batch(self, batch_size):
        raise StopIteration

    def reset(self):
        pass

    def get_flow_fields_shape(self):
        return self.shape

    @property
    def file_list(self) -> list[str]:
        return []

    @file_list.setter
    def file_list(self, new_file_list: list[str]) -> None:
        pass

    @classmethod
    def from_config(cls, scheduler, config) -> "AlwaysEmptyScheduler":
        return cls()


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
        pf.get_batch(2)

    pf.shutdown()


def test_get_batch_starts_worker_thread():
    sched = MinimalScheduler(total_batches=100)
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=2)

    assert not pf.is_running()
    pf.get_batch(1)
    time.sleep(0.001)  # allow thread to start
    assert pf.is_running()

    pf.shutdown()


def test_multiple_get_batch_in_parallel_threads():
    sched = MinimalScheduler(total_batches=100)
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=10)

    results = []
    exceptions = []

    def worker():
        try:
            batch = pf.get_batch(1)
            results.append(batch)
        except Exception as e:
            exceptions.append(e)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 5
    assert len(exceptions) == 0

    pf.shutdown()


def worker(queue, total_batches, batch_size, buffer_size):
    try:
        # Create scheduler *inside* the subprocess
        sched = MinimalScheduler(total_batches=total_batches)
        pf = PrefetchingFlowFieldScheduler(
            sched, batch_size=batch_size, buffer_size=buffer_size
        )

        batch = pf.get_batch(1)
        pf.shutdown()
        queue.put(("ok", batch))

    except Exception as e:
        queue.put(("err", repr(e)))


def test_multiple_get_batch_in_parallel_processes():
    total_batches = 100
    batch_size = 1
    buffer_size = 10

    result_queue = multiprocessing.Queue()

    processes = [
        multiprocessing.Process(
            target=worker, args=(result_queue, total_batches, batch_size, buffer_size)
        )
        for _ in range(5)
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    oks = 0
    errs = []

    while not result_queue.empty():
        tag, payload = result_queue.get()
        if tag == "ok":
            oks += 1
        else:
            errs.append(payload)

    assert oks == 5, f"Expected 5 successes, got {oks}"
    assert len(errs) == 0, f"Errors: {errs}"


class SlowMinimalScheduler(MinimalScheduler):
    """Scheduler that is too slow for the prefetcher timeout on first batch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._first_call = True

    def get_batch(self, batch_size: int):
        # First call: sleep longer than the *configured* startup_timeout in the test.
        if self._first_call:
            self._first_call = False
            time.sleep(0.2)
        return super().get_batch(batch_size)


def test_prefetcher_raises_stopiteration_when_producer_is_too_slow():
    sched = SlowMinimalScheduler(total_batches=10)
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=4)

    # Make startup_timeout smaller than the 0.2s sleep to force a timeout
    pf.startup_timeout = 0.05
    pf.steady_state_timeout = 0.05

    # Now the first get_batch *must* hit queue.Empty → StopIteration
    with pytest.raises(StopIteration):
        pf.get_batch(batch_size=1)

    pf.shutdown()


def test_startup_and_steady_timeouts_are_used(monkeypatch):
    """First batch should use startup_timeout, subsequent batches steady_state_timeout."""
    sched = MinimalScheduler(total_batches=10)
    pf = PrefetchingFlowFieldScheduler(
        sched,
        batch_size=1,
        buffer_size=2,
        startup_timeout=42.0,
        steady_state_timeout=7.0,
    )

    recorded_timeouts: list[float | None] = []

    # We don't want to block on the real queue, just capture the timeout arg.
    def fake_get(*_args, **kwargs):
        recorded_timeouts.append(kwargs.get("timeout"))
        # Return a dummy batch to keep the rest of the pipeline happy.
        return SchedulerData(flow_fields=np.ones((1,) + sched.shape))

    monkeypatch.setattr(pf._queue, "get", fake_get, raising=True)

    # First get_batch -> must use startup_timeout
    pf.get_batch(1)
    # Second get_batch -> must use steady_state_timeout
    pf.get_batch(1)

    assert recorded_timeouts[0] == pf.startup_timeout
    assert recorded_timeouts[1] == pf.steady_state_timeout

    pf.shutdown()


def test_startup_flag_clears_after_first_batch():
    sched = MinimalScheduler(total_batches=3)
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=2)

    # By construction, we expect startup mode at initialization.
    assert getattr(pf, "startup", None) is True

    batch = pf.get_batch(1)
    assert batch.flow_fields.shape == (1,) + sched.shape

    # After the first successful get_batch, startup must be False.
    assert pf.startup is False

    pf.shutdown()


def test_reset_restores_startup_flag_and_behavior(monkeypatch):
    sched = MinimalScheduler(total_batches=5)
    pf = PrefetchingFlowFieldScheduler(sched, batch_size=1, buffer_size=2)

    # 1) Consume first batch -> startup should become False.
    _ = pf.get_batch(1)
    assert pf.startup is False

    # 2) Reset → startup should be restored to True.
    pf.reset()
    assert pf.startup is True

    # To avoid depending on actual thread timing, patch queue.get again to
    # verify that after reset the next get_batch uses startup_timeout.
    pf.startup_timeout = 11.0
    pf.steady_state_timeout = 3.0

    timeouts: list[float | None] = []

    def fake_get(*_args, **kwargs):
        timeouts.append(kwargs.get("timeout"))
        return SchedulerData(flow_fields=np.ones((1,) + sched.shape))

    monkeypatch.setattr(pf._queue, "get", fake_get, raising=True)

    _ = pf.get_batch(1)
    assert timeouts[0] == pf.startup_timeout
    assert pf.startup is False  # and it flips again after that first batch

    pf.shutdown()
