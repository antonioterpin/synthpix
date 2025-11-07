"""PrefetchingFlowFieldScheduler to asynchronously prefetch flow fields."""

import queue
import threading
import time
from typing_extensions import Self
import numpy as np

from goggles import get_logger

from synthpix.utils import SYNTHPIX_SCOPE
from synthpix.scheduler.protocol import (
    EpisodicSchedulerProtocol,
    SchedulerProtocol,
    PrefetchedSchedulerProtocol
)

logger = get_logger(__name__, scope=SYNTHPIX_SCOPE)


class PrefetchingFlowFieldScheduler(PrefetchedSchedulerProtocol):
    """Prefetching Wrapper around a FlowFieldScheduler or an EpisodicScheduler.

    It asynchronously prefetches batches of flow fields using a
    background thread to keep the GPU fed.
    """

    def __init__(
        self,
        scheduler: SchedulerProtocol,
        batch_size: int,
        buffer_size: int = 8,
    ):
        """Initializes the prefetching scheduler.

        If the underlying scheduler is episodic, it will recognize it and handle
        moving to the next episode seamlessly.

        Args:
            scheduler: The underlying flow field scheduler.
            batch_size: Flow field slices per batch.
                Must match the underlying scheduler.
            buffer_size: Number of batches to prefetch.
        """
        self.scheduler = scheduler

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(buffer_size, int) or buffer_size <= 0:
            raise ValueError("buffer_size must be a positive integer.")
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self._queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)

        self._started = False
        self._t = 0

    def get_batch(self, batch_size: int) -> np.ndarray:
        """Return the next batch from the prefetch queue.

        The batch matches the underlying scheduler's interface.
        
        Args:
            batch_size: Number of flow field slices to retrieve.

        Returns: A preloaded batch of flow fields.
        
        Raises:
            ValueError: If the requested batch_size does not match 
                the prefetching batch size.
        """
        if batch_size != self.batch_size:
            raise ValueError(
                f"Batch size {batch_size} does not match the "
                f"prefetching batch size {self.batch_size}."
            )
        self._start_worker()

        if isinstance(self.scheduler, EpisodicSchedulerProtocol):
            if self._t >= self.scheduler.episode_length:
                raise StopIteration(
                    "Episode ended. No more flow fields available. "
                    "Use next_episode() to continue."
                )
        try:
            batch = self._queue.get(block=True, timeout=2)
            self._t += 1
            if batch is None:
                logger.info("End of stream reached, stopping iteration.")
                raise StopIteration
            return batch
        except queue.Empty:
            logger.info("Unable to get data.")
            raise StopIteration

    def get_flow_fields_shape(self) -> tuple[int, ...]:
        """Return the shape of the flow fields from the underlying scheduler.

        Returns:
            Shape of the flow fields as returned by the underlying scheduler.
        """
        return self.scheduler.get_flow_fields_shape()
    
    def _start_worker(self) -> None:
        """Starts the background prefetching thread if not already started."""
        if not self._started:
            self._started = True
            self._thread.start()
            logger.debug("Background thread started.")

    def _worker(self, eos_timeout: float = 2.0) -> None:
        """Background thread that fetches batches from the scheduler.

        Args:
            eos_timeout: Timeout in seconds for putting the end-of-stream
                signal in the queue.
        """
        while not self._stop_event.is_set():
            try:
                batch = self.scheduler.get_batch(self.batch_size)
            except EpisodeEnd as e:
                assert isinstance(self.scheduler, EpisodicSchedulerProtocol)
                self.scheduler.next_episode()
                continue
            except StopIteration:
                # Intended behavior here:
                # I called get_batch() and ran into a StopIteration,
                # it means there is no more data left.
                # The underlying scheduler can be implemented in a way that it raises
                # StopIteration when it has no more data to provide or when it has
                # produced an incomplete batch. In the latter case, the behavior is so
                # that the prefetching scheduler will ignore the incomplete batch
                # and signal end‑of‑stream to consumer
                try:
                    self._queue.put(None, block=True, timeout=eos_timeout)
                except queue.Full:
                    # If the queue is full for <eos_timeout>, I remove one item
                    # before I can put the end‑of‑stream signal.

                    # Acquire the mutex to ensure atomicity
                    with self._queue.mutex:
                        if self._queue.queue:
                            # Remove one item from the queue to free up a slot
                            self._queue.queue.popleft()

                        # Write the EOS sentinel atomically
                        self._queue.queue.append(None)

                        # Notify the consumer that the end-of-stream signal is available
                        self._queue.not_empty.notify_all()

                logger.info("No more data to fetch, stopping prefetching thread.")
                self._stop_event.set()
                return

            # This will block until there is free space in the queue:
            # no busy‑waiting needed.
            # Exception cannot be raised here, would be dead code.
            self._queue.put(batch, block=True)

    def reset(self) -> None:
        """Resets the prefetching scheduler and underlying scheduler."""
        # Set the stop event to stop the current thread
        self._stop_event.set()

        # If the thread is stuck on put(), free up one slot in the queue
        # so the thread can check the stop event.
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

        # Wait for the thread to finish
        if self._thread.is_alive():
            self._thread.join()

        # Clear the queue to remove any remaining items
        with self._queue.mutex:
            self._queue.queue.clear()

        logger.debug("Prefetching thread stopped, queue cleared.")

        # Reinitialize the scheduler and start the thread
        self.scheduler.reset()
        self._t = 0
        self._started = False
        self._stop_event.clear()
        self._queue = queue.Queue(maxsize=self.buffer_size)
        self._thread = threading.Thread(target=self._worker, daemon=True)

        logger.debug("Prefetching thread reinitialized, scheduler reset.")

    def shutdown(self, join_timeout: float = 2.0) -> None:
        """Gracefully shuts down the background prefetching thread.

        Args:
            join_timeout: Timeout in seconds for joining the thread.
        """
        try:
            self._stop_event.set()
        except AttributeError:
            # stop event not initialized, nothing to do
            return

        # If producer is stuck on put(), free up one slot
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

        # If consumer is stuck on get(), inject the end-of-stream signal
        try:
            self._queue.put(None, block=False)
        except queue.Full:
            pass

        # Wait for the thread to finish
        if self._thread.is_alive():
            self._thread.join(timeout=join_timeout)

    def __del__(self) -> None:
        """Gracefully shuts down the scheduler upon deletion."""
        self.shutdown()

    def is_running(self) -> bool:
        """Check if the prefetching thread is currently running.

        Returns: True if the prefetching thread is alive, False otherwise.
        """
        t = self._thread
        return t is not None and t.is_alive()

    def steps_remaining(self) -> int:
        """Returns the number of steps remaining in the current episode.

        Returns: Number of steps remaining.
        """
        if not isinstance(self.scheduler, EpisodicSchedulerProtocol):
            # return 1 if not episodic... never ending ;)
            return 1
        return self.scheduler.episode_length - self._t

    def next_episode(self, join_timeout: float = 2.0) -> None:
        """Flush the current episode and prepare for the next one.

        The scheduler should reset any internal state necessary for
        starting a new episode.

        Also it starts the worker if not already started.

        Args:
            join_timeout: Timeout in seconds for joining the thread.
        """
        if not isinstance(self.scheduler, EpisodicSchedulerProtocol):
            # do nothing if not episodic
            return
        
        print("Called next_episode() in prefetching scheduler")
        print(f"steps remaining before next_episode(): {self.steps_remaining()}")
        if self._started and self.steps_remaining() > 0:
            to_discard = self.steps_remaining()
            discarded = 0
            deadline = time.time() + join_timeout
            while discarded < to_discard:
                remaining_time = deadline - time.time()
                if remaining_time <= 0:
                    break
                try:
                    item = self._queue.get(block=True, timeout=remaining_time)
                except queue.Empty:
                    continue
                if item is None:  # End-of-stream signal
                    break
                discarded += 1
        
        self._t = 0

        self._start_worker()

    @property
    def episode_length(self) -> int:
        """Returns the length of the episode.

        Returns: The length of the episode.
        """
        if not isinstance(self.scheduler, EpisodicSchedulerProtocol):
            raise AttributeError(
                "Underlying scheduler lacks episode_length property."
            )
        return self.scheduler.episode_length
    
    @property
    def file_list(self) -> list[str]:
        """Returns the list of files used by the underlying scheduler.

        Returns: The list of files.
        """
        return self.scheduler.file_list
    
    @file_list.setter
    def file_list(self, new_file_list: list[str]) -> None:
        """Sets a new list of files for the underlying scheduler.

        Args:
            new_file_list: The new list of files to set.
        """
        self.scheduler.file_list = new_file_list
