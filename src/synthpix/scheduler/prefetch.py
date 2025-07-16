"""PrefetchingFlowFieldScheduler to asynchronously prefetch flow fields."""
import queue
import threading

from ..utils import logger


class PrefetchingFlowFieldScheduler:
    """Prefetching Wrapper around a FlowFieldScheduler.

    It asynchronously prefetches batches of flow fields using a
    background thread to keep the GPU fed.
    """

    def __init__(self, scheduler, batch_size: int, buffer_size: int = 8):
        """Initializes the prefetching scheduler.

        If the underlying scheduler is episodic, it will recognize it and handle
        moving to the next episode seamlessly.

        Args:
            scheduler:
                The underlying flow field scheduler.
            batch_size: int
                Flow field slices per batch, must match the underlying scheduler.
            buffer_size: int
                Number of batches to prefetch.
        """
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Episodic behavior
        if hasattr(self.scheduler, "episode_length"):
            self.episode_length = getattr(self.scheduler, "episode_length")
            self._t = 0

        self._queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)

        self._started = False

    def __iter__(self):
        """Returns the iterator instance itself and starts the background thread.

        If the background thread is not started yet, it will be started.
        This behavior also takes care of the case where there has been an Exception
        in the previous run, and the thread needs to be restarted.
        """
        if not self._started:
            self._started = True
            self._thread.start()
            logger.debug("Background thread started.")
        return self

    def __next__(self):
        """Returns the next batch of flow fields from the prefetch queue.

        Raises:
            StopIteration: If the queue is empty or no more data is available.
        """
        try:
            batch = self._queue.get(block=True, timeout=2)
            if hasattr(self.scheduler, "episode_length"):
                if self._t >= self.episode_length:
                    self._t = 0
                self._t += 1
            if hasattr(self.scheduler, "episode_length"):
                logger.debug(f"t = {self._t}")
            if batch is None:
                logger.info("End of stream reached, stopping iteration.")
                raise StopIteration
            return batch
        except queue.Empty:
            logger.info("Unable to get data.")
            raise StopIteration

    def get_batch(self, batch_size):
        """Return the next batch from the prefetch queue, matching scheduler interface.

        Returns:
            np.ndarray: A preloaded batch of flow fields.
        """
        if batch_size != self.batch_size:
            raise ValueError(
                f"Batch size {batch_size} does not match the "
                f"prefetching batch size {self.batch_size}."
            )
        if not self._started:
            self.__iter__()
        return next(self)

    def get_flow_fields_shape(self):
        """Return the shape of the flow fields from the underlying scheduler.

        Returns:
            tuple: Shape of the flow fields as returned by the underlying scheduler.
        """
        return self.scheduler.get_flow_fields_shape()

    def _worker(self, eos_timeout=2):
        """Background thread that continuously fetches batches from the scheduler."""
        while not self._stop_event.is_set():
            try:
                batch = self.scheduler.get_batch(self.batch_size)
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
                    # If the queue is full for <timeout>, I remove one item
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

    def reset(self):
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
        self._started = False
        self._stop_event.clear()
        self._queue = queue.Queue(maxsize=self.buffer_size)
        self._thread = threading.Thread(target=self._worker, daemon=True)

        logger.debug("Prefetching thread reinitialized, scheduler reset.")

    def next_episode(self, join_timeout=2):
        """Flush the remaining items of the current episode and restart.

        This method removes the remaining items from the current episode from
        the queue but does not inform the underlying scheduler.
        This implementation works as far as the buffer size is larger than
        the episode length.

        Args:
            join_timeout: float
                Timeout for the thread to join.
        """
        if self._started and self.steps_remaining() > 0:
            # if this is a premature halt, I need to eliminate the first
            # steps_remaining batches from the queue.
            # Two cases:
            # 1. If the prefetching thread is already at the next episode,
            # I can just flush the queue.
            # 2. If the prefetching thread is still in the current episode,
            # I need to stop it and then flush the queue.
            # Both cases can be handled by stopping the thread and flushing
            # steps_remaining items from the queue, or until the queue is empty.
            # If the thread is stuck on put(),
            # I need to free up one slot in the queue, which is safe since
            # I am going to discard the items anyway.
            self._stop_event.set()
            if self._thread.is_alive():
                try:
                    self._queue.get_nowait()  # Free up one slot in the queue
                    self._t += 1  # Increment t to account for the discarded batch
                except queue.Empty:
                    logger.debug("Queue is empty, no need to free up a slot.")
                    pass
                self._thread.join(timeout=join_timeout)

            # Drain the part of the queue that belongs to the unfinished episode
            # discard up to steps_remaing batches (may be fewer if prefetching is
            # slower than consuming)
            for i in range(self.steps_remaining()):
                logger.debug(
                    "Moving to next episode, "
                    f"of length {self.episode_length}, "
                    f"while {self._t} steps have been taken so far"
                )
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

        self._t = 0
        # restart producer thread
        if self._started:
            self._stop_event.clear()
        else:
            self._started = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        logger.debug("Next episode started.")

    def steps_remaining(self) -> int:
        """Return the number of steps remaining in the current episode.

        Returns:
            int: Number of steps remaining in the current episode.
        """
        return self.episode_length - self._t

    def shutdown(self, join_timeout=2):
        """Gracefully shuts down the background prefetching thread."""
        self._stop_event.set()

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

    def __del__(self):
        """Gracefully shuts down the scheduler upon deletion."""
        self.shutdown()
