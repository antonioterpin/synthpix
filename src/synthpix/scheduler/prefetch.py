"""PrefetchingFlowFieldScheduler to asynchronously prefetch flow fields."""
import queue
import threading

from synthpix.utils import logger


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
        """Returns the iterator instance itself and starts the background thread."""
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
            batch = self._queue.get(timeout=0.1)
            if hasattr(self.scheduler, "episode_length"):
                if self._t >= self.episode_length:
                    self._t = 0
                self._t += 1

            if batch is None:
                logger.info("No more data available. " "Stopping.")
                raise StopIteration
            if hasattr(self.scheduler, "episode_length"):
                logger.debug(f"t = {self._t}")
            return batch
        except queue.Empty:
            logger.info("Prefetch queue is empty. " "Waiting for data.")
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

    def _worker(self):
        """Background thread that continuously fetches batches from the scheduler."""
        # This will run until the stop event is set:
        while not self._stop_event.is_set():
            try:
                batch = self.scheduler.get_batch(self.batch_size)
            except StopIteration:
                # Intended behavior here: I called get_batch() and ran into a
                # StopIteration, it means there is no more data left. The underlying
                # scheduler can be implemented in a way that it raises
                # StopIteration when it has no more data to provide or when it has
                # produced an incomplete batch. In the latter case, the behavior is so
                # that the prefetching scheduler will ignore the incomplete batch
                # and signal end‑of‑stream to consumer
                try:
                    self._queue.put(None, block=False)
                except queue.Full:
                    logger.debug("Queue full when signalling EOS; waiting for space…")
                    # block until it fits
                    self._queue.put(None, block=True)
                return

            # This will block until there is free space in the queue:
            # no busy‑waiting needed.
            try:
                self._queue.put(batch, block=True)
            except Exception as e:
                logger.warning(f"Failed to put batch: {e}")
                return

    def reset(self):
        """Resets the prefetching scheduler and underlying scheduler."""
        # Stop the background thread and clear the queue
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
        self._queue.queue.clear()

        logger.debug("Prefetching thread stopped, queue cleared.")

        # Reinitialize the scheduler and start the thread
        self._stop_event.clear()
        self._queue = queue.Queue(maxsize=self.buffer_size)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._started = False
        self.scheduler.reset()

        logger.debug("Prefetching thread reinitialized, scheduler reset.")

    def next_episode(self, join_timeout=0.1):
        """Flush the remaining items of the current episode and restart.

        This method removes the remaining items from the current episode from
        the queue but does not inform the underlying scheduler.
        This implementation works as far as the buffer size is larger than
        the episode length.

        Args:
            join_timeout: float
                Timeout for the thread to join.
        """
        if self._started:
            # halt producer thread
            self._stop_event.set()
            if self._thread.is_alive():
                self._thread.join(timeout=join_timeout)

            # drain the part of the queue that belongs to the unfinished episode
            # discard up to `n_left` batches (may be fewer if queue < n_left)
            for _ in range(self.steps_remaining()):
                logger.debug(
                    "Moving to next episode, "
                    f"of length {self.episode_length}, "
                    f"while {self._t} steps have been taken so far"
                )
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

            logger.debug(f"Flushed {self.steps_remaining()} batches from queue.")

        self._t = 0
        # restart producer thread
        if self._started:
            self._stop_event.clear()
        else:
            self._started = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        logger.debug("Next episode started, " "remaining items flushed from the queue.")

    def steps_remaining(self) -> int:
        """Return the number of steps remaining in the current episode.

        Returns:
            int: Number of steps remaining in the current episode.
        """
        return self.episode_length - self._t

    def shutdown(self, join_timeout=0.1):
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
