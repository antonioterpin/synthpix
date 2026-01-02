"""Adapters to make Grain DataLoaders compatible with Legacy Schedulers."""

import logging
from typing import cast

import grain.python as grain
import numpy as np

from synthpix.data_sources.base import FileDataSource
from synthpix.data_sources.episodic import EpisodicDataSource
from synthpix.scheduler.protocol import (
    EpisodeEndError,
    EpisodicSchedulerProtocol,
    SchedulerProtocol,
)
from synthpix.types import SchedulerData

logger = logging.getLogger(__name__)


class GrainSchedulerAdapter(SchedulerProtocol):
    """Adapts a Grain DataLoader to the SchedulerProtocol interface.

    This allows legacy Samplers to consume data from a Grain pipeline
    without modification.
    """

    def __init__(self, loader: grain.DataLoader):
        """Initialize the adapter.

        Args:
            loader: A grain.DataLoader instance.

        Raises:
            ValueError: If loader is not a grain.DataLoader or not iterable.
        """
        if not isinstance(loader, grain.DataLoader):
            raise ValueError("loader must be a grain.DataLoader instance")
        self.loader = loader
        if not hasattr(self.loader, "__iter__"):
            raise ValueError("loader must be iterable")
        self._iterator = iter(self.loader)

        # Shape of the flow fields (H, W, 2)
        self._cached_shape: tuple[int, int, int] | None = None

    def shutdown(self) -> None:
        """Compatibility layer for shutdown."""
        # Grain loaders clean up on GC or exit.
        pass

    def get_flow_fields_shape(self) -> tuple[int, int, int]:
        """Returns the shape of the flow field.

        Returns:
            The shape of the flow field (H, W, 2).

        Raises:
            StopIteration: If loader is exhausted before shape is determined.
        """
        if self._cached_shape is not None:
            return self._cached_shape

        # We peek by taking one item and recreating the iterator.
        # This is potentially expensive but safe.
        logger.debug("Peeking at first batch to determine shape...")
        temp_iter = iter(self.loader)
        try:
            batch = next(temp_iter)
        except StopIteration:
            # If the loader is empty, we propagate StopIteration so the caller
            # knows it's empty.
            raise
        # batch is a dict, likely with "flow_fields" key
        # shape: (B, H, W, 2)
        flow = batch["flow_fields"]
        # Return (H, W, 2)
        res_shape = cast(tuple[int, int, int], tuple(flow.shape[1:]))
        self._cached_shape = res_shape

        return res_shape

    @property
    def include_images(self) -> bool:
        """Returns True if the underlying data source provides images.

        Returns:
            True if the underlying data source provides images.
        """
        if hasattr(self.loader, "_data_source"):
            ds = self.loader._data_source
            return getattr(ds, "include_images", False)
        return False

    def _to_scheduler_data(
        self, batch: dict, target_batch_size: int
    ) -> SchedulerData:
        """Converts Grain batch dict to SchedulerData with padding and masking.

        Args:
            batch: A dict containing the batch data.
            target_batch_size: The target batch size.

        Returns:
            A batch of flow fields.

        Raises:
            KeyError: If images are expected but not found in batch.
            ValueError: If batch data has invalid types or shapes.
        """
        flow = batch["flow_fields"]
        images1 = batch.get("images1")
        images2 = batch.get("images2")
        files = tuple(batch.get("file", ()))

        if self.include_images and (images1 is None or images2 is None):
            raise KeyError("Images expected but not found in batch.")

        # Validation
        if not isinstance(flow, np.ndarray):
            raise ValueError(
                f"Flow fields must be a np.ndarray, got {type(flow)}"
            )
        if images1 is not None and not isinstance(images1, np.ndarray):
            raise ValueError(
                f"Images1 must be a np.ndarray, got {type(images1)}"
            )
        if images2 is not None and not isinstance(images2, np.ndarray):
            raise ValueError(
                f"Images2 must be a np.ndarray, got {type(images2)}"
            )

        current_batch_size = flow.shape[0]

        # Calculate padding needed
        pad_size = target_batch_size - current_batch_size

        valid_mask = np.ones((target_batch_size,), dtype=bool)
        if pad_size > 0:
            valid_mask[current_batch_size:] = False

            # Pad flow
            padding = [(0, pad_size)] + [(0, 0)] * (flow.ndim - 1)
            flow = np.pad(flow, padding, mode="constant")

            # Pad images if present
            if images1 is not None:
                img_padding = [(0, pad_size)] + [(0, 0)] * (images1.ndim - 1)
                images1 = np.pad(images1, img_padding, mode="constant")
            if images2 is not None:
                img_padding = [(0, pad_size)] + [(0, 0)] * (images2.ndim - 1)
                images2 = np.pad(images2, img_padding, mode="constant")

            # Pad files
            files = files + ("",) * pad_size

        elif target_batch_size < current_batch_size:
            # Truncate
            flow = flow[:target_batch_size]
            if images1 is not None:
                images1 = images1[:target_batch_size]
            if images2 is not None:
                images2 = images2[:target_batch_size]
            files = files[:target_batch_size]

        return SchedulerData(
            flow_fields=flow,
            images1=images1,
            images2=images2,
            files=files,
            mask=valid_mask,
        )

    def get_batch(self, batch_size: int) -> SchedulerData:
        """Retrieves a batch of flow fields.

        Args:
            batch_size: The batch size to retrieve.

        Returns:
            SchedulerData: A batch of flow fields.

        Raises:
            StopIteration: If the Grain DataLoader is exhausted.
        """
        try:
            batch = next(self._iterator)
        except StopIteration:
            # Grain iterator finished
            raise StopIteration("Grain DataLoader exhausted.") from None

        return self._to_scheduler_data(batch, batch_size)

    def reset(self) -> None:
        """Resets the state (re-creates iterator)."""
        self._iterator = iter(self.loader)
        logger.debug("GrainSchedulerAdapter reset.")

    @property
    def file_list(self) -> list[str]:
        """Returns the list of files (if accessible).

        Returns:
            The list of files.
        """
        # Best effort: try to get from source
        if hasattr(self.loader, "_data_source") and isinstance(
            self.loader._data_source, FileDataSource
        ):
            return list(self.loader._data_source.file_list)
        return []

    @file_list.setter
    def file_list(self, value: list[str]) -> None:
        """Sets the file list (not supported on compiled loaders)."""
        # Grain loaders are usually immutable after creation regarding file
        # list.
        # Supporting this would require rebuilding the loader.
        # We will remove this from the protocol once we transition completely
        # to Grain.
        raise NotImplementedError(
            "Setting file_list on GrainSchedulerAdapter is not supported."
        )


class GrainEpisodicAdapter(GrainSchedulerAdapter, EpisodicSchedulerProtocol):
    """Adapts a Grain DataLoader to the EpisodicSchedulerProtocol interface.

    Requires data to contain episodic metadata ('_timestep', '_is_last_step')
    produced by EpisodicDataSource.
    """

    def __init__(
        self,
        loader: grain.DataLoader,
    ) -> None:
        """Initialize the adapter.

        Args:
            loader: A grain.DataLoader instance.

        Raises:
            ValueError: If data source is not an EpisodicDataSource.
        """
        super().__init__(loader)
        self._current_timestep: int = 0
        self._episode_length: int

        src = getattr(self.loader, "_data_source", None)
        if isinstance(src, EpisodicDataSource):
            self._episode_length = src.episode_length
        else:
            raise ValueError("Data source is not an EpisodicDataSource.")

    def get_batch(self, batch_size: int) -> SchedulerData:
        """Retrieves a batch of flow fields.

        Args:
            batch_size: The batch size to retrieve.

        Returns:
            SchedulerData: A batch of flow fields.

        Raises:
            EpisodeEndError: If the episode sequence has finished.
        """
        try:
            batch = next(self._iterator)
        except StopIteration:
            raise EpisodeEndError("Episode Sequence Finished.") from None

        # Update tracking
        if "_timestep" in batch:
            t = batch["_timestep"][0]
            self._current_timestep = int(t)

        return self._to_scheduler_data(batch, batch_size)

    def steps_remaining(self) -> int:
        """Returns steps remaining in current episode.

        Returns:
            Number of steps remaining in current episode.
        """
        return int(self._episode_length - (self._current_timestep + 1))

    def next_episode(self) -> None:
        """Fast-forwards to the start of the next episode.

        Raises:
            KeyError: If the batch is missing the required '_timestep' metadata.
        """
        # In Grain, "next episode" just means "keep reading until timestep goes
        # back to 0".

        # If we are NOT at end, we must skip.
        while self.steps_remaining() > 0:
            try:
                batch = next(self._iterator)
                # Update state so loop terminates
                if "_timestep" in batch:
                    t = batch["_timestep"][0]
                    self._current_timestep = t
                else:
                    raise KeyError(
                        "Batch missing required '_timestep' metadata"
                    )
            except StopIteration:
                # End of data
                break

        # Acts as "before 0"
        self._current_timestep = -1

    @property
    def episode_length(self) -> int:
        """Returns the length of the current episode.

        Returns:
            int: The length of the current episode.

        Raises:
            ValueError: If the episode length is unknown.
        """
        if self._episode_length is None:
            raise ValueError("Episode length unknown.")
        return int(self._episode_length)
