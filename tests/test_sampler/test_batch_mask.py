"""Regression tests for boolean, non-optional sampler output masks."""

from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from synthpix.sampler import RealImageSampler, SyntheticImageSampler
from synthpix.scheduler.protocol import SchedulerProtocol
from synthpix.types import ImageGenerationSpecification, SchedulerData


@pytest.mark.parametrize("sampler_kind", ["real", "synthetic"])
@pytest.mark.parametrize("mask_kind", ["missing", "boolean", "integer"])
def test_sampler_masks_support_boolean_indexing(
    sampler_kind: str, mask_kind: str
) -> None:
    """Default missing masks to True and preserve supplied validity flags.

    Args:
        sampler_kind: Real-image or synthetic sampler implementation.
        mask_kind: Missing metadata, boolean flags, or integer flags.
    """
    count = 2
    mask: NDArray[np.bool_] | NDArray[np.int32] | None
    if mask_kind == "missing":
        mask = None
    elif mask_kind == "boolean":
        mask = np.array([True, False])
    else:
        mask = np.array([1, 0], dtype=np.int32)
    data = SchedulerData(
        flow_fields=np.zeros((count, 20, 20, 2), dtype=np.float32),
        images1=np.zeros((count, 8, 8), dtype=np.float32),
        images2=np.zeros((count, 8, 8), dtype=np.float32),
        mask=mask,
    )
    scheduler = MagicMock(spec=SchedulerProtocol)
    scheduler.include_images = True
    scheduler.get_batch.return_value = data
    scheduler.get_flow_fields_shape.return_value = (20, 20, 2)

    sampler: RealImageSampler | SyntheticImageSampler
    if sampler_kind == "real":
        sampler = RealImageSampler(scheduler=scheduler, batch_size=count)
    else:
        # One flow produces two image pairs; mask metadata must expand too.
        scheduler.get_batch.return_value = data.update(
            flow_fields=data.flow_fields[:1],
            mask=None if mask is None else mask[1:],
        )
        sampler = SyntheticImageSampler(
            scheduler=scheduler,
            batches_per_flow_batch=2,
            flow_fields_per_batch=1,
            flow_field_size=(20.0, 20.0),
            resolution=1.0,
            velocities_per_pixel=1.0,
            seed=42,
            max_speed_x=0.0,
            max_speed_y=0.0,
            min_speed_x=0.0,
            min_speed_y=0.0,
            output_units="pixels",
            device_ids=[0],
            generation_specification=ImageGenerationSpecification(
                batch_size=count, image_shape=(8, 8), img_offset=(2, 2)
            ),
        )

    try:
        for _ in range(2):
            batch = next(sampler)
            assert batch.mask is not None
            assert batch.mask.dtype == jnp.bool_
            assert batch.mask.shape == (count,)
            expected = (
                np.ones(count, dtype=bool)
                if mask is None
                else (
                    np.zeros(count, dtype=bool)
                    if sampler_kind == "synthetic"
                    else mask.astype(bool)
                )
            )
            np.testing.assert_array_equal(batch.mask, expected)
            assert batch.images1[batch.mask].shape[0] == np.count_nonzero(
                expected
            )
            assert batch.flow_fields[batch.mask].shape[0] == np.count_nonzero(
                expected
            )
    finally:
        sampler.shutdown()
