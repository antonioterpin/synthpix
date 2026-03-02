"""Benchmark identifiable-flow overhead on sampler throughput.

Measures throughput (image pairs / second) with and without identifiable
flow enabled, asserting that disabling identifiable flow yields higher
throughput.
"""

from __future__ import annotations

import pytest
from synthpix.utils import benchmark_throughput

@pytest.mark.run_explicitly # marks test as not run by default
def test_identifiable_flow_throughput():
    """Throughput without identifiable flow should exceed throughput with it.
    """
    # environment settings
    CONFIG_PATH = "config/test_benchmark.yaml"
    NUM_BATCHES = 10000

    # test
    # computes throughput with identifiable_flow control flow option disabled
    throughput_base = benchmark_throughput(
        CONFIG_PATH,
        batches=NUM_BATCHES,
        use_identifiable_flow=False,
    )
    print( # visible with "-s" flag in pytest
        f"\nThroughput Benchmark: disabled identifiable flow: {throughput_base} pairs/s"
    )

    # computes throughput with identifiable_flow control flow option enabled
    throughput_identifiable_flow = benchmark_throughput(
        CONFIG_PATH,
        batches=NUM_BATCHES,
        use_identifiable_flow=True,
    )
    print( # visible with "-s" flag in pytest
        f"Throughput Benchmark: enabled identifiable flow:    {throughput_identifiable_flow} pairs/s"
    )

    # assert
    assert throughput_base > throughput_identifiable_flow, (
        f"Expected higher throughput without identifiable flow, "
        f"got {throughput_base} pairs/s vs {throughput_identifiable_flow} pairs/s"
    )
