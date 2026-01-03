"""Speed comparison between Legacy and Grain schedulers."""

import os
import gc
import threading
import time
import timeit

import grain.python as grain
import pytest

from synthpix.data_sources.adapter import GrainEpisodicAdapter
from synthpix.data_sources.episodic import EpisodicDataSource
from synthpix.data_sources.mat import MATDataSource
from synthpix.data_sources.numpy import NumpyDataSource
from synthpix.scheduler.episodic import EpisodicFlowFieldScheduler
from synthpix.scheduler.mat import MATFlowFieldScheduler


@pytest.mark.parametrize(
    "mock_mat_files", 
    [{"num_files": 200, "dims": {"height": 256, "width": 256}}], 
    indirect=True
)
def test_benchmark_legacy_vs_grain(tmp_path, mock_mat_files):
    """Benchmark and compare the throughput of Legacy and Grain schedulers.

    Iterates through batches using both stacks and prints a performance 
    table comparing Legacy (fixed baseline) against Grain with varying 
    thread counts.
    
    Uses 256x256 input files (via fixture) to match output shape, eliminating 
    resize overhead to isolate I/O.
    """
    # Unpack fixture
    _, _ = mock_mat_files  # Files are already created in tmp_path
    
    num_files = 200
    episode_length = 5
    batch_size = 1
    dataset_dir = tmp_path
    steps = 500
    grain_thread_counts = [None, 1, 2, 4, 8, 16, 32, 64]

    print(f"\n[Benchmark] Running {steps} steps per configuration (256x256 input)...")

    def loop_system(sys_next_fn, reset_fn, n_steps):
        count = 0
        while count < n_steps:
            try:
                sys_next_fn()
                count += 1
            except (StopIteration, Exception):
                reset_fn()

    # 1. Benchmark Legacy Stack (Single-threaded baseline)
    legacy_base = MATFlowFieldScheduler(
        file_list=[str(dataset_dir)],
        include_images=True,
        output_shape=(256, 256),
        randomize=False,
    )
    legacy_episodic = EpisodicFlowFieldScheduler(
        scheduler=legacy_base,
        batch_size=batch_size,
        episode_length=episode_length,
    )

    t_legacy = timeit.timeit(
        lambda: loop_system(
            lambda: legacy_episodic.get_batch(batch_size),
            legacy_episodic.reset,
            steps,
        ),
        number=1,
    )

    print(f"\nLegacy Baseline: {t_legacy:.4f} s")
    print(f"\n{'Grain Threads':<15} | {'Duration (s)':<15} | {'Speedup vs Legacy':<20}")
    print("-" * 55)

    results = {}

    # 2. Benchmark Grain Stack with varying thread counts
    for num_threads in grain_thread_counts:
        label = "Default" if num_threads is None else str(num_threads)
        grain_source = MATDataSource(
            dataset_path=str(dataset_dir),
            include_images=True,
            output_shape=(256, 256),
        )
        grain_episodic_ds = EpisodicDataSource(
            source=grain_source,
            batch_size=batch_size,
            episode_length=episode_length,
            seed=42,
        )

        read_options = grain.ReadOptions(num_threads=num_threads) if num_threads is not None else None
        grain_loader = grain.DataLoader(
            data_source=grain_episodic_ds,
            sampler=grain.IndexSampler(
                num_records=len(grain_episodic_ds),
                shuffle=False,
                shard_options=grain.NoSharding(),
                num_epochs=1,
            ),
            worker_count=0,
            read_options=read_options,
            operations=[grain.Batch(batch_size=batch_size)],
        )

        grain_adapter = GrainEpisodicAdapter(grain_loader)

        t_grain = timeit.timeit(
            lambda: loop_system(
                lambda: grain_adapter.get_batch(batch_size),
                grain_adapter.reset,
                steps,
            ),
            number=1,
        )
        
        del grain_loader
        del grain_adapter
        gc.collect()
        
        # Cleanup goggles if active
        try:
            import goggles as gg
            gg.finish()
        except ImportError:
            pass
            
        time.sleep(0.5)

        speedup = t_legacy / t_grain
        results[num_threads] = t_grain
        print(f"{label:<15} | {t_grain:<15.4f} | {speedup:<20.2f}x")

    # Assertions for speedup
    # Grain should be at least slightly faster even with 1 thread due to better implementation,
    # and significantly faster with multiple threads/default config.
    assert results[1] < t_legacy * 1.5, f"Grain with 1 thread is too slow: {results[1]:.4f}s vs Legacy {t_legacy:.4f}s"
    assert results[16] < t_legacy, f"Grain with 16 threads should be faster than Legacy: {results[16]:.4f}s vs {t_legacy:.4f}s"
    assert results[None] < t_legacy, f"Grain default should be faster than Legacy: {results[None]:.4f}s vs {t_legacy:.4f}s"


@pytest.mark.parametrize(
    "mock_numpy_files",
    [{"num_files": 200, "dims": {"height": 256, "width": 256}}],
    indirect=True
)
def test_benchmark_numpy_speed(tmp_path, mock_numpy_files):
    """Benchmark Grain with NumpyDataSource to show threading scaling.
    
    Unlike HDF5/MAT, Numpy (~np.load) releases the GIL, allowing Grain to 
    actually parallelize data loading across multiple threads.
    """
    # Unpack fixture
    _, _ = mock_numpy_files
    
    dataset_dir = tmp_path
    steps = 200
    batch_size = 1
    episode_length = 5
    thread_counts = [1, 2, 4, 8]

    print(f"\n[Numpy Benchmark] Running {steps} steps per configuration...")
    print(f"\n{'Threads':<10} | {'Duration (s)':<15} | {'Speedup vs 1T':<15}")
    print("-" * 45)

    def loop_system(sys_next_fn, reset_fn, n_steps):
        count = 0
        while count < n_steps:
            try:
                sys_next_fn()
                count += 1
            except (StopIteration, Exception):
                reset_fn()

    results = {}

    for num_threads in thread_counts:
        # We use a fresh data source and loader for each run to be safe
        source = NumpyDataSource(
            dataset_path=str(dataset_dir),
            include_images=True,
        )
        episodic_ds = EpisodicDataSource(
            source=source,
            batch_size=batch_size,
            episode_length=episode_length,
            seed=42,
        )

        read_options = grain.ReadOptions(num_threads=num_threads)
        loader = grain.DataLoader(
            data_source=episodic_ds,
            sampler=grain.IndexSampler(
                num_records=len(episodic_ds),
                shuffle=False,
                shard_options=grain.NoSharding(),
                num_epochs=1,
            ),
            worker_count=0,
            read_options=read_options,
            operations=[grain.Batch(batch_size=batch_size)],
        )

        adapter = GrainEpisodicAdapter(loader)

        t = timeit.timeit(
            lambda: loop_system(
                lambda: adapter.get_batch(batch_size),
                adapter.reset,
                steps,
            ),
            number=1,
        )

        results[num_threads] = t
        speedup = results[1] / t
        print(f"{num_threads:<10} | {t:<15.4f} | {speedup:<15.2f}x")

        # Cleanup
        del loader
        del adapter
        gc.collect()
        try:
            import goggles as gg
            gg.finish()
        except ImportError:
            pass
        time.sleep(0.5)

    # Verify scaling: 4 threads should be noticeably faster than 1 thread
    assert results[4] < results[1] * 0.9, f"Expected at least 10% speedup with 4 threads for Numpy, but got {results[4]:.4f}s vs {results[1]:.4f}s"



class SleepingDataSource(grain.RandomAccessDataSource):
    """Mock data source that simulates high-latency I/O using `time.sleep`.
    
    Used to verify that the data loader correctly parallelizes data 
    loading across multiple threads or processes.
    """
    def __init__(self, size=10, sleep_time=0.1):
        super().__init__()
        self.size = size
        self.sleep_time = sleep_time

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        time.sleep(self.sleep_time)
        return {"idx": idx, "pid": os.getpid(), "thread": threading.get_ident()}

def test_grain_threading_speedup():
    """Verify that Grain achieves significant speedup using multi-threading.

    Compares the total duration of loading 'slow' records across different 
    thread counts to verify parallel execution. Also verifies that Grain's 
    default configuration (worker_count=0, read_options=None) enables 
    multi-threading automatically.
    """
    size = 64
    sleep_time = 0.1
    # We include None as 'Default' to verify Grain's automatic multi-threading
    thread_counts = [None, 1, 2, 4, 8, 16, 32, 64]
    results = {}

    print(f"\n{'Threads':<10} | {'Duration (s)':<15} | {'Unique Threads':<15} | {'Speedup':<10}")
    print("-" * 60)

    # We use 1 thread as the baseline for speedup calculations
    one_thread_duration = None

    for num_threads in thread_counts:
        read_options = grain.ReadOptions(num_threads=num_threads) if num_threads is not None else None
        
        ds = SleepingDataSource(size=size, sleep_time=sleep_time)
        loader = grain.DataLoader(
            data_source=ds,
            sampler=grain.IndexSampler(
                num_records=len(ds),
                shuffle=False,
                shard_options=grain.NoSharding(),
                num_epochs=1,
            ),
            worker_count=0,
            read_options=read_options,
            operations=[grain.Batch(batch_size=1, drop_remainder=False)],
        )

        start = time.perf_counter()
        threads = set()
        for batch in loader:
            threads.add(batch["thread"][0])
        duration = time.perf_counter() - start

        if num_threads == 1:
            one_thread_duration = duration

        results[num_threads] = (duration, len(threads))

    # Print results after we have the one_thread_duration
    for num_threads in thread_counts:
        duration, num_unique_threads = results[num_threads]
        label = "Default" if num_threads is None else str(num_threads)
        speedup = one_thread_duration / duration if one_thread_duration else 1.0
        print(f"{label:<10} | {duration:<15.4f} | {num_unique_threads:<15} | {speedup:<10.2f}x")

    # Verify that multi-threading actually happened and improved performance
    # 1. Single thread should take at least size * sleep_time
    assert results[1][0] >= (size * sleep_time), f"Expected 1 thread to take >= {size * sleep_time}s, got {results[1][0]:.4f}s"

    # 2. Max threads should be faster than single thread
    assert results[16][0] < results[1][0], f"Expected 16 threads ({results[16][0]:.4f}s) to be faster than 1 thread ({results[1][0]:.4f}s)"

    # 3. Max threads should use multiple threads
    assert results[16][1] > 1, f"Expected 16 threads to use > 1 unique thread ID, got {results[16][1]}"

    # 4. Check for scaling: 4 threads should be roughly faster than 2 threads, etc.
    # We use a lenient margin because overhead/GIL can reduce ideal scaling.
    assert results[4][0] < results[1][0], f"Expected 4 threads ({results[4][0]:.4f}s) to be faster than 1 thread ({results[1][0]:.4f}s)"
    assert results[8][0] < results[2][0], f"Expected 8 threads ({results[8][0]:.4f}s) to be faster than 2 threads ({results[2][0]:.4f}s)"

    # 5. Verify Default configuration (read_options=None)
    # It should use multiple threads and be significantly faster than 1 thread
    default_duration, default_threads = results[None]
    assert default_threads > 1, f"Default configuration only used {default_threads} thread(s)"
    assert default_duration < results[1][0], "Default configuration was not faster than 1 thread"
