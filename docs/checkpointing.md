# Checkpointing and reproducibility in SynthPix 💾

SynthPix provides a robust, bit-perfect checkpointing system built on top of [Orbax](https://orbax.readthedocs.io/). Developing data-hungry PIV algorithms requires large-scale training, and being able to pause/resume with guaranteed deterministic results is critical.

## Core concepts

### Bit-perfect reproducibility
SynthPix guarantees that the images generated after a restore are **identical** to those that would have been generated if the run had never stopped. This includes:
1.  **Data order**: Ensuring exactly the same flow field records are loaded from disk in the same order.
2.  **Randomness**: Synchronizing the JAX RNG state on the GPU with the data loader's position on the CPU.
3.  **Variation logic**: Preserving repetition counters (e.g., when `batches_per_flow_batch > 1`).

### Unified state
A SynthPix pipeline has two main state components:
-   **CPU state (Grain)**: Manages file pointers, shuffling logic, and the deterministic `jax_seed` passed to the GPU.
-   **GPU state (Sampler)**: Manages internal iteration counters and (optionally) its own JAX RNG key for legacy paths.

Orbax acts as the transaction layer that saves both states atomically.

---

## API Usage

### Saving a checkpoint
Use the high-level `save_checkpoint` function.

```python
import synthpix

# ... inside your training loop ...
synthpix.save_checkpoint(
    checkpoint_dir="path/to/checkpoints",
    sampler=sampler,
    step=global_step,
    max_to_keep=3
)
```

### Restoring a pipeline
The `make` function can automatically restore the entire pipeline state if the `load_from` argument is provided.

```python
import synthpix

# Resume from the latest checkpoint in the directory
sampler = synthpix.make(
    config_path,
    load_from="path/to/checkpoints"
)

# The sampler is now at the exact state it was when saved.
# You can immediately continue taking batches.
for batch in sampler:
    ...
```

---

## Architecture

### How it works
The `Sampler` base class is the orchestrator for state. It implements `get_state()` and `set_state()`, which:
1.  Captures/Restores internal sampler metadata (e.g., `_step`, `_batches_generated`).
2.  Recursively calls the underlying `Scheduler` state methods.

When `synthpix.save_checkpoint` is called, it:
1.  Retrieves the sampler's dictionary state.
2.  Retrieves the binary state of the Grain iterator (`grain.PyGrainCheckpointSave`).
3.  Packs them into a `Composite` Orbax argument for atomic saving.

### Randomness
To ensure deterministic randomness, SynthPix uses a **derivation** strategy:
- **Grain** provides a base `jax_seed` for every flow field record.
- **Sampler** maintains a `_batches_generated` counter.
- The final noise key is derived as: `jax.random.fold_in(PRNGKey(jax_seed), batches_generated)`.

This makes the noise for "variation #3 of flow #401" stable and independent of the resume point.

---

## Best practices

1.  **Checkpoint frequency**: Checkpointing the data pipeline is lightweight. Saving every few thousand steps is recommended.
2.  **Storage**: Use a persistent storage location. Orbax creates subdirectories named by the `step` number.
3.  **Manual cleanup**: Use `max_to_keep` in `save_checkpoint` to prevent your disk from filling up with old checkpoints.
