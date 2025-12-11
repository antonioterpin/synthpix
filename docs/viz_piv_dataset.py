# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: flowgym
#     language: python
#     name: python3
# ---

# %% [markdown]
# Notebook for visualizing the PIV dataset and testing flow estimators.


# %% [markdown]
# This notebook will be your go-to testing toolkit.
# The workflow encompasses the following key steps:
#
# - **Image Generation:**
#   With synthpix, generate batches of image pairs with their ground truth flow fields.
# - **Flow Estimation:**
#   Multiple methods (e.g., DIS, deep flow, openPIV etc.) are already available for you
#   to try out and estimate the flow between the synthetic images.
#   We encourage you to implement and test your own here!
# - **Evaluation Metrics:**
#   The results include metrics such as angle error, magnitude error, end-point-error
#   to assess the performance of each estimator.
#
# This notebook uses the fluids referenced from the piv_dataset.

# %% [markdown]
# Let's import the necessary libraries...
#
# if you're working in a multi-GPU setup remember to choose which GPUs to use,
# this notebook will occupy them.

# %%
import os
import sys


def setup_paths(target_dir=".."):
    """Set up the paths for the notebook to run correctly."""
    os.chdir(target_dir)
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)
    print(f"Current working directory: {os.getcwd()}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


setup_paths()

import synthpix
import numpy as np
import jax.numpy as jnp
import jax

# Utils
import flow_estimator.utils as utils
from flow_estimator.flow.utils import compute_divergence_and_vorticity, viz
from flow_estimator.utils import flow_magnitude_heatmap, load_configuration
from flow_estimator.make import make_estimator
from flow_estimator.common.evaluation import angle_error

# Visualization
import ipywidgets as wd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from analyses.utils import visualize_errors


# %% [markdown]
# Initialize the synthetic image generation process:
# All dataset parameters can be found in:
# ```
# config/piv_dataset_class1_eval.yaml
# ```

# %%
# Select the dataset config
dataset_path = "piv_dataset_class1_eval.yaml"
dataset_config = utils.load_configuration(dataset_path)

# Load the sampler
sampler = synthpix.make(dataset_config)

# %% [markdown]
# Let's use the sampler to generate a batch of images and the relative flow field.
# To make sure the flow is interesting to look at we set a minimum value that the
# ground truth should be above (nobody wants to be looking at all zeroes... duh!)

# %%
gt = jnp.zeros((1, 512, 512, 2), dtype=jnp.float32)
while jnp.max(jnp.abs(gt)) < 0.00001:
    batch = next(sampler)
    img1 = batch["images1"]
    img2 = batch["images2"]
    gt = batch["flow_fields"]

# %% [markdown]
# From here onwards we will visualize our results on one image couple at a time,
# but the losses and errors will be computed on the entire batch.
# Select which image to use by changing ```sample_idx```

# %%
# Display the images side by side
sample_idx = 6
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(np.asarray(img1[sample_idx, ...]), cmap="gray")
axs[0].set_title("Image 1")
axs[0].axis("off")

axs[1].imshow(np.asarray(img2[sample_idx, ...]), cmap="gray")
axs[1].set_title("Image 2")
axs[1].axis("off")

plt.tight_layout()
plt.show()

# %%
divergence, vorticity = compute_divergence_and_vorticity(gt)

VISUALIZE_FLOW = True
VISUALIZE_DIVERGENCE = False
VISUALIZE_VORTICITY = False

flow_field_sample = gt[sample_idx, 1:-1, 1:-1, :]
vorticity_sample = vorticity[sample_idx, ...]
divergence_sample = divergence[sample_idx, ...]
magnitude = np.linalg.norm(flow_field_sample, axis=-1)

if VISUALIZE_FLOW:
    viz(
        np.asarray(flow_field_sample),
        scalar_field=np.asarray(magnitude),
        scalar_label="Magnitude",
        bilinear=True,
        arrow_stride=8,
    )
    # To keep the scale equal between gt and estimated flow
    mag = np.linalg.norm(flow_field_sample, axis=-1)  # (H, W)
    maxrad = np.max(mag) + 1e-6  # avoid divide-by-zero

    color = flow_magnitude_heatmap(np.asarray(flow_field_sample), maxrad=maxrad)
    plt.imshow(color)
if VISUALIZE_DIVERGENCE:
    plt.figure(figsize=(10, 5))
    plt.imshow(np.asarray(divergence_sample), cmap="jet")
    plt.colorbar(label="Divergence")
    plt.title("Divergence")
if VISUALIZE_VORTICITY:
    plt.figure(figsize=(10, 5))
    plt.imshow(np.asarray(vorticity_sample), cmap="jet")
    plt.colorbar(label="Vorticity")
    plt.title("Vorticity")


# %% [markdown]
# This is where the magic happens!
# We've provided you with plenty of PIV and Optical Flow algorithms to experiment with.
#
# Just change the configuration file's path to one of those you can find in the folder:
# ```
# ../../config/estimators
# ```
#
# and see which one performs best.

# %%
estimator_config = load_configuration("estimators/flow/piv_dataset/dis/cylinder.yaml")

# Create the estimator
(trainable_state, create_state_fn, compute_flow_fn, model) = make_estimator(
    estimator_config,
    image_shape=dataset_config["image_shape"],
    estimate_shape=gt.shape,
)

# %% [markdown]
# Let's instantiate the model and run it for one iteration

# %%
key = jax.random.PRNGKey(0)
estimation_state = create_state_fn(img1, key)
estimation_state = compute_flow_fn(
    img2, estimation_state, None
)  # TODO : add trainable state
flow_field = estimation_state.history_estimates[:, -1]

# %% [markdown]
# Let's also visualize the estimated flow along with divergence and vorticity

# %%
divergence, vorticity = compute_divergence_and_vorticity(flow_field)

VISUALIZE_FLOW = True
VISUALIZE_DIVERGENCE = False
VISUALIZE_VORTICITY = False

flow_field_sample = flow_field[sample_idx, 1:-1, 1:-1, :]
vorticity_sample = vorticity[sample_idx, ...]
divergence_sample = divergence[sample_idx, ...]
magnitude = np.linalg.norm(flow_field_sample, axis=-1)

if VISUALIZE_FLOW:
    viz(
        np.asarray(flow_field_sample),
        scalar_field=np.asarray(magnitude),
        scalar_label="Magnitude",
        bilinear=True,
        arrow_stride=8,
    )
    color = flow_magnitude_heatmap(np.asarray(flow_field_sample), maxrad=maxrad)
    plt.imshow(color)
if VISUALIZE_DIVERGENCE:
    plt.figure(figsize=(10, 5))
    plt.imshow(np.asarray(divergence_sample), cmap="jet")
    plt.colorbar(label="Divergence")
    plt.title("Divergence")
if VISUALIZE_VORTICITY:
    plt.figure(figsize=(10, 5))
    plt.imshow(np.asarray(vorticity_sample), cmap="jet")
    plt.colorbar(label="Vorticity")
    plt.title("Vorticity")

# %% [markdown]
# How well does the model perform? Let's take a look with angle, magnitude, and total loss

# %%
VISUALIZE_ERRORS = True

if VISUALIZE_ERRORS:
    visualize_errors(
        flow_field[sample_idx, 1:-1, 1:-1, :], gt[sample_idx, 1:-1, 1:-1, :]
    )

err_angle = angle_error(flow_field, gt)
magnitude = jnp.linalg.norm(flow_field, axis=-1)
magnitude_gt = jnp.linalg.norm(gt, axis=-1)
max_magnitude_gt = jnp.maximum(jnp.max(magnitude_gt), 1e-10)
err_magnitude = jnp.abs(magnitude - magnitude_gt) / max_magnitude_gt
loss = err_angle + err_magnitude

print("Angle error: ", jnp.mean(err_angle))
print("Magnitude error: ", jnp.mean(err_magnitude))
print("Loss: ", jnp.mean(loss))

# %% [markdown]
# Let's put it all together in one spot!
# Once you've run all the previous cells, here you can tune your estimator
# on a fixed batch and get more batches to explore how your model performs
# in all scenarios from your dataset. You can also automatically keep
# or delete batches based on their EPE and save them if you like them!

# %%
# Instantiate the sampler with the dataset configuration
dataset_config = utils.load_configuration("piv_dataset_class1_eval_original.yaml")
sampler = synthpix.make(
    dataset_config, episode_length=30, buffer_size=100, images_from_file=True
)

# %%

# ── Interactive batch explorer (sampler / estimator + native metrics) ──────
# Needs these objects, already defined earlier in the notebook:
#   • sampler  – iterator → (img1, img2, flow_gt, …)
#   • create_state_fn(img1)  and  compute_flow_fn(img2, state)
#   • flow_to_rgb(flow)          – helper that turns flow into an RGB image
#   • visualize_errors(pred, gt) – makes a matplotlib figure of error maps
#   • angle_error(pred, gt)      – pixel-wise angular error  (B,H,W)
# --------------------------------------------------------------------------
# widgets ------------------------------------------------------------------

# Buttons for batch control
btn_next = wd.Button(description="▶  Next batch", button_style="success")
btn_next.layout = wd.Layout(width="200px")
btn_again = wd.Button(description="↻  Recompute current batch", button_style="warning")
btn_again.layout = wd.Layout(width="200px")
btn_loop = wd.Button(description="🔁  Loop until good batch", button_style="primary")
btn_loop.layout = wd.Layout(width="200px")
btn_save = wd.Button(description="💾  Save current batch", button_style="info")
btn_save.layout = wd.Layout(width="200px")

button_row = wd.HBox([btn_next, btn_again, btn_loop, btn_save])

out_best = wd.Output()
slider = wd.IntSlider(
    description="Sample", min=0, max=0, value=0, continuous_update=False
)
out_detail = wd.Output()
ui = wd.VBox([button_row, out_best, slider, out_detail])
display(ui)
out_console = wd.Output()
display(out_console)


# globals for the current batch -------------------------------------------
img1 = img2 = flow_gt = pred_flow = None
losses = aae = ame = epe = None


# helpers ------------------------------------------------------------------
def _show_tbl(df, title):
    display(wd.HTML(f"<h4>{title}</h4>"))
    display(df.style.format("{:.4f}").hide(axis="index"))


def _to_np(x):
    """Ensure x is a plain NumPy ndarray (needed for OpenCV)."""
    return np.asarray(x)  # works for JAX, NumPy, and Python lists


def _flow_img(flow, maxrad=None):
    if maxrad is None:
        return flow_magnitude_heatmap(_to_np(flow))
    return flow_magnitude_heatmap(_to_np(flow), maxrad=maxrad)


def _flow_mag(flow):
    return np.linalg.norm(_to_np(flow), axis=-1)


def _refresh_bestworst():
    """Show Top-5 / Bottom-5 flows with their metrics under each pair."""
    with out_best:
        clear_output(wait=True)

        idx_sorted = jnp.argsort(epe)
        groups = [
            ("Top-5 flows (lowest EPE)", idx_sorted[:5]),
            ("Bottom-5 flows (highest EPE)", idx_sorted[-5:][::-1]),
        ]

        # ── tune these two lines to taste ──────────────────────────────────
        FIGSIZE = (18, 9)  # width, height in inches
        METRIC_FSIZE = 18  # font-size for the numbers
        # -------------------------------------------------------------------

        for title, inds in groups:

            # rows: GT  |  pred  |  metrics text
            height_ratios = [1, 1, 0.12, 0.12]  # <‒ adjust 0.35 to taste

            fig, ax = plt.subplots(
                4,
                5,
                figsize=FIGSIZE,
                gridspec_kw={"height_ratios": height_ratios},
                constrained_layout=True,
            )
            fig.suptitle(title, fontsize=18, y=1.02)

            for col, i in enumerate(inds):

                # Compute per-column max magnitude from both GT and prediction
                gt_mag = _flow_mag(flow_gt[i])
                pre_flow_mag = _flow_mag(pred_flow[i])
                col_max = np.max(gt_mag) + 1e-6  # avoid zero division

                # ── Ground truth image ─────────────────────────────
                ax_gt = ax[0, col]
                _ = ax_gt.imshow(gt_mag, cmap="jet", vmin=0, vmax=col_max)
                ax_gt.set_title(f"GT {int(i)}", fontsize=12)
                # ax_gt.axis("off")

                # ── Prediction image ───────────────────────────────
                ax_pred = ax[1, col]
                _ = ax_pred.imshow(pre_flow_mag, cmap="jet", vmin=0, vmax=col_max)
                ax_pred.set_title("pred", fontsize=12)
                # ax_pred.axis("off")

                # ── Shared colorbar under the column ───────────────
                _ = ax[0, col].imshow(
                    _flow_img(_to_np(flow_gt[i]), maxrad=col_max),
                    cmap="jet",
                    vmin=0,
                    vmax=col_max,
                )
                _ = ax[1, col].imshow(
                    _flow_img(_to_np(pred_flow[i]), maxrad=col_max),
                    cmap="jet",
                    vmin=0,
                    vmax=col_max,
                )
                ax_cbar = ax[2, col]
                im = ax[0, col].images[0]  # Grab the image from pred row
                _ = plt.colorbar(im, cax=ax_cbar, orientation="horizontal")
                ax_cbar.set_title("Flow magnitude", fontsize=10)

                # ── Metrics text (no image) ─────────────────────────
                txt = (
                    f"loss = {losses[i]:.3f}\n"
                    f"ang  = {aae[i]:.3f}\n"
                    f"mag  = {ame[i]:.3f}\n"
                    f"epe  = {epe[i]:.3f}\n"
                )
                ax[3, col].text(
                    0.5, 0.5, txt, ha="center", va="center", fontsize=METRIC_FSIZE
                )
                ax[3, col].axis("off")

    return None


def _refresh_detail(change=None):
    i = slider.value
    with out_detail:
        clear_output(wait=True)

        # 1) image pair + flows
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        gt_mag = _flow_mag(flow_gt[i])
        col_max = np.max(gt_mag) + 1e-6  # avoid zero division
        ax[0, 0].imshow(img1[i])
        ax[0, 0].set_title("Image 1")
        ax[0, 1].imshow(img2[i])
        ax[0, 1].set_title("Image 2")
        ax[1, 0].imshow(_flow_img(_to_np(flow_gt[i]), maxrad=col_max))
        ax[1, 0].set_title("GT flow")
        ax[1, 1].imshow(_flow_img(_to_np(pred_flow[i]), maxrad=col_max))
        ax[1, 1].set_title("Pred flow")
        for a in ax.flatten():
            a.axis("off")

        plt.tight_layout()

        # 2) pixel-wise error graphics
        visualize_errors(pred_flow[i, :, :, :], flow_gt[i, :, :, :])

        # 3) scalar metrics
        print(
            f"loss={losses[i]:.4f}   |   " f"angle={aae[i]:.4f}   |   mag={ame[i]:.4f}"
        )
    return None


def _sample_and_evaluate(_=None, again=False, loop=False):
    global img1, img2, flow_gt, pred_flow, losses, aae, ame, epe

    with out_console:
        clear_output(wait=True)

        not_good_enough = True
        while not_good_enough:

            # 1) Fetch a fresh batch if requested
            if not again:
                batch = next(sampler)
                img1 = batch["images1"]
                img2 = batch["images2"]
                flow_gt = batch["flow_fields"]
            else:
                not_good_enough = False
                print("Recomputing the current batch...")

            # 2) Instantiate and run the estimator
            estimator_config = load_configuration(
                "estimators/flow/piv_dataset/dis/cylinder.yaml"
            )
            (_, create_state_fn, compute_flow_fn, _) = make_estimator(
                estimator_config,
                image_shape=dataset_config["image_shape"],
                estimate_shape=flow_gt.shape,
            )
            state = create_state_fn(img1, jax.random.PRNGKey(0))
            state = compute_flow_fn(img2, state, None)
            pred_flow = state.history_estimates[:, -1]  # (B,H,W,2)

            # 3) Compute metrics
            err_angle = angle_error(pred_flow, flow_gt)  # (B,H,W)
            mag_pred = jnp.linalg.norm(pred_flow, axis=-1)  # (B,H,W)
            mag_gt = jnp.linalg.norm(flow_gt, axis=-1)
            max_mag_gt = jnp.maximum(jnp.max(mag_gt, axis=(1, 2), keepdims=True), 1e-10)
            err_mag = jnp.abs(mag_pred - mag_gt) / max_mag_gt  # (B,H,W)
            per_px_loss = err_angle + err_mag  # (B,H,W)

            # compute epe
            epe_map = jnp.linalg.norm(pred_flow - flow_gt, axis=-1)

            # per-sample reductions
            aae = jnp.mean(err_angle, axis=(1, 2))
            ame = jnp.mean(err_mag, axis=(1, 2))
            losses = jnp.mean(per_px_loss, axis=(1, 2))
            epe = jnp.mean(epe_map, axis=(1, 2))

            # Compute the per-pixel displacement (use ground truth or pred)
            disp = jnp.linalg.norm(flow_gt, axis=-1)  # or pred_flow

            # Define your displacement range
            disp_min1 = 0
            disp_max1 = 1.5
            disp_min2 = 1.5
            disp_max2 = 3
            disp_min3 = 3
            disp_max3 = 50

            # Create a mask for the selected displacement range
            mask1 = (disp >= disp_min1) & (disp < disp_max1)
            mask2 = (disp >= disp_min2) & (disp < disp_max2)
            mask3 = (disp >= disp_min3) & (disp < disp_max3)

            # Compute the masked mean EPE
            epe_masked1 = jnp.mean(epe_map[mask1])
            epe_masked2 = jnp.mean(epe_map[mask2])
            epe_masked3 = jnp.mean(epe_map[mask3])

            print(f"EPE total: {jnp.mean(epe_map):.4f}")
            print(f"EPE in range [{disp_min1}, {disp_max1}):", epe_masked1)
            print(f"EPE in range [{disp_min2}, {disp_max2}):", epe_masked2)
            print(f"EPE in range [{disp_min3}, {disp_max3}):", epe_masked3)

            # 4) Update slider & visuals
            slider.max = img1.shape[0] - 1
            slider.value = 0
            _refresh_bestworst()
            _refresh_detail()

            if loop:
                if jnp.mean(epe_map) < 0.1:
                    print(f"Batch is good enough, {jnp.mean(epe_map):.4f} < 0.1")
                    not_good_enough = False
                else:
                    sampler.next_episode()
            else:
                not_good_enough = False

        return None


def save_current_batch(_=None):
    """Save the current batch to a file.

    This will save the ground truth and predicted flow fields as images and numpy arrays.
    """
    for i in range(pred_flow.shape[0]):
        col_max = np.max(_flow_mag(flow_gt[i])) + 1e-6
        gt_img = _flow_img(flow_gt[i], maxrad=col_max)
        pred_img = _flow_img(pred_flow[i], maxrad=col_max)

        os.makedirs("heatmaps", exist_ok=True)
        os.makedirs("flowfields", exist_ok=True)

        plt.imsave(f"heatmaps/gt_{i:04d}.png", gt_img)
        plt.imsave(f"heatmaps/pred_{i:04d}.png", pred_img)
        np.save(f"flowfields/gt_{i:04d}.npy", np.asarray(flow_gt[i]))
        np.save(f"flowfields/pred_{i:04d}.npy", np.asarray(pred_flow[i]))

    return None


# connect widgets
btn_next.on_click(_sample_and_evaluate)
btn_again.on_click(lambda _: _sample_and_evaluate(again=True))
btn_save.on_click(save_current_batch)
btn_loop.on_click(lambda _: _sample_and_evaluate(loop=True))
slider.observe(_refresh_detail, names="value")

# kick-off with the first batch
_sample_and_evaluate()

# %% [markdown]
# What you might know empirically is that one of the biggest factors that impact
# prediction accuracy is the seeding density of your fluid. Let's see if that's true

# %%
# We'll evaluate a few batches to see how loss varies with seeding density.
num_batches = 20
density_list = []
loss_list = []

for _ in range(num_batches):
    # Get a fresh batch; densities is provided by the sampler.
    batch = next(sampler)
    img1 = batch["images1"]
    img2 = batch["images2"]
    flow_gt = batch["flow_fields"]
    densities = batch["params"][
        "seeding_densities"
    ]  # Assuming this is provided by the sampler

    # Run the estimator.
    state = create_state_fn(img1, jax.random.PRNGKey(0))
    state = compute_flow_fn(img2, state, None)
    pred_flow = state.history_estimates[:, -1]

    # Compute the per-pixel losses.
    err_angle = angle_error(pred_flow, flow_gt)
    mag_pred = jnp.linalg.norm(pred_flow, axis=-1)
    mag_gt = jnp.linalg.norm(flow_gt, axis=-1)
    max_mag = jnp.maximum(jnp.max(mag_gt, axis=(1, 2), keepdims=True), 1e-10)
    err_mag = jnp.abs(mag_pred - mag_gt) / max_mag
    per_px_loss = err_angle + err_mag

    # Compute mean loss per image.
    losses = jnp.mean(per_px_loss, axis=(1, 2))

    # Collect densities and losses (flattening the batch dimensions).
    density_list.extend(np.array(densities).flatten().tolist())
    loss_list.extend(np.array(losses).flatten().tolist())

# Plot loss as a function of seeding density.
plt.figure(figsize=(8, 6))
plt.scatter(density_list, loss_list)
plt.xlabel("Seeding Density")
plt.ylabel("Loss")
plt.title("Loss vs Seeding Density")
plt.show()
