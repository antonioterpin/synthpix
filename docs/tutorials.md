# More examples 🌊

## Generating image pairs from custom flow data

To maintain support and facilitate comparisons with the already established piv_dataset, the most supported file format is ``.mat``. Each file should contain the key:
- ``flow``

and the locations of the files should be provided in the configuration file under ``scheduler_files``. The parameter can either provide ``SynthPix`` with the directory of the folder containing the files, in which case all subdirectories will be opened to check for files, as well as a list of file directories like:

```yaml
scheduler_files:
- file1.mat
- file2.mat
```

Here's an example of what the full config should look like:

```yaml
# Dataset parameters
seed: 0 # Random seed for reproducibility
randomize: false # Whether to randomize the order of the files
loop: true # Whether to loop through the files indefinitely
device_ids: [0] # The devices to run the pipeline on
batch_size: 64 # Number of (img1, img2, flow) tuples per batch
flow_fields_per_batch: 32 # Number of flow fields used for each batch
batches_per_flow_batch: 4 # Number of batches of (imgs1, imgs2, flows) tuples per flow batch

# Image generation parameters
image_shape: [1024, 1024] # Shape of the images to be generated
dt: 1 # Time step between images
seeding_density_range: [0.03, 0.04] # Range of density of particles in the images
p_hide_img1: 0.0001 # Probability of hiding particles in the first image
p_hide_img2: 0.0001 # Probability of hiding particles in the second image
diameter_ranges: [[1.5, 2.5]]  # Diameter of particles in pixels
diameter_var: 0.03 # Variance of the diameter of particles
intensity_ranges: [[150, 230]] # Range of intensity values for the particles
intensity_var: 0.01 # Variance of the intensity of particles
rho_ranges: [[-0.1, 0.1]] # Correlation coefficients for the particles (similarity)
rho_var: 0.1 # Variance of the correlation coefficients for the particles
noise_level: 0.001 # Maximum amplitude of the uniform noise to add.

# Flow generation parameters
velocities_per_pixel: 1.0 # Number of velocities per pixel
resolution: 1 # Pixels per unit length
# In this dataset, the velocities are already in pixels (they are displacements)
# so we set the resolution and the dt to 1, they shouldn't be modified.
# Same goes for image_shape and flow_field_size

# -- Flow parameters --
# Size of the flow field in normalized units. It can be meters or others,
# but need to be consistent in the other dependent settings.
flow_field_size: [1024, 1024]
img_offset: [0, 0] # Offset of the area captured in the images in normalized units
# Speeds bounds in the flow field, used to have particles covering the images
min_speed_x: 0 # Minimum speed in the x direction
max_speed_x: 0 # Maximum speed in the x direction
min_speed_y: 0 # Minimum speed in the y direction
max_speed_y: 0 # Maximum speed in the y direction
# Units of the output flow field. Can be "measure units per second" or "pixels"
output_units: "pixels"
# Flows files
scheduler_files: /shared/fluids/fluids-estimation/eye_candies_main
scheduler_class: ".mat"
```

Let's understand what some of them mean with some examples, we refer to a detailed description of all of them in ... #TODO.

### Dataset parameters

Let's say that you need some data to train your custom neural network. You decide that you would like to have batches of 64 image pairs. To speed up the generation you also decide that you're fine with some of the image pairs to share the same ground truth, in particular you settle for 16 flows per batch. You're also fine with multiple batches sharing the same flows, so each flow batch will generate 8 batches of image pairs (we refer to ablation number X on how these parameters affect performance).

You then set these parameters:

```yaml
batch_size: 64
flow_fields_per_batch: 16
batches_per_flow_batch: 8
```

To check what you're generating you can run:

```python
sampler = synthpix.make(config_path)

for i, batch in enumerate(sampler):
    imgs1 = batch["images1"]
    imgs2 = batch["images2"]
    flows = batch["flow_fields"]
    print("{imgs1.shape=}") # shape (64, H, W)
    print("{imgs2.shape=}") # shape (64, H, W)
    print("{flows.shape=}") # shape (64, H, W, 2)
```

Crucially, while each batch of images will be different, ``flows[0] == flows[16] == flows[32] == flows[48]`` and, in general, the same batch of 16 flow fields will be repeated 4 times in the batch. Moreover, the same batch of flows will be used to generate 8 consecutive batches of 64 images before switching to the next batch.

### Image and flow generation parameters

To generate images of a physical system, the workflow should be the following:

- set ``flow_field_size`` to the size in physical units of the area observed by your flow data.
- set ``image_shape`` to the resolution of the camera you want to model.
- set ``img_offset`` to the top-left coordinates (x0, y0) of the area you want to capture with your camera, in physical units.
- measure the bottom-right coordinates (x1, y1) of the area you want to capture with your camera, in physical units.
- calculate ``resolution`` by dividing the width of the image by (x1 - x0). Notice that this works when your camera has square pixels, since virtually all current cameras do.
- set ``velocities_per_pixel`` to your desired value, in particular >1 if you want to do super resolution, <1 for the opposite.
- set ``output_units`` to what you need, choosing between ``pixels`` and ``measure_units_per_second``

## Using a custom dataset with images provided (e.g., from a real setup)

To further support testing on existing datasets without significantly changing the API, ``SynthPix`` allows to open images directly from files. In particular, the most supported file format is ``.mat``. Each file should have the following keys:

- ``I0``: the first image
- ``I1``: the second image
- ``V``: the flow field.

Here the flexibility with the parameters is limited to ``batch_size``, ``loop``, ``seed``, ``randomize``, which have the same behavior explained hereabove.
