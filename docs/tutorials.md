# More examples 🌊

## Generating image pairs from custom flow data

``SynthPix`` supports MATLAB ``.mat`` files from all versions (v4 through v7.3). For optimal loading speed, v7.3 with HDF5 structure is recommended. The files should have a flat structure with the top-level variable:

- ``'V'``: shape (H, W, 2) or (2, H, W), both are automatically recognized and supported

The locations of the files should be provided in the configuration file under ``scheduler_files``. You can either provide ``SynthPix`` with the directory containing the files, in which case all subdirectories will be opened to check for files, or with a list of file directories as such:

```yaml
scheduler_files:
- /path/to/file1.mat
- /path/to/file2.mat
```

Here's an example of what the full config should look like:

```yaml
# ----- Dataset parameters --------------------------------------------------------------
seed: 0                   # Random seed for reproducibility
batch_size: 1             # Number of (img1, img2, flow) tuples per batch
flow_fields_per_batch: 1  # Number of flow fields used for each batch
batches_per_flow_batch: 1 # Number of batches of (imgs1, imgs2, flows) tuples
                          # per flow batch
randomize: false          # Whether to randomize the order of the files
loop: true                # Whether to loop through the files indefinitely
device_ids: [0]           # List of device IDs to use for processing

# ----- Image generation parameters -----------------------------------------------------
image_shape: [1024, 1024]           # Shape of the images to be generated
dt: 1                               # Time step between images
seeding_density_range: [0.03, 0.04] # Range of density of particles in the images
p_hide_img1: 0.0001                 # Probability of hiding particles in the first image
p_hide_img2: 0.0001                 # Probability of hiding particles in the second image
diameter_ranges: [[1.5, 2.5]]       # Diameter of particles in pixels
diameter_var: 0.03                  # Variance of the diameter of the particles
intensity_ranges: [[150, 230]]      # Range of intensity values for the particles
intensity_var: 0.01                 # Variance of the intensity of the particles
rho_ranges: [[-0.1, 0.1]]           # Correlation coefficients for the particles
rho_var: 0.1                        # Variance of the correlation coefficients
                                    # for the particles
noise_level: 0.001                  # Maximum amplitude of the
                                    # uniform noise to be added.

# ----- Flow generation parameters ------------------------------------------------------
velocities_per_pixel: 1.0           # Number of velocities per pixel
resolution: 1                       # Pixels per unit length

# In this dataset, the velocities are already in pixels (they are displacements)
# and, so, we set the resolution and the dt to 1, and they shouldn't be modified.
# Same goes for image_shape and flow_field_size.
# If your dataset has different units, you can change these parameters accordingly.

# ----- Flow parameters -----------------------------------------------------------------
flow_field_size: [1024, 1024] # Size of the flow field in "measure units" (in the case of
                              # this dataset, it is pixels, in other datasets, it may be
                              # meters or normalized units).
img_offset: [0, 0]            # Offset of the area captured in the images in units.

# Speeds bounds in the flow field, used to sample particles outside the field of view
# and ensure particles covering the entirety of the images.
min_speed_x: 0                # Minimum speed in the x direction
max_speed_x: 0                # Maximum speed in the x direction
min_speed_y: 0                # Minimum speed in the y direction
max_speed_y: 0                # Maximum speed in the y direction

output_units: "pixels"        # Units of the output flow field. Can be
                              # "measure units per second" or "pixels".

# ----- Flows files ----------------------------------------------
scheduler_files: /shared/fluids/fluids-estimation/eye_candies_main
scheduler_class: ".mat"
```

Let's understand what some of them mean with some examples, we refer to a detailed description of all of them in the README.

### Dataset parameters

To start, let's configure a dataset for training a neural net on flow data. Suppose that we want:

- 64 image pairs per batch
- 16 flows per batch, i.e. 64 / 16 = 4 image pairs will share the same flow field.
- 4 batches of image pairs from each flow batch, i.e. the same batch of flows will be used to generate 4 consecutive batches before moving onto a new flow batch.

We then set these parameters:

```yaml
batch_size: 64
flow_fields_per_batch: 16
batches_per_flow_batch: 4
```

To check what we're generating you can run:

```python
sampler = synthpix.make(config_path)

for i, batch in enumerate(sampler):
    imgs1 = batch["images1"]
    imgs2 = batch["images2"]
    flows = batch["flow_fields"]
    print(f"{imgs1.shape=}") # shape (64, H, W)
    print(f"{imgs2.shape=}") # shape (64, H, W)
    print(f"{flows.shape=}") # shape (64, H, W, 2)
```

With these settings, ``SynthPix`` first samples a batch of 16 unique flow fields. Each of these fields is then used multiple times within each image batch. In this case, since batch_size=64 and flow_fields_per_batch=16, each unique flow is repeated every 16 images within a batch. Thus, the 0th, 16th, 32nd, and 48th flows are identical. This same set of flow fields is repeated across 4 consecutive batches (256 total image pairs) before SynthPix samples a new set of flow fields.

### Image and flow generation parameters

To generate images of a physical system, the workflow should be the following:

- set ``flow_field_size`` to the size in physical units of the area observed by your flow data.
- set ``image_shape`` to the resolution of the camera you want to model.
- set ``img_offset`` to the top-left coordinates ``(x0, y0)`` of the area you want to capture with your camera, in physical units.
- measure the bottom-right coordinates ``(x1, y1)`` of the area you want to capture with your camera, in physical units.
- assuming square pixels (typical of modern cameras), calculate ``resolution`` by dividing the image width in pixels by ``(x1 - x0)``
- set ``velocities_per_pixel`` to your desired value, in particular >1 if you want to do super resolution, <1 for the opposite.
- set ``output_units`` to what you need, choosing between ``pixels`` and ``measure_units_per_second``

## Using a custom dataset with images provided (e.g., from a real setup)

To further support testing on existing datasets without significantly changing the API, ``SynthPix`` allows opening images directly from files. In particular, the most supported file format is ``.mat``. Each file should have a flat structure and the following top-level variables:

- ``I0``: the first image, shape (H, W)
- ``I1``: the second image, shape (H, W)
- ``V``: the flow field, shape (H, W, 2) or (2, H, W)

Since ``SynthPix`` directly reads provided image pairs and flows, parameters related to particle simulation and flow generation are no longer applicable. The only dataset parameters still applicable are ``batch_size``, ``loop``, ``randomize``, and ``seed``.
