# SynthPix: A lightspeed PIV images generator 🌊

[![GitHub stars](https://img.shields.io/github/stars/IDSCETHZurich/synthpix?style=social)](https://github.com/IDSCETHZurich/synthpix/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![codecov](https://codecov.io/gh/IDSCETHZurich/synthpix/graph/badge.svg?token=UQ48NNZSI4)](https://codecov.io/gh/IDSCETHZurich/synthpix)
[![Tests](https://github.com/IDSCETHZurich/synthpix/actions/workflows/test.yaml/badge.svg)](https://github.com/IDSCETHZurich/synthpix/actions/workflows/test.yaml)
[![PyPI version](https://img.shields.io/pypi/v/synthpix.svg)](https://pypi.org/project/synthpix)

[![Follow @antonio_terpin](https://img.shields.io/twitter/follow/antonio_terpin.svg?style=social)](https://twitter.com/antonio_terpin)

`SynthPix` is a synthetic image generator for Particle Image Velocimetry (PIV) with a focus on performance and parallelism on accelerators, implemented in [JAX](https://docs.jax.dev/en/latest/quickstart.html). `SynthPix` supports the same configuration parameters as existing tools but achieves a throughput several orders of magnitude higher in image-pair generation per second, enabling comprehensive validation and comparison of PIV algorithms, rapid experimental design iterations, and the development of data-hungry methods.
![The SynthPix image generation pipeline](docs/synthpix.jpg)

In a nutshell, if you need many synthetic PIV images and you do not want to wait ages, you are better off with `SynthPix` 😄. Below are the performances (image pairs per second) with and without GPU for different batch sizes B.
![Performances](docs/times.jpg)

`SynthPix` is also fairly easy to use:
```python
import synthpix

sampler = synthpix.make(config_path)
for i, batch in enumerate(sampler):
   """batch contains images1, images2, flow_fields, params"""
sampler.shutdown()
```
See `src/main.py` for a fully working example, and check out the [paper]() for more information and performance analysis 🔥.

## Getting started 🚀
Alright, now that hopefully we convinced you to try SynthPix, let's get to it. Don't worry, installing it is even easier than using it:
```bash
pip install synthpix
```
If you have CUDA GPUs,
```bash
pip install "synthpix[cuda12]"
```
If you have issues with CUDA drivers, please follow the official instructions for [cuda12](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) and [cudnn](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)
(*Note: wheels only available on linux*).

Check out our [instructions](docs/installing.md) for installing `SynthPix` with Docker and from source.

To generate the images, you need flow data. We provide two scripts to download the commonly used PIV datasets:
```bash
sh scripts/download_piv_1.sh <output_folder> <configs_directory>
```
```sh
sh scripts/download_piv_2.sh <output_folder> <configs_directory>
```
These scripts will automatically save also one configuration file each in `<configs_directory>`. You can use these paths as in the example above.

For more examples and tutorials to use custom flow data or real-world data, check out our [tutorials page](docs/tutorials.md).

## Configuring the synthetic images ⚙️

Configuration is handled through a YAML file, organized into four main groups. Here’s a quick guide to what each set of parameters does:

### 1. Dataset Parameters
By these parameters one can personalize how the extraction of the flow fields works and the shape of the output batch

| **Parameter**            | **Description**                         |
| ------------------------ | --------------------------------------- |
| `seed`                   | Random seed for reproducibility         |
| `batch_size`             | Number of image pairs generated at once |
| `flow_fields_per_batch`  | Number of unique flow fields per batch  |
| `batches_per_flow_batch` | Number of batches generated before updating to a new set of flow fields |

### 2. Image Generation Parameters
Define the look and realism of your synthetic PIV images.

| **Parameter**                 | **Description**                                             |
| ----------------------------- | ----------------------------------------------------------- |
| `image_shape`                 | Output image size `[height, width]` (pixels)                |
| `dt`                          | Time between frames (seconds)                               |
| `seeding_density_range`       | Range of particle densities (particles per pixel)           |
| `diameter_ranges`             | List of possible ranges for particle sizes (in pixels).     |
| `diameter_var`                | Variability in particle size                                |
| `intensity_ranges`            | List of possible intensity (brightness) ranges.             |
| `intensity_var`               | Variability in intensity                                    |
| `p_hide_img1` / `p_hide_img2` | Per-particle probability of being hidden in image 1 (or 2). |
| `rho_ranges`                  | List of possible correlation coefficients ranges            |
| `rho_var`                     | Variability in correlation                                  |
| `noise_level`                 | Amplitude of image noise (simulates camera noise)           |

For diameter_ranges, intensity_ranges, and rho_ranges, each parameter is a list of possible ranges. For every image pair, one range is randomly selected from each list, and then the particle parameters are sampled from these selected ranges.

Each *_var parameter controls how much the property of each particle can change between the first and second image of a pair. The change is applied as Gaussian noise with zero mean and variance given by the corresponding *_var value.

Here’s what each *_var parameter controls:

diameter_var:
Simulates changes in particle size between frames, making the effect of focus, depth movement, or slight deformations more realistic.

Effect of diameter_var on Particle Size
<p align="center"> <b>Low</b> <img src="docs/images/diam2.png" width="90" style="image-rendering: pixelated; margin: 0 10px; vertical-align: middle;"/> <span style="font-size: 2em; vertical-align: middle;">&#8594;</span> <b>Medium</b> <img src="docs/images/base.png" width="90" style="image-rendering: pixelated; margin: 0 10px; vertical-align: middle;"/> <span style="font-size: 2em; vertical-align: middle;">&#8594;</span> <b>High</b> <img src="docs/images/diam3.png" width="90" style="image-rendering: pixelated; margin: 0 10px; vertical-align: middle;"/> </p> <p align="center"> <sub>Increasing the particle diameter makes particles appear larger and more diffuse.</sub> </p>

intensity_var:
Models natural brightness changes due to variations in illumination, camera response, or particles moving in and out of the light sheet mimicking out-of-plane motion.

Effect of intensity_var on Particle Brightness
<p align="center"> <b>Low</b> <img src="docs/images/50_Int.png" width="90" style="image-rendering: pixelated; margin: 0 10px; vertical-align: middle;"/> <span style="font-size: 2em; vertical-align: middle;">&#8594;</span> <b>Medium</b> <img src="docs/images/150_Int.png" width="90" style="image-rendering: pixelated; margin: 0 10px; vertical-align: middle;"/> <span style="font-size: 2em; vertical-align: middle;">&#8594;</span> <b>High</b> <img src="docs/images/base.png" width="90" style="image-rendering: pixelated; margin: 0 10px; vertical-align: middle;"/> </p> <p align="center"> <sub>Raising the intensity parameter produces brighter particles (higher signal-to-noise).</sub> </p>

rho_var:
Adds variability to the shape and orientation (elongation/rotation) of particles between frames.

Effect of rho_var on Particle Shape
<p align="center"> <b>Positive</b> <img src="docs/images/rho_positive.png" width="90" style="image-rendering: pixelated; margin: 0 10px; vertical-align: middle;"/> <span style="font-size: 2em; vertical-align: middle;">&#8594;</span> <b>Zero</b> <img src="docs/images/no_rho.png" width="90" style="image-rendering: pixelated; margin: 0 10px; vertical-align: middle;"/> <span style="font-size: 2em; vertical-align: middle;">&#8594;</span> <b>Negative</b> <img src="docs/images/base.png" width="90" style="image-rendering: pixelated; margin: 0 10px; vertical-align: middle;"/> </p> <p align="center"> <sub>Increasing <code>rho</code> module makes particle spots more elliptical and/or rotated, changing their shape from circular to stretched.</sub> </p>


### 3. Flow Generation Parameters
| **Parameter**          | **Description**                          |
| ---------------------- | ---------------------------------------- |
| `velocities_per_pixel` | Spatial resolution of the velocity field |
| `resolution`           | Pixels per unit of physical length       |


### 4. Flow Field Parameters
Describe which region of your flow field is captured, and its velocity bounds.

| **Parameter**                    | **Description**                                         |
| -------------------------------- | ------------------------------------------------------- |
| `flow_field_size`                | Physical area imaged (units must match your flow files) |
| `img_offset`                     | Offset for camera’s position within the flow field      |
| `min_speed_x/y`, `max_speed_x/y` | Range of allowed velocities in each direction           |
| `output_units`                   | Units of the output flow field ("pixels" or physical)   |
| `scheduler_files`                | List of ground-truth flow field files (e.g. `.h5`)      |
| `scheduler_class`                | Loader class for your flow files (usually by extension) |

Minimum and maximum speeds for each axis are not absolute values; they define the full range (positive and negative) of possible velocities along that axis.

## Contributing 🤗
Contributions are more than welcome! 🙏 Please check out our [how to contribute page](docs/contributing.md), and feel free to open an issue for problems and feature requests⚠️.

## Citation 📈
If you use this code in your research, please cite our paper:
```bash
   @article{terpin2025synthpix,
      title={SynthPix: A lightspeed PIV images generator},
      author={Terpin, Antonio and Bonomi, Alan and Banelli, Francesco and D'Andrea, Raffaello},
      year={2025}
   }
```
