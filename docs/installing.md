# Additional installation options ✚

## Try SynthPix with Docker 🐳
We provide a [https://docker.com](Docker) image to run synthpix. If you need to integrate it in your project, consider using this as starting point or integrate it with your Docker image.

To build the docker image:
```bash
docker compose build
```

To run all tests:
```bash
docker compose run --rm synthpix
```

To run a custom command, e.g. `python main.py` or a specific test:
```bash
docker compose run --rm --entrypoint "" synthpix <your-command>
```

Note (to remove once goggles is distributed as a package):
For development, while installing repos:
```bash
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519
export DOCKER_BUILDKIT=1 && docker build -t synthpix . --ssh default
```

## Install SynthPix from source ⚙️

Clone the repository:
```bash
git clone git@github.com:antonioterpin/synthpix.git
```
Then, using [https://www.anaconda.com/docs/getting-started/miniconda/main](conda),
```bash
conda create -n synthpix python=3.12
conda activate synthpix
pip install --upgrade pip uv
```

Dev with CUDA12:
```bash
uv pip install .[cuda12,dev]
```

Dev without CUDA12:
```bash
uv pip install .[dev]
```

You can of course also use the `-e` option.
