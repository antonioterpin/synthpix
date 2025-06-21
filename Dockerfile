# Base image with NVIDIA driver for GPU support
FROM ghcr.io/nvidia/driver:7c5f8932-550.144.03-ubuntu24.04
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set non-interactive frontend and timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Zurich

# System packages
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg git vim curl software-properties-common grep \
    libglew-dev x11-xserver-utils xvfb wget \
    python3-pip python3-venv openjdk-8-jdk \
    unzip wget \
 && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
 && rm /tmp/miniconda.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

# Create and activate a Conda env with Python 3.11
RUN conda update -y conda \
 && conda create -y -n synthpix python=3.11 pip \
 && conda clean -afy

# Ensure the environment is on PATH
ENV CONDA_DEFAULT_ENV=synthpix
ENV PATH="$CONDA_DIR/envs/$CONDA_DEFAULT_ENV/bin:$PATH"

# Copy and install Python dependencies
COPY . /app/
WORKDIR /app
RUN mkdir -p /root/.ssh \
    && ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts
RUN --mount=type=ssh pip install -e .[cuda12,dev]

# Copy your source code
COPY . /app
RUN chown -R 1000:root /app

# Env
ENV PYTHONUNBUFFERED=1


# Use conda-run to guarantee activation in ENTRYPOINT
ENTRYPOINT ["conda", "run", "-n", "synthpix", "--no-capture-output"]
CMD ["bash"]
