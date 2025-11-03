"""Setup script for the SynthPix package."""

from setuptools import find_packages, setup

setup(
    name="synthpix",
    version="0.1.0",
    author="Antonio Terpin",
    description="Synthetic particle image generator",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "jax>=0.6.0",
        "tqdm>=4.67.1",
        "h5py>=3.13.0",
        "ruamel.yaml>=0.18.10",
        "imageio>=2.37.0",
        "matplotlib>=3.10.1",
        "opencv-python-headless>=4.5.5",
        "robo-goggles>=0.1.3",
    ],
    extras_require={
        "dev": [
            "pre_commit==4.0.1",
            "pytest==7.4.4",
            "pytest-cov",
            "snowballstemmer",
        ],
        "cuda12": [
            "jax[cuda12]>=0.6.0",
            "nvidia-cublas-cu12==12.8.4.1",
        ],
    },
    python_requires=">=3.10",
)
