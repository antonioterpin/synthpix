"""Package initialization for the SynthPix module."""

import os

from .make import checkpoint_args, make, save_checkpoint
from .types import SynthpixBatch
from .utils import SYNTHPIX_SCOPE

ON_UNIX = os.name == "posix"  # Check if the OS is Unix-based


__all__ = [
    "SYNTHPIX_SCOPE",
    "SynthpixBatch",
    "make",
    "save_checkpoint",
    "checkpoint_args"]
