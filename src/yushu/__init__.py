"""Yushu: reusable building blocks for PyTorch models."""

from importlib.metadata import PackageNotFoundError, version

from . import nn
from .nn import (
    StandardCausalLM,
    StandardMLP,
)
from .nn.functional import (
    set_seed,
)

try:
    __version__ = version("yushu")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["nn", "StandardMLP", "StandardCausalLM"]