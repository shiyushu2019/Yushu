"""Functional helpers shared across modules."""

import random

import torch

__all__ = ["set_seed"]


def set_seed(seed: int = 42, *, deterministic: bool = False) -> int:
    """Seed every random number generator used by Python and PyTorch.

    Seeds ``random``, NumPy (only when it is installed), the torch CPU
    generator and every accelerator device.

    Args:
        seed: Value applied to every random number generator.
        deterministic: When True, also disable the cuDNN heuristics so that
            repeated runs give identical results. This can slow training
            down, and some CUDA kernels additionally require
            ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` to be exported before launch.

    Returns:
        The seed that was applied, handy for logging.

    Note:
        A fixed seed does not make every operation reproducible. DataLoader
        worker processes, ``PYTHONHASHSEED`` and non-deterministic CUDA
        kernels still need to be handled separately.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    # Redundant on recent torch releases, kept so older ones are covered too.
    torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
