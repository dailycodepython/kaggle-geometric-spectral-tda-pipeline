"""
Configuration and JAX Hardware Backend Initialization for Multi-Scale Spectral TDA.

Defines hyperparameters for k-NN manifold construction, Forman-Ricci curvature,
JAX-batched Persistent Laplacians, 2D density-radius filtration, Information-Theoretic
Mapper, and 1D Optimal Transport (Wasserstein signatures).
"""

import os
import random
from dataclasses import dataclass
import numpy as np
import jax


@dataclass
class TDAConfig:
    """Hyperparameters for the Multi-Scale Geometric & Spectral TDA Pipeline."""
    seed: int = 42
    target_col: str = "target"
    id_col: str = "id"
    n_folds: int = 5
    inner_folds: int = 3

    # Manifold & Graph Parameters
    k_neighbors: int = 24          # Neighborhood size for k-NN graph & Persistent Laplacians
    k_wasserstein: int = 15        # Sorted distance profile length for 1D Optimal Transport
    density_alpha: float = 0.50    # Relative density threshold for 2D filtration (Radius x Density)

    # Information-Theoretic Mapper Parameters
    mapper_bins: int = 8           # Number of overlapping intervals along entropy lens
    mapper_overlap: float = 0.35   # Interval overlap ratio for Reeb graph construction
    mapper_smoothing: float = 10.0 # Bayesian prior weight for node target rate estimation

    # JAX Compute Parameters
    batch_size: int = 4096         # Batch size for jax.vmap eigensolver


def seed_everything(seed: int = 42) -> None:
    """Fix random seeds across Python and NumPy for deterministic manifold construction."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def init_jax_backend() -> str:
    """Detect and report the active JAX execution device (CPU, GPU, or TPU)."""
    backend = jax.default_backend().upper()
    devices = [str(d) for d in jax.devices()]
    info = f"JAX Backend: {backend} | Devices: {', '.join(devices)}"
    print(info)
    return info
