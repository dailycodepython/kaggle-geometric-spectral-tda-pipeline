"""
Multi-Scale Geometric & Spectral TDA Engine (JAX + LightGBM).

A modular, leak-free Topological Data Analysis feature engineering library
combining Forman-Ricci Curvature, JAX-accelerated Persistent Laplacians,
2D Density-Radius Filtration, Information-Theoretic Mapper, and 1D Optimal Transport.
"""

__version__ = "0.1.0"
__author__ = "dailycodepython"
__philosophy__ = "Math over brute compute"

# Safe progressive imports as modules are added to the repository
try:
    from .config import TDAConfig, seed_everything, init_jax_backend
except ImportError:
    pass

try:
    from .manifold import build_knn_manifold
except ImportError:
    pass

try:
    from .ricci_2d_filtration import compute_curvature_and_2d_filtration
except ImportError:
    pass

try:
    from .spectral_laplacians_jax import extract_spectral_topology_jax
except ImportError:
    pass

try:
    from .oof_mapper_wasserstein import compute_oof_tda_features
except ImportError:
    pass
