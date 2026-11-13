"""
Metric Manifold Construction and Sparse k-NN Graph Skeleton.

Performs Rank-Gaussian normalization (QuantileTransformer) to protect Euclidean
distance metrics from heavy-tailed tabular outliers, followed by parallel cKDTree
nearest-neighbor graph extraction across combined train and test manifolds.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import QuantileTransformer


def build_knn_manifold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    k_neighbors: int = 24,
    seed: int = 42,
    n_quantiles: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Normalize tabular features into a Rank-Gaussian metric space and build a k-NN graph.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset containing raw features.
    test_df : pd.DataFrame
        Test dataset containing raw features.
    feature_cols : List[str]
        List of numerical feature column names to embed into the manifold.
    k_neighbors : int, default=24
        Number of nearest neighbors to extract (excluding self).
    seed : int, default=42
        Random seed for QuantileTransformer subsampling.
    n_quantiles : int, default=1000
        Maximum number of quantiles for rank-normal mapping.

    Returns
    -------
    X_full_norm : np.ndarray of shape (N_train + N_test, D)
        Rank-Gaussian normalized feature matrix (float32).
    dists : np.ndarray of shape (N_train + N_test, k_neighbors)
        Euclidean distances to the k nearest neighbors (float32).
    indices : np.ndarray of shape (N_train + N_test, k_neighbors)
        Indices of the k nearest neighbors in X_full_norm.
    X_train_clean : pd.DataFrame
        Median-imputed raw training features for downstream GBDT models.
    X_test_clean : pd.DataFrame
        Median-imputed raw test features for downstream GBDT models.
    """
    medians = train_df[feature_cols].median()
    X_train_clean = train_df[feature_cols].fillna(medians)
    X_test_clean = test_df[feature_cols].fillna(medians)

    qt = QuantileTransformer(
        n_quantiles=min(n_quantiles, len(train_df)),
        output_distribution="normal",
        random_state=seed,
    )

    X_train_norm = qt.fit_transform(X_train_clean)
    X_test_norm = qt.transform(X_test_clean)
    X_full_norm = np.vstack([X_train_norm, X_test_norm]).astype(np.float32)

    tree = cKDTree(X_full_norm)
    dists, indices = tree.query(X_full_norm, k=k_neighbors + 1, workers=-1)

    # Exclude self-loop at index 0
    dists = dists[:, 1:].astype(np.float32)
    indices = indices[:, 1:]

    return X_full_norm, dists, indices, X_train_clean, X_test_clean
