"""
clustering/agglomerative.py
───────────────────────────
Agglomerative (hierarchical) speaker clustering on cosine embeddings.

Usage
-----
    from clustering.agglomerative import cluster_speakers

    labels = cluster_speakers(embeddings, similarity_threshold=0.75)
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cosine, squareform


def _cosine_distance_matrix(embeddings: List[np.ndarray]) -> np.ndarray:
    n    = len(embeddings)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d            = float(cosine(embeddings[i], embeddings[j]))
            dist[i, j]   = d
            dist[j, i]   = d
    return dist


def cluster_speakers(
    embeddings: List[np.ndarray],
    similarity_threshold: float = 0.75,
    linkage_method: str = "average",
) -> List[int]:
    """
    Assign speaker labels to a list of embeddings.

    Parameters
    ----------
    embeddings           : list of L2-normalised embedding vectors.
    similarity_threshold : cosine similarity above which two segments are
                           considered the same speaker (0 – 1).
    linkage_method       : scipy linkage method ('average', 'ward', etc.).

    Returns
    -------
    List[int]  zero-based speaker label per segment (same order as input).
    """
    if len(embeddings) == 0:
        return []
    if len(embeddings) == 1:
        return [0]

    distance_threshold = 1.0 - similarity_threshold
    dist_matrix        = _cosine_distance_matrix(embeddings)
    condensed          = squareform(dist_matrix, checks=False)

    Z      = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    return (labels - 1).tolist()   # 0-based
