from __future__ import annotations

import numpy as np


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute cosine similarity between rows of A and rows of B.

    A: [n, d], B: [m, d] -> S: [n, m]
    """
    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)
    An = np.linalg.norm(A, axis=1, keepdims=True) + eps
    Bn = np.linalg.norm(B, axis=1, keepdims=True) + eps
    A2 = A / An
    B2 = B / Bn
    return A2 @ B2.T
