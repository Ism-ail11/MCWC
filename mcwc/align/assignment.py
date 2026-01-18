from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


AlignMethod = Literal["hungarian", "greedy", "sortproj"]


@dataclass
class AlignResult:
    perm: np.ndarray  # shape [H], int32, mapping old->new indices
    score: float


def hungarian_max(sim: np.ndarray) -> AlignResult:
    """Max-sim assignment using Hungarian algorithm.

    sim: [H, H] similarity matrix. Returns perm such that sim[i, perm[i]] is maximized.
    """
    cost = -sim
    r, c = linear_sum_assignment(cost)
    perm = np.zeros(sim.shape[0], dtype=np.int32)
    perm[r] = c.astype(np.int32)
    score = float(sim[r, c].sum())
    return AlignResult(perm=perm, score=score)


def greedy_max(sim: np.ndarray) -> AlignResult:
    """Greedy max matching for square sim matrix."""
    H = sim.shape[0]
    used_cols = np.zeros(H, dtype=bool)
    perm = -np.ones(H, dtype=np.int32)
    score = 0.0

    # process rows in descending row-max to reduce collisions
    order = np.argsort(-sim.max(axis=1))
    for i in order:
        j = int(np.argmax(sim[i]))
        if not used_cols[j]:
            perm[i] = j
            used_cols[j] = True
            score += float(sim[i, j])
        else:
            # find next best free col
            candidates = np.argsort(-sim[i])
            for jj in candidates:
                jj = int(jj)
                if not used_cols[jj]:
                    perm[i] = jj
                    used_cols[jj] = True
                    score += float(sim[i, jj])
                    break

    # fill any remaining
    free = np.where(~used_cols)[0]
    for i in range(H):
        if perm[i] < 0:
            perm[i] = int(free[0])
            free = free[1:]

    return AlignResult(perm=perm, score=score)


def sortproj_max(A: np.ndarray, B: np.ndarray, seed: int = 0) -> AlignResult:
    """Scalable "sort-by-random-projection" alignment.

    A, B: [H, d]. We project to 1D then sort. This is fast and often good enough.
    """
    rng = np.random.default_rng(seed)
    d = A.shape[1]
    v = rng.standard_normal(size=(d,), dtype=np.float32)
    a = (A @ v).astype(np.float32)
    b = (B @ v).astype(np.float32)
    ia = np.argsort(a)
    ib = np.argsort(b)

    # perm defined in original index space: for each row index in A, match to row index in B
    perm = np.zeros(A.shape[0], dtype=np.int32)
    perm[ia] = ib.astype(np.int32)

    # approximate score (not exact cosine)
    score = float(np.mean((a[ia] - b[ib]) ** 2))
    return AlignResult(perm=perm, score=-score)


def solve_alignment(sim: np.ndarray, method: AlignMethod = "hungarian") -> AlignResult:
    if method == "hungarian":
        return hungarian_max(sim)
    if method == "greedy":
        return greedy_max(sim)
    raise ValueError("sortproj needs A,B; use solve_alignment_sortproj")


def solve_alignment_sortproj(A: np.ndarray, B: np.ndarray, seed: int = 0) -> AlignResult:
    return sortproj_max(A, B, seed=seed)
