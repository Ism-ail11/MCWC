from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class AffinePredictor:
    """Simple affine predictor: X_hat = a * X_prev + b."""

    a: float
    b: float

    def predict(self, x_prev: np.ndarray) -> np.ndarray:
        return (self.a * x_prev + self.b).astype(np.float32)


def fit_affine(x_prev: np.ndarray, x_cur: np.ndarray) -> AffinePredictor:
    """Fit scalar affine predictor minimizing ||a*x_prev+b - x_cur||_2."""
    xp = x_prev.reshape(-1).astype(np.float64)
    yc = x_cur.reshape(-1).astype(np.float64)
    # solve least squares for [a,b]
    A = np.stack([xp, np.ones_like(xp)], axis=1)
    theta, *_ = np.linalg.lstsq(A, yc, rcond=None)
    a = float(theta[0])
    b = float(theta[1])
    # clip to stable range
    a = float(np.clip(a, -2.0, 2.0))
    b = float(np.clip(b, -1e2, 1e2))
    return AffinePredictor(a=a, b=b)
