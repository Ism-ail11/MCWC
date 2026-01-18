from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class QuantParams:
    scale: float
    zero: float
    qmin: int
    qmax: int


def symmetric_quant_params(x: np.ndarray, qbits: int) -> QuantParams:
    """Symmetric scalar quantization parameters for an array."""
    assert 2 <= qbits <= 16
    qmax = (1 << (qbits - 1)) - 1
    qmin = - (1 << (qbits - 1))
    maxabs = float(np.max(np.abs(x)))
    scale = max(maxabs / max(qmax, 1), 1e-12)
    return QuantParams(scale=scale, zero=0.0, qmin=qmin, qmax=qmax)


def quantize(x: np.ndarray, qp: QuantParams) -> np.ndarray:
    y = np.round((x - qp.zero) / qp.scale)
    y = np.clip(y, qp.qmin, qp.qmax)
    return y.astype(np.int16)


def dequantize(q: np.ndarray, qp: QuantParams) -> np.ndarray:
    return (q.astype(np.float32) * qp.scale + qp.zero).astype(np.float32)
