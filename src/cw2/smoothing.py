from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def explicit_laplacian_smoothing(
    vertices: np.ndarray,
    neighbors: list[list[int]],
    step_size: float,
    iterations: int,
) -> np.ndarray:
    v = vertices.copy()
    for _ in range(iterations):
        lap = np.zeros_like(v)
        for i, nb in enumerate(neighbors):
            if not nb:
                continue
            lap[i] = v[nb].mean(axis=0) - v[i]
        v = v + step_size * lap
    return v


def implicit_laplacian_smoothing(
    vertices: np.ndarray,
    l: sp.csr_matrix,
    step_size: float,
    iterations: int,
) -> np.ndarray:
    v = vertices.copy()
    n = vertices.shape[0]
    a = sp.identity(n, format="csr") + step_size * l
    for _ in range(iterations):
        for d in range(3):
            v[:, d] = spla.spsolve(a, v[:, d])
    return v
