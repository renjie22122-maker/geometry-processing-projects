from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def smallest_eigenpairs(l: sp.csr_matrix, m: sp.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = l.shape[0]
    k_eff = min(max(2, k + 1), n - 1)
    # Shift-invert greatly accelerates retrieval of eigenvalues near zero.
    evals, evecs = spla.eigsh(l, k=k_eff, M=m, sigma=-1e-5, which="LM")
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    return evals[1 : k + 1], evecs[:, 1 : k + 1]


def reconstruct_from_modes(vertices: np.ndarray, modes: np.ndarray) -> np.ndarray:
    coeff = modes.T @ vertices
    return modes @ coeff
