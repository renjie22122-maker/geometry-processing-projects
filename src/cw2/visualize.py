from __future__ import annotations

import numpy as np
import trimesh


def scalar_to_rgba(values: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))
    x = (values - lo) / (hi - lo + 1e-12)
    x = np.clip(x, 0.0, 1.0)
    rgba = np.zeros((values.shape[0], 4), dtype=np.uint8)
    rgba[:, 0] = (255 * x).astype(np.uint8)
    rgba[:, 1] = (255 * (1.0 - np.abs(x - 0.5) * 2.0)).astype(np.uint8)
    rgba[:, 2] = (255 * (1.0 - x)).astype(np.uint8)
    rgba[:, 3] = 255
    return rgba


def mesh_with_vertex_colors(mesh: trimesh.Trimesh, values: np.ndarray) -> trimesh.Trimesh:
    out = mesh.copy()
    out.visual.vertex_colors = scalar_to_rgba(values)
    return out
