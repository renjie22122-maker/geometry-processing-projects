from __future__ import annotations

from collections import defaultdict

import numpy as np
import scipy.sparse as sp

from .adjacency import vertex_neighbors_from_faces


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def barycentric_vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n = vertices.shape[0]
    areas = np.zeros(n, dtype=float)
    fa = _face_areas(vertices, faces)
    for fi, (i, j, k) in enumerate(faces):
        a = fa[fi] / 3.0
        areas[i] += a
        areas[j] += a
        areas[k] += a
    return areas


def gaussian_curvature_angle_deficit(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n = vertices.shape[0]
    angle_sum = np.zeros(n, dtype=float)
    areas = barycentric_vertex_areas(vertices, faces)

    for i, j, k in faces:
        tri = [(i, j, k), (j, k, i), (k, i, j)]
        for a, b, c in tri:
            u = vertices[b] - vertices[a]
            v = vertices[c] - vertices[a]
            nu = np.linalg.norm(u)
            nv = np.linalg.norm(v)
            if nu == 0 or nv == 0:
                continue
            cosv = float(np.dot(u, v) / (nu * nv))
            cosv = np.clip(cosv, -1.0, 1.0)
            angle_sum[a] += np.arccos(cosv)

    eps = 1e-12
    return (2.0 * np.pi - angle_sum) / np.maximum(areas, eps)


def uniform_mean_curvature(vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray) -> np.ndarray:
    nbrs = vertex_neighbors_from_faces(vertices.shape[0], faces)
    h = np.zeros(vertices.shape[0], dtype=float)
    for i, nb in enumerate(nbrs):
        if not nb:
            continue
        lap = vertices[nb].mean(axis=0) - vertices[i]
        mag = 0.5 * np.linalg.norm(lap)
        sign = np.sign(np.dot(lap, normals[i]))
        h[i] = mag * (1.0 if sign == 0 else sign)
    return h


def cotangent_laplacian(vertices: np.ndarray, faces: np.ndarray) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    n = vertices.shape[0]
    w = defaultdict(float)

    def cot(a: np.ndarray, b: np.ndarray) -> float:
        cross = np.linalg.norm(np.cross(a, b))
        if cross < 1e-15:
            return 0.0
        return float(np.dot(a, b) / cross)

    for i, j, k in faces:
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        cot_i = cot(vj - vi, vk - vi)
        cot_j = cot(vi - vj, vk - vj)
        cot_k = cot(vi - vk, vj - vk)

        w[tuple(sorted((j, k)))] += cot_i
        w[tuple(sorted((i, k)))] += cot_j
        w[tuple(sorted((i, j)))] += cot_k

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    diag = np.zeros(n, dtype=float)

    for (i, j), cij in w.items():
        wij = 0.5 * cij
        if wij == 0.0:
            continue
        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([-wij, -wij])
        diag[i] += wij
        diag[j] += wij

    rows.extend(range(n))
    cols.extend(range(n))
    vals.extend(diag.tolist())

    l = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    m_diag = barycentric_vertex_areas(vertices, faces)
    m = sp.diags(m_diag)
    return l, m


def cotangent_mean_curvature(
    vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray
) -> tuple[np.ndarray, np.ndarray, sp.csr_matrix, sp.csr_matrix]:
    l, m = cotangent_laplacian(vertices, faces)
    m_inv = sp.diags(1.0 / np.maximum(m.diagonal(), 1e-12))
    h_vec = m_inv @ (l @ vertices)
    h_mag = 0.5 * np.linalg.norm(h_vec, axis=1)
    sign = np.sign(np.sum(h_vec * normals, axis=1))
    sign[sign == 0] = 1.0
    return h_mag * sign, h_vec, l, m
