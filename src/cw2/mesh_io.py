from __future__ import annotations

from pathlib import Path

import trimesh


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(str(path), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")
    return mesh


def save_mesh(mesh: trimesh.Trimesh, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
