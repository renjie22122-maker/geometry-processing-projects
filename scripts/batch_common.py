from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
from cw2.common import MESH_DIR, get_output_root


def list_clean_obj_meshes() -> list[str]:
    objs = sorted(p for p in MESH_DIR.rglob("*.obj") if "__MACOSX" not in p.parts)
    return [str(p.relative_to(MESH_DIR)).replace("\\", "/") for p in objs]


def filter_meshes(max_vertices: int | None = None) -> list[str]:
    if max_vertices is None or max_vertices <= 0:
        return list_clean_obj_meshes()
    import trimesh

    out: list[str] = []
    for rel in list_clean_obj_meshes():
        mesh = trimesh.load_mesh(str(MESH_DIR / rel), process=False)
        if mesh.vertices.shape[0] <= max_vertices:
            out.append(rel)
    return out


def make_batch_dir(task: str) -> Path:
    out = get_output_root() / "batch" / task
    out.mkdir(parents=True, exist_ok=True)
    return out
