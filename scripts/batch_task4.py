from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from batch_common import filter_meshes, make_batch_dir
from cw2.common import MESH_DIR
from cw2.curvature import cotangent_mean_curvature
from cw2.mesh_io import load_mesh, save_mesh
from cw2.report_plots import save_scalar_heatmap
from cw2.visualize import mesh_with_vertex_colors


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4 batch cotangent mean curvature")
    parser.add_argument("--max-vertices", type=int, default=0)
    args = parser.parse_args()

    out_dir = make_batch_dir("task4")
    meshes = filter_meshes(max_vertices=args.max_vertices)
    for rel in meshes:
        mesh = load_mesh(MESH_DIR / rel)
        stem = rel.replace("/", "_").replace(".obj", "")
        h, _h_vec, _l, _m = cotangent_mean_curvature(mesh.vertices, mesh.faces, mesh.vertex_normals)
        save_mesh(mesh_with_vertex_colors(mesh, h), out_dir / f"{stem}_H_cotangent.ply")
        save_scalar_heatmap(mesh, h, out_dir / f"{stem}_H_cotangent_heatmap.png", f"Task4 H cotangent | {rel}")
        print(f"Task4 mesh done: {rel}")


if __name__ == "__main__":
    main()
