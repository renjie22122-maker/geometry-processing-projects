from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from batch_common import filter_meshes, make_batch_dir
from cw2.common import MESH_DIR
from cw2.curvature import gaussian_curvature_angle_deficit, uniform_mean_curvature
from cw2.mesh_io import load_mesh, save_mesh
from cw2.report_plots import save_scalar_heatmap
from cw2.visualize import mesh_with_vertex_colors


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3 batch uniform/gaussian curvature")
    parser.add_argument("--max-vertices", type=int, default=0)
    args = parser.parse_args()

    out_dir = make_batch_dir("task3")
    meshes = filter_meshes(max_vertices=args.max_vertices)
    for rel in meshes:
        mesh = load_mesh(MESH_DIR / rel)
        stem = rel.replace("/", "_").replace(".obj", "")
        h = uniform_mean_curvature(mesh.vertices, mesh.faces, mesh.vertex_normals)
        k = gaussian_curvature_angle_deficit(mesh.vertices, mesh.faces)

        save_mesh(mesh_with_vertex_colors(mesh, h), out_dir / f"{stem}_H_uniform.ply")
        save_mesh(mesh_with_vertex_colors(mesh, k), out_dir / f"{stem}_K_angle_deficit.ply")
        save_scalar_heatmap(mesh, h, out_dir / f"{stem}_H_uniform_heatmap.png", f"Task3 H uniform | {rel}")
        save_scalar_heatmap(mesh, k, out_dir / f"{stem}_K_angle_deficit_heatmap.png", f"Task3 K angle-deficit | {rel}")
        print(f"Task3 mesh done: {rel}")


if __name__ == "__main__":
    main()
