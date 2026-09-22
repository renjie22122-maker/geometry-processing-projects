from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from cw2.common import MESH_DIR, ensure_out_dir
from cw2.curvature import gaussian_curvature_angle_deficit, uniform_mean_curvature
from cw2.mesh_io import load_mesh, save_mesh
from cw2.report_plots import save_scalar_heatmap, save_scalar_heatmap_views
from cw2.visualize import mesh_with_vertex_colors


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: Uniform Laplace mean/Gaussian curvature")
    parser.add_argument("--mesh", default="lilium_s.obj")
    args = parser.parse_args()

    mesh = load_mesh(MESH_DIR / args.mesh)
    h = uniform_mean_curvature(mesh.vertices, mesh.faces, mesh.vertex_normals)
    k = gaussian_curvature_angle_deficit(mesh.vertices, mesh.faces)

    save_mesh(mesh_with_vertex_colors(mesh, h), ensure_out_dir("task3", f"{args.mesh[:-4]}_H_uniform.ply"))
    save_mesh(mesh_with_vertex_colors(mesh, k), ensure_out_dir("task3", f"{args.mesh[:-4]}_K_angle_deficit.ply"))
    save_scalar_heatmap(mesh, h, ensure_out_dir("task3", f"{args.mesh[:-4]}_H_uniform_heatmap.png"), f"Uniform mean curvature | {args.mesh}")
    save_scalar_heatmap(mesh, k, ensure_out_dir("task3", f"{args.mesh[:-4]}_K_angle_deficit_heatmap.png"), f"Gaussian curvature | {args.mesh}")
    save_scalar_heatmap_views(mesh, h, ensure_out_dir("task3", f"{args.mesh[:-4]}_H_uniform_dualview.png"), f"Uniform mean curvature | {args.mesh}")
    save_scalar_heatmap_views(mesh, k, ensure_out_dir("task3", f"{args.mesh[:-4]}_K_angle_deficit_dualview.png"), f"Gaussian curvature | {args.mesh}")
    print("Task 3 finished.")


if __name__ == "__main__":
    main()
