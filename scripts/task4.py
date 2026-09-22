from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from cw2.common import MESH_DIR, ensure_out_dir
from cw2.curvature import cotangent_mean_curvature
from cw2.mesh_io import load_mesh, save_mesh
from cw2.report_plots import save_scalar_heatmap, save_scalar_heatmap_views
from cw2.visualize import mesh_with_vertex_colors


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4: Cotangent Laplace-Beltrami mean curvature")
    parser.add_argument("--mesh", default="lilium_s.obj")
    args = parser.parse_args()

    mesh = load_mesh(MESH_DIR / args.mesh)
    h, _h_vec, _l, _m = cotangent_mean_curvature(mesh.vertices, mesh.faces, mesh.vertex_normals)
    save_mesh(mesh_with_vertex_colors(mesh, h), ensure_out_dir("task4", f"{args.mesh[:-4]}_H_cotangent.ply"))
    save_scalar_heatmap(mesh, h, ensure_out_dir("task4", f"{args.mesh[:-4]}_H_cotangent_heatmap.png"), f"Cotangent mean curvature | {args.mesh}")
    save_scalar_heatmap_views(mesh, h, ensure_out_dir("task4", f"{args.mesh[:-4]}_H_cotangent_dualview.png"), f"Cotangent mean curvature | {args.mesh}")
    print("Task 4 finished.")


if __name__ == "__main__":
    main()
