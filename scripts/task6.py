from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import trimesh
from cw2.adjacency import vertex_neighbors_from_faces
from cw2.common import MESH_DIR, ensure_out_dir
from cw2.mesh_io import load_mesh, save_mesh
from cw2.report_plots import save_before_after_views, save_one_before_many_after
from cw2.smoothing import explicit_laplacian_smoothing


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 6: Explicit Laplacian smoothing")
    parser.add_argument("--mesh", default=None, help="Optional single mesh override")
    parser.add_argument(
        "--meshes",
        nargs="+",
        default=["smoothing/plane_ns.obj", "smoothing/fandisk_ns.obj"],
        help="Meshes for control-variable plots",
    )
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--sweep-steps", nargs="+", type=float, default=[0.01, 0.03, 0.06, 0.1, 0.15, 0.25, 0.35])
    parser.add_argument("--sweep-iters", nargs="+", type=int, default=[1, 2, 5, 10, 20, 40, 80])
    parser.add_argument("--fixed-step", type=float, default=0.1)
    parser.add_argument("--fixed-iters", type=int, default=10)
    args = parser.parse_args()

    mesh_list = [args.mesh] if args.mesh else args.meshes
    for mesh_name in mesh_list:
        mesh = load_mesh(MESH_DIR / mesh_name)
        nbrs = vertex_neighbors_from_faces(len(mesh.vertices), mesh.faces)
        v_sm = explicit_laplacian_smoothing(mesh.vertices, nbrs, step_size=args.step, iterations=args.iters)

        sm = trimesh.Trimesh(vertices=v_sm, faces=mesh.faces, process=False)
        save_mesh(sm, ensure_out_dir("task6", f"{mesh_name[:-4]}_explicit_s{args.step}_i{args.iters}.obj"))
        save_before_after_views(
            mesh,
            sm,
            ensure_out_dir("task6", f"{mesh_name[:-4]}_explicit_s{args.step}_i{args.iters}_compare.png"),
            f"Task6 explicit smoothing | {mesh_name} | s={args.step} i={args.iters}",
        )

        step_meshes: list[trimesh.Trimesh] = []
        step_labels: list[str] = []
        for step in args.sweep_steps:
            v = explicit_laplacian_smoothing(mesh.vertices, nbrs, step_size=step, iterations=args.fixed_iters)
            m = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
            step_meshes.append(m)
            step_labels.append(f"s={step}")
            save_mesh(m, ensure_out_dir("task6", f"{mesh_name[:-4]}_explicit_s{step}_i{args.fixed_iters}.obj"))

        save_one_before_many_after(
            mesh_before=mesh,
            meshes_after=step_meshes,
            labels_after=step_labels,
            out_path=ensure_out_dir("task6", f"{mesh_name[:-4]}_explicit_control_step_i{args.fixed_iters}.png"),
            title_prefix=f"Task6 explicit | {mesh_name} | control-iteration (i={args.fixed_iters})",
        )

        iter_meshes: list[trimesh.Trimesh] = []
        iter_labels: list[str] = []
        for it in args.sweep_iters:
            v = explicit_laplacian_smoothing(mesh.vertices, nbrs, step_size=args.fixed_step, iterations=it)
            m = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
            iter_meshes.append(m)
            iter_labels.append(f"i={it}")
            save_mesh(m, ensure_out_dir("task6", f"{mesh_name[:-4]}_explicit_s{args.fixed_step}_i{it}.obj"))

        save_one_before_many_after(
            mesh_before=mesh,
            meshes_after=iter_meshes,
            labels_after=iter_labels,
            out_path=ensure_out_dir("task6", f"{mesh_name[:-4]}_explicit_control_iter_s{args.fixed_step}.png"),
            title_prefix=f"Task6 explicit | {mesh_name} | control-step (s={args.fixed_step})",
        )
        print(f"Task 6 mesh finished: {mesh_name}")
    print("Task 6 finished.")


if __name__ == "__main__":
    main()
