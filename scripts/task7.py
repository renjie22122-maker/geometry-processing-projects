from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import trimesh
from cw2.common import MESH_DIR, ensure_out_dir
from cw2.curvature import cotangent_laplacian
from cw2.mesh_io import load_mesh, save_mesh
from cw2.report_plots import save_before_after_views, save_one_before_many_after
from cw2.smoothing import implicit_laplacian_smoothing


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 7: Implicit Laplacian smoothing")
    parser.add_argument("--mesh", default=None, help="Optional single mesh override")
    parser.add_argument(
        "--meshes",
        nargs="+",
        default=["smoothing/plane_ns.obj", "smoothing/fandisk_ns.obj"],
        help="Meshes for control-variable plots",
    )
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--sweep-steps", nargs="+", type=float, default=[0.01, 0.03, 0.06, 0.1, 0.2, 0.5, 1.0, 2.0])
    parser.add_argument("--sweep-iters", nargs="+", type=int, default=[1, 2, 5, 10, 20, 40, 80, 120])
    parser.add_argument("--fixed-step", type=float, default=0.1)
    parser.add_argument("--fixed-iters", type=int, default=10)
    args = parser.parse_args()

    mesh_list = [args.mesh] if args.mesh else args.meshes
    for mesh_name in mesh_list:
        mesh = load_mesh(MESH_DIR / mesh_name)
        l, _m = cotangent_laplacian(mesh.vertices, mesh.faces)
        v_sm = implicit_laplacian_smoothing(mesh.vertices, l=l, step_size=args.step, iterations=args.iters)

        sm = trimesh.Trimesh(vertices=v_sm, faces=mesh.faces, process=False)
        stem = mesh_name.replace("/", "_").replace(".obj", "")
        save_mesh(sm, ensure_out_dir("task7", f"{stem}_implicit_s{args.step}_i{args.iters}.obj"))
        save_before_after_views(
            mesh,
            sm,
            ensure_out_dir("task7", f"{stem}_implicit_s{args.step}_i{args.iters}_compare.png"),
            f"Task7 implicit smoothing | {mesh_name} | s={args.step} i={args.iters}",
        )

        step_meshes: list[trimesh.Trimesh] = []
        step_labels: list[str] = []
        for step in args.sweep_steps:
            v = implicit_laplacian_smoothing(mesh.vertices, l=l, step_size=step, iterations=args.fixed_iters)
            m_step = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
            step_meshes.append(m_step)
            step_labels.append(f"s={step}")
            save_mesh(m_step, ensure_out_dir("task7", f"{stem}_implicit_s{step}_i{args.fixed_iters}.obj"))

        save_one_before_many_after(
            mesh_before=mesh,
            meshes_after=step_meshes,
            labels_after=step_labels,
            out_path=ensure_out_dir("task7", f"{stem}_implicit_control_step_i{args.fixed_iters}.png"),
            title_prefix=f"Task7 implicit | {mesh_name} | control-iteration (i={args.fixed_iters})",
        )

        iter_meshes: list[trimesh.Trimesh] = []
        iter_labels: list[str] = []
        for it in args.sweep_iters:
            v = implicit_laplacian_smoothing(mesh.vertices, l=l, step_size=args.fixed_step, iterations=it)
            m_iter = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
            iter_meshes.append(m_iter)
            iter_labels.append(f"i={it}")
            save_mesh(m_iter, ensure_out_dir("task7", f"{stem}_implicit_s{args.fixed_step}_i{it}.obj"))

        save_one_before_many_after(
            mesh_before=mesh,
            meshes_after=iter_meshes,
            labels_after=iter_labels,
            out_path=ensure_out_dir("task7", f"{stem}_implicit_control_iter_s{args.fixed_step}.png"),
            title_prefix=f"Task7 implicit | {mesh_name} | control-step (s={args.fixed_step})",
        )
        print(f"Task 7 mesh finished: {mesh_name}")
    print("Task 7 finished.")


if __name__ == "__main__":
    main()
