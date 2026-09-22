from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import trimesh
from batch_common import filter_meshes, make_batch_dir
from cw2.adjacency import vertex_neighbors_from_faces
from cw2.common import MESH_DIR
from cw2.mesh_io import load_mesh, save_mesh
from cw2.report_plots import save_before_after_views
from cw2.smoothing import explicit_laplacian_smoothing


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 6 batch explicit smoothing")
    parser.add_argument("--steps", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.4])
    parser.add_argument("--iters", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--max-vertices", type=int, default=0)
    args = parser.parse_args()

    out_dir = make_batch_dir("task6")
    meshes = filter_meshes(max_vertices=args.max_vertices)

    for rel in meshes:
        mesh = load_mesh(MESH_DIR / rel)
        nbrs = vertex_neighbors_from_faces(len(mesh.vertices), mesh.faces)
        stem = rel.replace("/", "_").replace(".obj", "")
        for step in args.steps:
            for it in args.iters:
                v_sm = explicit_laplacian_smoothing(mesh.vertices, nbrs, step_size=step, iterations=it)
                sm = trimesh.Trimesh(vertices=v_sm, faces=mesh.faces, process=False)
                tag = f"{stem}_explicit_s{step}_i{it}"
                save_mesh(sm, out_dir / f"{tag}.obj")
                save_before_after_views(mesh, sm, out_dir / f"{tag}_compare.png", f"Task6 {rel} s={step} i={it}")
                print(f"Task6 done: {rel}, step={step}, it={it}")


if __name__ == "__main__":
    main()
