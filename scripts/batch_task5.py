from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import trimesh
from batch_common import filter_meshes, make_batch_dir
from cw2.common import MESH_DIR
from cw2.curvature import cotangent_laplacian
from cw2.mesh_io import load_mesh, save_mesh
from cw2.spectral import reconstruct_from_modes, smallest_eigenpairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 5 batch spectral reconstruction")
    parser.add_argument("--k", nargs="+", type=int, default=[5, 15, 100])
    parser.add_argument("--max-vertices", type=int, default=0)
    args = parser.parse_args()

    out_dir = make_batch_dir("task5")
    meshes = filter_meshes(max_vertices=args.max_vertices)
    for rel in meshes:
        mesh = load_mesh(MESH_DIR / rel)
        stem = rel.replace("/", "_").replace(".obj", "")
        l, m = cotangent_laplacian(mesh.vertices, mesh.faces)
        for k in args.k:
            evals, modes = smallest_eigenpairs(l, m, k=k)
            v_rec = reconstruct_from_modes(mesh.vertices, modes)
            rec = trimesh.Trimesh(vertices=v_rec, faces=mesh.faces, process=False)
            save_mesh(rec, out_dir / f"{stem}_recon_k{k}.obj")
            print(f"Task5 done: {rel}, k={k}, lambdas=[{evals[0]:.3e}, {evals[-1]:.3e}]")


if __name__ == "__main__":
    main()
