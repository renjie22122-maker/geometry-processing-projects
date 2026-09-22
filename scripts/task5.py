from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import trimesh
from cw2.common import MESH_DIR, ensure_out_dir
from cw2.curvature import cotangent_laplacian
from cw2.mesh_io import load_mesh, save_mesh
from cw2.report_plots import save_one_before_many_after
from cw2.spectral import reconstruct_from_modes, smallest_eigenpairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 5: Spectral mesh reconstruction")
    parser.add_argument("--mesh", default="armadillo.obj")
    parser.add_argument("--k", nargs="+", type=int, default=[5, 15, 100])
    args = parser.parse_args()

    mesh = load_mesh(MESH_DIR / args.mesh)
    l, m = cotangent_laplacian(mesh.vertices, mesh.faces)

    rec_meshes: list[trimesh.Trimesh] = []
    labels: list[str] = []

    for k in args.k:
        evals, modes = smallest_eigenpairs(l, m, k=k)
        v_rec = reconstruct_from_modes(mesh.vertices, modes)
        rec = trimesh.Trimesh(vertices=v_rec, faces=mesh.faces, process=False)
        save_mesh(rec, ensure_out_dir("task5", f"{args.mesh[:-4]}_recon_k{k}.obj"))
        rec_meshes.append(rec)
        labels.append(f"k={k}")
        print(f"k={k}: smallest lambda={evals[0]:.6e}, largest lambda={evals[-1]:.6e}")

    save_one_before_many_after(
        mesh_before=mesh,
        meshes_after=rec_meshes,
        labels_after=labels,
        out_path=ensure_out_dir("task5", f"{args.mesh[:-4]}_recon_k_sweep_compare.png"),
        title_prefix=f"Task5 spectral reconstruction | {args.mesh}",
    )
    print("saved k-sweep comparison figure")


if __name__ == "__main__":
    main()
