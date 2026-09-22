from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from cw2.adjacency import vertex_neighbors_from_faces
from cw2.common import MESH_DIR, get_output_root
from cw2.curvature import cotangent_laplacian
from cw2.mesh_io import load_mesh, save_mesh
from cw2.report_plots import save_before_after_views
from cw2.smoothing import explicit_laplacian_smoothing, implicit_laplacian_smoothing


def bbox_diag(vertices: np.ndarray) -> float:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    return float(np.linalg.norm(maxs - mins))


def main() -> None:
    parser = argparse.ArgumentParser(description="Show explicit blow-up vs implicit stability")
    parser.add_argument("--mesh", default="smoothing/plane_ns.obj")
    parser.add_argument("--step", type=float, default=1.5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    out_dir = get_output_root() / "stability_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.mesh.replace("/", "_").replace(".obj", "")

    mesh = load_mesh(MESH_DIR / args.mesh)
    v0 = mesh.vertices.copy()
    d0 = bbox_diag(v0)

    nbrs = vertex_neighbors_from_faces(len(mesh.vertices), mesh.faces)
    v_exp = explicit_laplacian_smoothing(v0, nbrs, step_size=args.step, iterations=args.iters)

    l, _m = cotangent_laplacian(v0, mesh.faces)
    v_imp = implicit_laplacian_smoothing(v0, l=l, step_size=args.step, iterations=args.iters)

    d_exp = bbox_diag(v_exp)
    d_imp = bbox_diag(v_imp)
    ratio_exp = d_exp / (d0 + 1e-12)
    ratio_imp = d_imp / (d0 + 1e-12)

    exp_mesh = trimesh.Trimesh(vertices=v_exp, faces=mesh.faces, process=False)
    imp_mesh = trimesh.Trimesh(vertices=v_imp, faces=mesh.faces, process=False)

    save_mesh(mesh, out_dir / f"{stem}_original.obj")
    save_mesh(exp_mesh, out_dir / f"{stem}_explicit_s{args.step}_i{args.iters}.obj")
    save_mesh(imp_mesh, out_dir / f"{stem}_implicit_s{args.step}_i{args.iters}.obj")

    save_before_after_views(
        mesh,
        exp_mesh,
        out_dir / f"{stem}_explicit_compare.png",
        f"Explicit {args.mesh} s={args.step} i={args.iters}",
    )
    save_before_after_views(
        mesh,
        imp_mesh,
        out_dir / f"{stem}_implicit_compare.png",
        f"Implicit {args.mesh} s={args.step} i={args.iters}",
    )

    fig = plt.figure(figsize=(6, 4))
    plt.bar(["original", "explicit", "implicit"], [d0, d_exp, d_imp], color=["#999999", "#D84A4A", "#2E8B57"])
    plt.ylabel("Bounding-box diagonal")
    plt.title("Stability Demo: explicit blow-up vs implicit")
    plt.tight_layout()
    fig.savefig(out_dir / f"{stem}_stability_bar.png", dpi=220)
    plt.close(fig)

    summary = {
        "mesh": args.mesh,
        "step": args.step,
        "iterations": args.iters,
        "bbox_diag_original": d0,
        "bbox_diag_explicit": d_exp,
        "bbox_diag_implicit": d_imp,
        "ratio_explicit_over_original": ratio_exp,
        "ratio_implicit_over_original": ratio_imp,
        "explicit_diverged": bool(ratio_exp > 10.0),
    }
    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
