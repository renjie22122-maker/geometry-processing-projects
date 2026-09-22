from __future__ import annotations

import argparse
import csv
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


def is_stable(ratio: float) -> bool:
    return 0.2 <= ratio <= 5.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable parameter demos for explicit and implicit smoothing")
    parser.add_argument("--mesh", default="smoothing/plane_ns.obj")
    parser.add_argument("--steps", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--iters", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--show-step", type=float, default=0.1)
    parser.add_argument("--show-iters", type=int, default=10)
    args = parser.parse_args()

    out_dir = get_output_root() / "stability_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.mesh.replace("/", "_").replace(".obj", "")

    mesh = load_mesh(MESH_DIR / args.mesh)
    v0 = mesh.vertices.copy()
    d0 = bbox_diag(v0)
    nbrs = vertex_neighbors_from_faces(len(mesh.vertices), mesh.faces)
    l, _m = cotangent_laplacian(v0, mesh.faces)

    rows: list[dict[str, object]] = []
    for it in args.iters:
        for step in args.steps:
            v_exp = explicit_laplacian_smoothing(v0, nbrs, step_size=step, iterations=it)
            v_imp = implicit_laplacian_smoothing(v0, l=l, step_size=step, iterations=it)
            ratio_exp = bbox_diag(v_exp) / (d0 + 1e-12)
            ratio_imp = bbox_diag(v_imp) / (d0 + 1e-12)
            rows.append(
                {
                    "mesh": args.mesh,
                    "step": step,
                    "iterations": it,
                    "explicit_ratio": ratio_exp,
                    "implicit_ratio": ratio_imp,
                    "explicit_stable": is_stable(ratio_exp),
                    "implicit_stable": is_stable(ratio_imp),
                }
            )

    json_path = out_dir / f"{stem}_stable_grid.json"
    csv_path = out_dir / f"{stem}_stable_grid.csv"
    plot_path = out_dir / f"{stem}_stable_grid.png"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mesh",
                "step",
                "iterations",
                "explicit_ratio",
                "implicit_ratio",
                "explicit_stable",
                "implicit_stable",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, method in zip(axes, ["explicit", "implicit"]):
        for it in args.iters:
            xs = []
            ys = []
            for step in args.steps:
                row = next(r for r in rows if r["iterations"] == it and r["step"] == step)
                xs.append(step)
                ys.append(float(row[f"{method}_ratio"]))
            ax.plot(xs, ys, marker="o", linewidth=2, label=f"iters={it}")
        ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1)
        ax.set_xlabel("step size")
        ax.set_title(f"{method.capitalize()} | {args.mesh}")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("bbox ratio over original")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    v_exp_show = explicit_laplacian_smoothing(v0, nbrs, step_size=args.show_step, iterations=args.show_iters)
    v_imp_show = implicit_laplacian_smoothing(v0, l=l, step_size=args.show_step, iterations=args.show_iters)
    exp_mesh = trimesh.Trimesh(vertices=v_exp_show, faces=mesh.faces, process=False)
    imp_mesh = trimesh.Trimesh(vertices=v_imp_show, faces=mesh.faces, process=False)
    save_mesh(exp_mesh, out_dir / f"{stem}_stable_explicit_s{args.show_step}_i{args.show_iters}.obj")
    save_mesh(imp_mesh, out_dir / f"{stem}_stable_implicit_s{args.show_step}_i{args.show_iters}.obj")
    save_before_after_views(
        mesh,
        exp_mesh,
        out_dir / f"{stem}_stable_explicit_compare_s{args.show_step}_i{args.show_iters}.png",
        f"Explicit stable {args.mesh} s={args.show_step} i={args.show_iters}",
    )
    save_before_after_views(
        mesh,
        imp_mesh,
        out_dir / f"{stem}_stable_implicit_compare_s{args.show_step}_i{args.show_iters}.png",
        f"Implicit stable {args.mesh} s={args.show_step} i={args.show_iters}",
    )

    stable_rows = [r for r in rows if r["explicit_stable"] and r["implicit_stable"]]
    print(json.dumps({"mesh": args.mesh, "stable_cases": stable_rows}, indent=2))


if __name__ == "__main__":
    main()
