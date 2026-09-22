from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from cw2.adjacency import ordered_one_ring
from cw2.common import MESH_DIR, ensure_out_dir
from cw2.mesh_io import load_mesh
from cw2.report_plots import (
    save_task2_full_traversal_series,
    save_task2_onering_growth_series,
    save_task2_onering_process_series,
    save_task2_onering_visualization,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2: Ordered one-ring neighbors")
    parser.add_argument("--mesh", default="lilium_s.obj")
    parser.add_argument("--vertex", type=int, default=0)
    parser.add_argument("--no-viz", action="store_true", help="Disable Task2 visualization export")
    parser.add_argument("--simple-example", default="cube.obj", help="A simple mesh used as an additional visualization example")
    parser.add_argument("--simple-vertex", type=int, default=0)
    parser.add_argument(
        "--simple-all-vertices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, generate simple-example visualizations for all vertices (default: true)",
    )
    args = parser.parse_args()

    mesh = load_mesh(MESH_DIR / args.mesh)
    n_vertices = len(mesh.vertices)
    all_samples = []
    for vid in range(n_vertices):
        ring_i = ordered_one_ring(vid, mesh.faces)
        all_samples.append({"vertex": int(vid), "degree": int(len(ring_i)), "ring": ring_i})

    stem = args.mesh.replace("/", "_").replace(".obj", "")
    all_out = ensure_out_dir("task2", f"{stem}_all_vertices_onering.json")
    all_out.write_text(json.dumps(all_samples, indent=2), encoding="utf-8")

    ring = ordered_one_ring(args.vertex, mesh.faces)
    print(f"Task 2: ordered one-ring on {args.mesh}")
    print(f"computed all vertices: {n_vertices}")
    print(f"saved: {all_out}")
    print(f"vertex={args.vertex}, degree={len(ring)}")
    print(ring)

    if not args.no_viz:
        out = ensure_out_dir("task2", f"{stem}_v{args.vertex}_onering.png")
        save_task2_onering_visualization(
            mesh,
            center_vertex=args.vertex,
            ring=ring,
            out_path=out,
            title_prefix=f"Task2 one-ring | {args.mesh} | v={args.vertex} | deg={len(ring)}",
        )
        print(f"saved: {out}")

        series = save_task2_onering_process_series(
            mesh=mesh,
            center_vertex=args.vertex,
            ring=ring,
            out_prefix=ensure_out_dir("task2", f"{stem}_v{args.vertex}_onering_process"),
            title_prefix=f"Task2 process | {args.mesh} | v={args.vertex} | deg={len(ring)}",
        )
        for p in series:
            print(f"saved: {p}")

        growth = save_task2_onering_growth_series(
            mesh=mesh,
            center_vertex=args.vertex,
            ring=ring,
            out_prefix=ensure_out_dir("task2", f"{stem}_v{args.vertex}_onering_growth"),
            title_prefix=f"Task2 ordered growth | {args.mesh} | v={args.vertex}",
        )
        for p in growth:
            print(f"saved: {p}")

        if args.simple_example and (MESH_DIR / args.simple_example).exists():
            simple = mesh if args.simple_example == args.mesh else load_mesh(MESH_DIR / args.simple_example)
            simple_stem = args.simple_example.replace("/", "_").replace(".obj", "")
            simple_vertices = range(len(simple.vertices)) if args.simple_all_vertices else [args.simple_vertex]
            simple_traversal: list[dict[str, object]] = []
            visited_history: set[int] = set()
            for simple_vid in simple_vertices:
                simple_ring = ordered_one_ring(simple_vid, simple.faces)
                history_before = sorted(visited_history)
                simple_traversal.append(
                    {
                        "step": int(len(simple_traversal) + 1),
                        "center": int(simple_vid),
                        "degree": int(len(simple_ring)),
                        "ring": simple_ring,
                        "history_before": history_before,
                    }
                )
                visited_history.add(int(simple_vid))
                visited_history.update(int(v) for v in simple_ring)

                simple_out = ensure_out_dir("task2", f"{simple_stem}_v{simple_vid}_onering.png")
                save_task2_onering_visualization(
                    simple,
                    center_vertex=simple_vid,
                    ring=simple_ring,
                    out_path=simple_out,
                    title_prefix=f"Task2 simple example | {args.simple_example} | v={simple_vid} | deg={len(simple_ring)}",
                )
                print(f"saved: {simple_out}")

                simple_series = save_task2_onering_process_series(
                    mesh=simple,
                    center_vertex=simple_vid,
                    ring=simple_ring,
                    out_prefix=ensure_out_dir("task2", f"{simple_stem}_v{simple_vid}_onering_process"),
                    title_prefix=f"Task2 process | {args.simple_example} | v={simple_vid} | deg={len(simple_ring)}",
                )
                for p in simple_series:
                    print(f"saved: {p}")

                simple_growth = save_task2_onering_growth_series(
                    mesh=simple,
                    center_vertex=simple_vid,
                    ring=simple_ring,
                    out_prefix=ensure_out_dir("task2", f"{simple_stem}_v{simple_vid}_onering_growth"),
                    title_prefix=f"Task2 ordered growth | {args.simple_example} | v={simple_vid}",
                )
                for p in simple_growth:
                    print(f"saved: {p}")

            if args.simple_all_vertices and simple_traversal:
                traversal_json = ensure_out_dir("task2", f"{simple_stem}_full_traversal_history.json")
                traversal_json.write_text(json.dumps(simple_traversal, indent=2), encoding="utf-8")
                print(f"saved: {traversal_json}")

                traversal_series = save_task2_full_traversal_series(
                    mesh=simple,
                    traversal=simple_traversal,
                    out_prefix=ensure_out_dir("task2", f"{simple_stem}_full_traversal"),
                    title_prefix=f"Task2 full traversal | {args.simple_example}",
                )
                for p in traversal_series:
                    print(f"saved: {p}")


if __name__ == "__main__":
    main()
