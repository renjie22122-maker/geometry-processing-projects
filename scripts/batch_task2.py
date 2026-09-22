from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from batch_common import filter_meshes, make_batch_dir
from cw2.adjacency import ordered_one_ring
from cw2.common import MESH_DIR
from cw2.mesh_io import load_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 batch ordered one-ring")
    parser.add_argument("--max-vertices", type=int, default=0)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Limit vertices per mesh for debugging (0 means all vertices)",
    )
    args = parser.parse_args()

    out_dir = make_batch_dir("task2")
    out_path = out_dir / "task2_ordered_onering_samples.json"
    meshes = filter_meshes(max_vertices=args.max_vertices)

    payload: dict[str, list[dict[str, object]]] = {}
    for rel in meshes:
        mesh = load_mesh(MESH_DIR / rel)
        n = len(mesh.vertices)
        if args.sample_count > 0:
            step = max(1, n // args.sample_count)
            probe = list(range(0, n, step))
            if probe[-1] != n - 1:
                probe.append(n - 1)
        else:
            probe = list(range(n))
        samples: list[dict[str, object]] = []
        for vid in probe:
            ring = ordered_one_ring(vid, mesh.faces)
            samples.append({"vertex": int(vid), "degree": int(len(ring)), "ring": ring})
        payload[rel] = samples

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Task 2 batch done: {out_path}")


if __name__ == "__main__":
    main()
