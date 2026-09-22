from __future__ import annotations

from collections import defaultdict

import numpy as np


def vertex_neighbors_from_faces(n_vertices: int, faces: np.ndarray) -> list[list[int]]:
    nbrs: list[set[int]] = [set() for _ in range(n_vertices)]
    for i, j, k in faces:
        nbrs[i].update((j, k))
        nbrs[j].update((i, k))
        nbrs[k].update((i, j))
    return [sorted(s) for s in nbrs]


def ordered_one_ring(vertex_index: int, faces: np.ndarray) -> list[int]:
    """Return one-ring neighbors in consistent topological order around a vertex."""
    chains: dict[int, list[int]] = defaultdict(list)
    incoming_count: dict[int, int] = defaultdict(int)

    for f in faces:
        ids = list(map(int, f))
        if vertex_index not in ids:
            continue
        p = ids.index(vertex_index)
        prev_v = ids[(p - 1) % 3]
        next_v = ids[(p + 1) % 3]
        chains[prev_v].append(next_v)
        incoming_count[next_v] += 1

    if not chains:
        return []

    # Pick a boundary start if available; otherwise use any key for closed rings.
    starts = [v for v in chains.keys() if incoming_count[v] == 0]
    current = starts[0] if starts else next(iter(chains.keys()))

    ring: list[int] = [current]
    seen = {current}
    while current in chains and chains[current]:
        nxt = chains[current].pop(0)
        if nxt in seen:
            break
        ring.append(nxt)
        seen.add(nxt)
        current = nxt
    return ring
