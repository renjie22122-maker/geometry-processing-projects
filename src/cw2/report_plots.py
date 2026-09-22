from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh


def _set_axes_equal(ax: plt.Axes, vertices: np.ndarray) -> None:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_mesh_plain(ax: plt.Axes, mesh: trimesh.Trimesh, title: str, elev: float, azim: float) -> None:
    tri = ax.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        linewidth=0.08,
        antialiased=True,
        color="#D9D9D9",
        edgecolor="#666666",
        alpha=1.0,
    )
    tri.set_zsort("average")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    _set_axes_equal(ax, mesh.vertices)
    ax.set_axis_off()


def save_scalar_heatmap(
    mesh: trimesh.Trimesh,
    values: np.ndarray,
    out_path: str | Path,
    title: str,
    elev: float = 25,
    azim: float = 40,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    face_values = values[mesh.faces].mean(axis=1)
    tri = ax.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        linewidth=0.05,
        antialiased=True,
        edgecolor="none",
    )
    tri.set_array(face_values)
    tri.set_cmap("coolwarm")
    vmin = float(np.percentile(values, 2))
    vmax = float(np.percentile(values, 98))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6
    tri.set_clim(vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    _set_axes_equal(ax, mesh.vertices)
    ax.set_axis_off()
    cbar = fig.colorbar(tri, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label("curvature")

    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_scalar_heatmap_views(
    mesh: trimesh.Trimesh,
    values: np.ndarray,
    out_path: str | Path,
    title_prefix: str,
    views: tuple[tuple[float, float], tuple[float, float]] = ((25, 40), (20, 140)),
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    face_values = values[mesh.faces].mean(axis=1)
    vmin = float(np.percentile(values, 2))
    vmax = float(np.percentile(values, 98))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6

    fig = plt.figure(figsize=(13, 5))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.06], wspace=0.02)
    tris = []
    for idx, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(grid[0, idx - 1], projection="3d")
        tri = ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.vertices[:, 2],
            triangles=mesh.faces,
            linewidth=0.05,
            antialiased=True,
            edgecolor="none",
        )
        tri.set_array(face_values)
        tri.set_cmap("coolwarm")
        tri.set_clim(vmin=vmin, vmax=vmax)
        ax.set_title(f"{title_prefix} | view {idx}")
        ax.view_init(elev=elev, azim=azim)
        _set_axes_equal(ax, mesh.vertices)
        ax.set_axis_off()
        tris.append(tri)

    cax = fig.add_subplot(grid[0, 2])
    cbar = fig.colorbar(tris[-1], cax=cax)
    cbar.set_label("curvature")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_before_after_views(
    mesh_before: trimesh.Trimesh,
    mesh_after: trimesh.Trimesh,
    out_path: str | Path,
    title_prefix: str,
    views: tuple[tuple[float, float], tuple[float, float]] = ((25, 40), (20, 140)),
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(title_prefix, y=0.985)
    for r, (elev, azim) in enumerate(views):
        ax_b = fig.add_subplot(2, 2, 1 + r * 2, projection="3d")
        ax_a = fig.add_subplot(2, 2, 2 + r * 2, projection="3d")
        _plot_mesh_plain(ax_b, mesh_before, f"before | view {r + 1}", elev, azim)
        _plot_mesh_plain(ax_a, mesh_after, f"after  | view {r + 1}", elev, azim)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_one_before_many_after(
    mesh_before: trimesh.Trimesh,
    meshes_after: list[trimesh.Trimesh],
    labels_after: list[str],
    out_path: str | Path,
    title_prefix: str,
    views: tuple[tuple[float, float], tuple[float, float]] = ((25, 40), (20, 140)),
) -> None:
    if len(meshes_after) != len(labels_after):
        raise ValueError("meshes_after and labels_after must have the same length")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 1 + len(meshes_after)
    n_rows = len(views)
    fig = plt.figure(figsize=(3.4 * n, 3.9 * n_rows))
    fig.suptitle(title_prefix, y=0.995)

    for r, (elev, azim) in enumerate(views, start=1):
        ax0 = fig.add_subplot(n_rows, n, 1 + (r - 1) * n, projection="3d")
        _plot_mesh_plain(ax0, mesh_before, f"before | v{r}", elev, azim)

        for idx, (mesh_after, label) in enumerate(zip(meshes_after, labels_after), start=2):
            ax = fig.add_subplot(n_rows, n, idx + (r - 1) * n, projection="3d")
            _plot_mesh_plain(ax, mesh_after, f"{label} | v{r}", elev, azim)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_task2_onering_visualization(
    mesh: trimesh.Trimesh,
    center_vertex: int,
    ring: list[int],
    out_path: str | Path,
    title_prefix: str,
    views: tuple[tuple[float, float], tuple[float, float]] = ((25, 40), (20, 140)),
    show_center: bool = True,
    show_ring_points: bool = True,
    show_center_to_ring_edges: bool = True,
    show_ring_path: bool = True,
    show_incident_faces: bool = True,
    history_vertices: list[int] | None = None,
    show_history_points: bool = True,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    face_mask = np.any(mesh.faces == center_vertex, axis=1)
    face_sel = mesh.faces[face_mask]

    fig = plt.figure(figsize=(12, 5.4))
    fig.suptitle(title_prefix, y=0.99)
    for idx, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")

        tri_all = ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.vertices[:, 2],
            triangles=mesh.faces,
            linewidth=0.05,
            antialiased=True,
            color="#DCDCDC",
            edgecolor="#A6A6A6",
            alpha=0.35,
        )
        tri_all.set_zsort("average")

        if show_incident_faces and len(face_sel) > 0:
            tri_sel = ax.plot_trisurf(
                mesh.vertices[:, 0],
                mesh.vertices[:, 1],
                mesh.vertices[:, 2],
                triangles=face_sel,
                linewidth=0.6,
                antialiased=True,
                color="#FDBA74",
                edgecolor="#EA580C",
                alpha=0.65,
            )
            tri_sel.set_zsort("average")

        if show_center_to_ring_edges:
            for v in ring:
                p = mesh.vertices[v]
                q = mesh.vertices[center_vertex]
                ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="#16A34A", linewidth=2.0)

        if history_vertices and show_history_points:
            hp = mesh.vertices[np.array(history_vertices, dtype=int)]
            ax.scatter(hp[:, 0], hp[:, 1], hp[:, 2], color="#4B5563", s=28, depthshade=False)

        if show_center:
            c = mesh.vertices[center_vertex]
            ax.scatter(c[0], c[1], c[2], color="#DC2626", s=90, depthshade=False)

        if ring and show_ring_points:
            rp = mesh.vertices[np.array(ring, dtype=int)]
            ax.scatter(rp[:, 0], rp[:, 1], rp[:, 2], color="#2563EB", s=52, depthshade=False)

            if show_ring_path:
                for a, b in zip(ring, ring[1:]):
                    pa = mesh.vertices[a]
                    pb = mesh.vertices[b]
                    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color="#1D4ED8", linewidth=1.7)

        ax.set_title(f"view {idx}")
        ax.view_init(elev=elev, azim=azim)
        _set_axes_equal(ax, mesh.vertices)
        ax.set_axis_off()

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_task2_onering_process_series(
    mesh: trimesh.Trimesh,
    center_vertex: int,
    ring: list[int],
    out_prefix: str | Path,
    title_prefix: str,
    views: tuple[tuple[float, float], tuple[float, float]] = ((25, 40), (20, 140)),
) -> list[Path]:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    steps = [
        ("step1_center", dict(show_center=True, show_ring_points=False, show_center_to_ring_edges=False, show_ring_path=False, show_incident_faces=False)),
        ("step2_ring_points", dict(show_center=True, show_ring_points=True, show_center_to_ring_edges=False, show_ring_path=False, show_incident_faces=False)),
        ("step3_star_edges", dict(show_center=True, show_ring_points=True, show_center_to_ring_edges=True, show_ring_path=False, show_incident_faces=False)),
        ("step4_faces_and_order", dict(show_center=True, show_ring_points=True, show_center_to_ring_edges=True, show_ring_path=True, show_incident_faces=True)),
    ]

    for name, kwargs in steps:
        out_path = prefix.parent / f"{prefix.name}_{name}.png"
        save_task2_onering_visualization(
            mesh=mesh,
            center_vertex=center_vertex,
            ring=ring,
            out_path=out_path,
            title_prefix=f"{title_prefix} | {name}",
            views=views,
            **kwargs,
        )
        outputs.append(out_path)

    return outputs


def save_task2_full_traversal_series(
    mesh: trimesh.Trimesh,
    traversal: list[dict[str, object]],
    out_prefix: str | Path,
    title_prefix: str,
    views: tuple[tuple[float, float], tuple[float, float]] = ((25, 40), (20, 140)),
) -> list[Path]:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for step_idx, item in enumerate(traversal, start=1):
        center = int(item["center"])
        ring = [int(v) for v in item["ring"]]
        history_before = [int(v) for v in item.get("history_before", [])]

        out_path = prefix.parent / f"{prefix.name}_step{step_idx:02d}_v{center}.png"
        save_task2_onering_visualization(
            mesh=mesh,
            center_vertex=center,
            ring=ring,
            out_path=out_path,
            title_prefix=f"{title_prefix} | step={step_idx:02d} | v={center} | deg={len(ring)}",
            views=views,
            show_center=True,
            show_ring_points=True,
            show_center_to_ring_edges=True,
            show_ring_path=True,
            show_incident_faces=True,
            history_vertices=history_before,
            show_history_points=True,
        )
        outputs.append(out_path)

    return outputs


def save_task2_onering_growth_series(
    mesh: trimesh.Trimesh,
    center_vertex: int,
    ring: list[int],
    out_prefix: str | Path,
    title_prefix: str,
    views: tuple[tuple[float, float], tuple[float, float]] = ((25, 40), (20, 140)),
) -> list[Path]:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for k in range(1, len(ring) + 1):
        partial_ring = ring[:k]
        out_path = prefix.parent / f"{prefix.name}_k{k:02d}.png"
        save_task2_onering_visualization(
            mesh=mesh,
            center_vertex=center_vertex,
            ring=partial_ring,
            out_path=out_path,
            title_prefix=f"{title_prefix} | points=1+{k}",
            views=views,
            show_center=True,
            show_ring_points=True,
            show_center_to_ring_edges=True,
            show_ring_path=True,
            show_incident_faces=False,
        )
        outputs.append(out_path)

    return outputs
