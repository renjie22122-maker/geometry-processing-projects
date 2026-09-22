from __future__ import annotations

import csv
import math

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from cw2.task1_helicoid import normal_curvature_at_origin

from batch_common import make_batch_dir


def main() -> None:
    out_dir = make_batch_dir("task1")
    csv_path = out_dir / "task1_normal_curvature_vs_theta.csv"
    fig_path = out_dir / "task1_normal_curvature_vs_theta.png"

    rows: list[tuple[int, float]] = []
    for deg in range(0, 181, 5):
        rows.append((deg, normal_curvature_at_origin(math.radians(deg))))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["theta_deg", "k_n"])
        w.writerows(rows)

    x = [r[0] for r in rows]
    y = [r[1] for r in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, color="#0A6EBD", linewidth=2)
    plt.xlabel("theta (deg)")
    plt.ylabel("k_n")
    plt.title("Task 1: Normal curvature at origin vs direction")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()
    print(f"Task 1 batch done: {csv_path}")


if __name__ == "__main__":
    main()
