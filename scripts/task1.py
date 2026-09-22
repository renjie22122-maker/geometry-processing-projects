from __future__ import annotations

import math

import _bootstrap  # noqa: F401
from cw2.task1_helicoid import first_fundamental, normal_curvature_at_origin, second_fundamental


def main() -> None:
    e1, f1, g1 = first_fundamental(0.0, 0.0)
    e2, f2, g2 = second_fundamental(0.0, 0.0)
    print("Task 1: Helicoid fundamental forms at (u,v)=(0,0)")
    print(f"First form: E={e1:.6f}, F={f1:.6f}, G={g1:.6f}")
    print(f"Second form: e={e2:.6f}, f={f2:.6f}, g={g2:.6f}")
    for deg in [0, 30, 45, 60, 90]:
        theta = math.radians(deg)
        print(f"k_n(theta={deg:>3} deg) = {normal_curvature_at_origin(theta): .6f}")


if __name__ == "__main__":
    main()
