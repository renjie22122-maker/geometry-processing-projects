from __future__ import annotations

import math


def first_fundamental(u: float, _v: float) -> tuple[float, float, float]:
    e = 1.0
    f = 0.0
    g = 1.0 + u * u
    return e, f, g


def second_fundamental(u: float, _v: float) -> tuple[float, float, float]:
    e = 0.0
    f = -1.0 / math.sqrt(1.0 + u * u)
    g = 0.0
    return e, f, g


def normal_curvature(u: float, v: float, du: float, dv: float) -> float:
    e1, f1, g1 = first_fundamental(u, v)
    e2, f2, g2 = second_fundamental(u, v)
    num = e2 * du * du + 2.0 * f2 * du * dv + g2 * dv * dv
    den = e1 * du * du + 2.0 * f1 * du * dv + g1 * dv * dv
    if den == 0:
        raise ZeroDivisionError("Direction vector cannot be zero")
    return num / den


def normal_curvature_at_origin(theta: float) -> float:
    # At (u,v)=(0,0): E=G=1, F=0, e=g=0, f=-1.
    du, dv = math.cos(theta), math.sin(theta)
    return normal_curvature(0.0, 0.0, du, dv)
