from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    cmd = [sys.executable] + args
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    run(["scripts/task1.py"])
    run(["scripts/task2.py", "--mesh", "lilium_s.obj", "--vertex", "10"])

    run(["scripts/task3.py", "--mesh", "lilium_s.obj"])
    run(["scripts/task3.py", "--mesh", "plane.obj"])

    run(["scripts/task4.py", "--mesh", "lilium_s.obj"])
    run(["scripts/task4.py", "--mesh", "plane.obj"])

    run(["scripts/task5.py", "--mesh", "armadillo.obj", "--k", "5", "15", "100"])

    run(["scripts/task6.py", "--mesh", "smoothing/plane_ns.obj", "--step", "0.2", "--iters", "10"])
    run(["scripts/task6.py", "--mesh", "smoothing/fandisk_ns.obj", "--step", "0.2", "--iters", "10"])

    run(["scripts/task7.py", "--mesh", "smoothing/plane_ns.obj", "--step", "0.2", "--iters", "10"])
    run(["scripts/task7.py", "--mesh", "smoothing/fandisk_ns.obj", "--step", "0.2", "--iters", "10"])

    print("All tasks finished.")


if __name__ == "__main__":
    main()
