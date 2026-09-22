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
    run(["scripts/batch_task1.py"])
    run(["scripts/batch_task2.py"])
    run(["scripts/batch_task3.py"])
    run(["scripts/batch_task4.py"])
    run(["scripts/batch_task5.py", "--k", "5", "15", "100"])
    run(["scripts/batch_task6.py"])
    run(["scripts/batch_task7.py"])
    print("All batch experiments completed.")


if __name__ == "__main__":
    main()
