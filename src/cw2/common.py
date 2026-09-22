from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "meshes_CW2"


def get_output_root() -> Path:
    custom = os.environ.get("CW2_OUTPUT_DIR", "").strip()
    if custom:
        path = Path(custom)
        if not path.is_absolute():
            path = ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path
    default = ROOT / "outputs"
    default.mkdir(parents=True, exist_ok=True)
    return default


def ensure_out_dir(*parts: str) -> Path:
    out = get_output_root().joinpath(*parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out
