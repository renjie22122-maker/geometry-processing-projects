# Geometry Processing Projects

Selected COMP0119 implementations (2025–26): halfedge operations, surface fundamental forms, vertex adjacency, discrete curvature, cotangent operators, spectral reconstruction, and explicit/implicit smoothing.

## Run

Use Python 3.10+:

```bash
pip install -r requirements.txt
python scripts/task1.py
python scripts/task2.py --mesh /path/to/your/mesh.obj --vertex 10
python scripts/task6.py --mesh /path/to/your/mesh.obj --step 0.2 --iters 20
```

`task1.py` is analytic and needs no downloaded assets. Other scripts accept external meshes. Batch scripts and `run_all_tasks.py` retain original example filenames; change them to your own files before use. Outputs are written locally and are not included in this repository.

`halfedge/halfedge_geometry.ipynb` contains a separate halfedge notebook. Run it from that directory; install `scikit-learn` and optionally `open3d` for its additional operations and visualisation. It includes procedural examples and sections that require external meshes.

See `THIRD_PARTY_NOTICES.md` for the retained halfedge library attribution.

## Publication scope

This is a curated portfolio of coursework implementations from the author's local working files. Course questions, marking rubrics, slides, reports, student identifiers, notebook outputs, input datasets, trained weights and commercial models are not included. Original private archives remain separate.

Supply your own appropriately licensed inputs where required. Existing algorithm limitations are preserved; publication is not a claim of a new benchmark or a complete reproduction of the original assessment. Dependencies retain their own licences. No blanket licence is added to third-party material.
