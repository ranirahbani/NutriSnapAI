---
kind: dependency_management
name: Python Dependency Management via Flat requirements.txt
category: dependency_management
scope:
    - '**'
source_files:
    - requirements.txt
    - app.py
---

## What system/approach is used

This repository uses the simplest possible Python dependency management approach: a single flat `requirements.txt` file at the repository root, with no version pinning and no lockfile. There is no `setup.py`, `pyproject.toml`, `Pipfile`, `poetry.lock`, or vendored dependencies directory. The application is a single-file Gradio app (`app.py`) that imports its third-party libraries directly.

## Key files and packages

- **`requirements.txt`** — declares all runtime dependencies as unpinned package names:
  - `gradio` (web UI framework)
  - `torch`, `torchvision` (PyTorch deep learning runtime)
  - `transformers` (HuggingFace model loading for fallback classification)
  - `pillow` (image handling)
  - `pandas` (CSV/log data manipulation)
  - `matplotlib` (with `Agg` backend set in code to avoid GUI dependency)
  - `plotly` (interactive dashboard charts)
  - `ultralytics` (YOLOv8 detection)
  - `opencv-python-headless` (OpenCV without GUI, suitable for headless environments)

- **`app.py`** — imports these packages at module level (e.g., `import cv2`, `import pandas as pd`, `import gradio as gr`, `import plotly.express as px`) and performs lazy/conditional imports for heavy ML models inside functions (`from ultralytics import YOLO`, `from transformers import AutoImageProcessor, AutoModelForImageClassification`, `import torch`).

## Architecture and conventions

- **Flat dependency list**: All dependencies are declared on separate lines in `requirements.txt` with no grouping, comments, or environment markers.
- **No version constraints**: Every package is specified by name only, meaning `pip install -r requirements.txt` will resolve to whatever latest compatible versions are available at install time. This makes reproducible builds fragile.
- **No lockfile**: There is no `requirements.lock`, `Pipfile.lock`, `poetry.lock`, or equivalent artifact committed alongside the manifest. Reproducibility depends entirely on external tooling or manual pinning.
- **No private registry or authentication**: No custom index URLs, `--index-url`, `--extra-index-url`, or `GITHUB_TOKEN` / `PIP_INDEX_URL` configuration is present. All packages are expected to be pulled from PyPI.
- **No vendoring**: There is no `vendor/`, `lib/`, or inline copy of third-party source code. Dependencies are installed into the active Python environment.
- **Optional/heavy imports guarded**: Heavy ML libraries (`ultralytics`, `transformers`, `torch`) are imported lazily inside functions rather than at module top, which avoids importing them when not needed and allows graceful degradation if they fail to load.

## Conventions and constraints

Observed patterns (descriptive):
- Dependencies are listed one per line, alphabetically unordered, with no extras or optional groups.
- The app sets `matplotlib.use("Agg")` before importing `pyplot` to ensure compatibility with headless servers.
- OpenCV is pinned to the `headless` variant (`opencv-python-headless`) to avoid requiring an X11 display.
- Model weights (e.g., `yolov8n.pt`) and HuggingFace models (`yvelos/beit-food-384`) are downloaded at runtime; they are not bundled or versioned in this repo.

Enforced rules (none found):
- There is no CI step, lint rule, or script that enforces version pinning, lockfile generation, or dependency auditing.
- No `setup.cfg`, `pyproject.toml`, or `setup.py` exists to enforce metadata or install-time constraints.
- No pre-commit hook or Makefile target was found that would validate or update dependencies.

In summary, dependency management in this repository is minimal and informal: a bare `requirements.txt` with unpinned names and no lockfile, vendoring, or private registry configuration. This is sufficient for a small single-file demo but provides no reproducibility guarantees.