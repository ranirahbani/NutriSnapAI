---
kind: error_handling
name: Silent Failure and Fallback-Based Error Handling in a Single-File Gradio App
category: error_handling
scope:
    - '**'
source_files:
    - app.py
---

## Overview

NutriSnap AI is a single-file Python application (`app.py`) built with Gradio. It has no dedicated error-handling module, exception hierarchy, or middleware layer. Instead, it uses a **graceful-degradation strategy**: every potentially failing operation (model loading, detection, classification, CSV I/O) is wrapped in `try/except Exception` blocks that log via `print()` and return safe fallback values so the UI never crashes.

## Approach and Patterns Observed

### 1. Model-loading failures are non-fatal
- `load_yolo()` and `load_hf_classifier()` catch `Exception`, print a `[NutriSnap] ... failed to load: {e}` message, and return `False`. The global model variables remain `None`, which downstream code checks before use.
- This allows the app to start even when YOLOv8 or HuggingFace transformers cannot be imported or downloaded.

### 2. Detection/classification functions swallow exceptions and return empty results
- `detect_with_yolo()`: if `yolo_model is None`, returns `[]`; otherwise wraps inference in `try/except Exception`, prints `[NutriSnap] YOLO detection error: {e}`, and returns `[]`.
- `classify_with_hf()`: if `hf_classifier/hf_processor is None`, returns `[]`; wraps torch inference in `try/except Exception`, prints `[NutriSnap] HF classification error: {e}`, and returns `[]`.
- These empty lists propagate up as "no detections" rather than raising.

### 3. User-facing errors are returned as strings, not raised
- `analyze_image()` returns a three-tuple `(annotated_image, summary_markdown, detections)`. When no food is detected, it returns `(None, "No food items detected. Try a clearer photo of a meal.", None)` — a user-friendly string placed into the Gradio Markdown output.
- The Gradio handler `on_analyze` also returns the string `"Please upload an image first."` when the input file is `None`.

### 4. Data I/O errors are handled defensively
- `read_log()` catches `pd.errors.EmptyDataError` from pandas and returns an empty DataFrame with the expected columns, preventing dashboard crashes on a missing or empty CSV.
- `calculate_nutrition()` returns `None` for unknown foods instead of raising `KeyError`.
- Numeric conversion in `build_dashboard()` uses `errors="coerce"` and `.fillna(0)` to tolerate malformed CSV rows.

### 5. No structured exceptions, sentinel errors, or error codes exist
- There are no custom exception classes, no error-code constants, no `raise ValueError(...)` patterns, and no `logging` module usage beyond ad-hoc `print()` statements prefixed with `[NutriSnap]`.
- Errors are not propagated across function boundaries; they are converted into control-flow signals (`return []`, `return None`, `return False`).

### 6. Global state guards prevent cascading failures
- Module-level globals `yolo_model`, `hf_classifier`, `hf_processor` are initialized to `None`. Every consumer checks them before use, ensuring the app degrades gracefully when models fail to load.

### 7. Warnings are suppressed globally
- `warnings.filterwarnings("ignore")` at import time silences library warnings (e.g., from matplotlib, transformers), reducing noise but also hiding potential issues.

## Architecture Implications

- **Single responsibility boundary**: All logic lives in one file, so there is no layered error propagation chain to design around.
- **UI-first resilience**: The Gradio event handlers expect functions to always return valid outputs; raising unhandled exceptions would crash the Gradio server thread. The codebase deliberately avoids raising by converting all failures into safe defaults.
- **Observability is limited**: Errors are only visible in the console/stdout via `print()`. There is no structured logging, no error metrics, and no way to surface detailed diagnostics to the end user beyond the short markdown messages.

## Conventions and Constraints

| Pattern | Where observed | Behavior |
|---|---|---|
| `try/except Exception` per risky call | `load_yolo`, `load_hf_classifier`, `detect_with_yolo`, `classify_with_hf` | Logs via `print(f"[NutriSnap] ... error: {e}")`, returns safe default |
| Empty-list fallback | `detect_with_yolo`, `classify_with_hf` | Downstream pipeline treats failure as "no detections" |
| String status messages | `analyze_image`, `on_analyze` | Returned as Gradio Markdown content |
| `None` sentinel for unknown data | `calculate_nutrition`, model globals | Callers branch on truthiness |
| Pandas coercion + fillna | `build_dashboard` | Malformed numeric cells become 0 |
| Empty DataFrame fallback | `read_log` | Dashboard renders charts with no data instead of crashing |

There are no documented rules or lint/CI checks enforcing these conventions; they emerge from the single-file, prototype-style nature of the project.