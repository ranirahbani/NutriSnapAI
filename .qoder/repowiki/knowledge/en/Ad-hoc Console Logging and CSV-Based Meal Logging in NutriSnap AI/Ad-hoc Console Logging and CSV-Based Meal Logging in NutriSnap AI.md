---
kind: logging_system
name: Ad-hoc Console Logging and CSV-Based Meal Logging in NutriSnap AI
category: logging_system
scope:
    - '**'
source_files:
    - app.py
---

## What system/approach is used

The repository does **not** use a structured logging framework (no `logging`, `loguru`, `structlog`, or similar). Instead, it relies on two ad-hoc mechanisms:

1. **Console output via `print()`** — all runtime status messages are emitted through Python's built-in `print()` with a `[NutriSnap]` prefix to distinguish application logs from third-party library output.
2. **CSV-based meal persistence** — every analyzed meal is appended to a local `meal_log.csv` file using the `csv` module; this serves as both the data store for the dashboard and the only durable record of activity.

There is no log-level strategy (DEBUG/INFO/WARNING/ERROR), no log rotation, no structured fields, no sink configuration, and no centralized logger instance.

## Key files and packages

- `app.py` — the sole source file; contains every logging-related call.
- `requirements.txt` — lists dependencies but includes no logging library.
- `meal_log.csv` — created at runtime by `log_meal()`; schema defined by the `CSV_COLUMNS` constant (`Date`, `Time`, `Food`, `Calories`, `Protein (g)`, `Carbs (g)`, `Fat (g)`, `Portion`, `Confirmed`).

## Architecture and conventions

### Console logging convention
All console logs follow a uniform `[NutriSnap] <message>` format:
- Model loading: `load_yolo()` prints success/failure (`[NutriSnap] YOLOv8n loaded successfully`, `[NutriSnap] YOLO failed to load: ...`).
- Fallback classifier: `load_hf_classifier()` prints success/failure similarly.
- Detection errors: `detect_with_yolo()` catches exceptions and prints `[NutriSnap] YOLO detection error: {e}`.
- Classification errors: `classify_with_hf()` prints `[NutriSnap] HF classification error: {e}`.
- Startup/shutdown: the `__main__` block prints `[NutriSnap] Loading models...` and `[NutriSnap] Starting Gradio app...`.

This prefix acts as an informal namespace, making it easy to filter application logs from noisy third-party libraries (YOLO, transformers, Gradio).

### Error handling vs. logging
Errors are handled by catching broad `Exception`s and printing them rather than raising or routing them to a logger. There is no distinction between warnings and errors — all failures go through `print(f"[NutriSnap] ...")`.

### Meal logging (durable log)
The `log_meal(food, calories, protein, carbs, fat, portion, confirmed=True)` function is the single write path for meal records:
- It calls `ensure_csv()` first, which creates `meal_log.csv` with headers if missing.
- Each call appends one row with timestamped `Date` and `Time` columns plus nutrition fields.
- The same CSV is read back by `read_log()` (via pandas) for the dashboard and food-log table UI.
- Column migration is handled defensively: missing columns are auto-filled with empty strings so older CSVs remain readable.

### Structured fields
Structured data exists only in the CSV layer, not in console logs. The CSV schema is fixed by the `CSV_COLUMNS` list and enforced by the writer in `log_meal()` and the reader in `read_log()`. No JSON, key-value pairs, or machine-parseable console logs exist.

## Conventions and constraints

Observed conventions (descriptive):
- Every user-visible status message is prefixed with `[NutriSnap]`.
- All model-loading and analysis paths catch `Exception` broadly and print diagnostics instead of propagating errors.
- Meal data is always persisted to disk before being displayed in the UI.
- The CSV is treated as the single source of truth for historical meals; charts and tables re-read it on each refresh.

Constraints / enforcement:
- There is no programmatic enforcement of log levels, log formatting, or structured output — these are purely conventional and could be violated without compile-time or runtime checks.
- The CSV schema is enforced only by the writer/reader pair; adding new fields would require updating both `CSV_COLUMNS` and the corresponding read/write logic.
- Because `warnings.filterwarnings("ignore")` is called at import time, Python warning streams are suppressed globally, which also affects any future structured logging that might emit warnings.

In summary, NutriSnap AI uses a minimal, single-file logging approach: human-readable console messages with a shared `[NutriSnap]` prefix for operational visibility, and a persistent CSV file as the only structured, queryable log of meal events.