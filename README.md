# 🍔 NutriSnap AI

**Snap. Identify. Track. Eat Better.**

NutriSnap AI is a single-file food tracking application that uses computer vision to automatically detect, classify, and estimate nutrition from meal photos. It combines YOLOv8 object detection with a HuggingFace BEiT food classifier, sub-region scanning, and a multi-source nutrition lookup chain (USDA API → Open Food Facts → local database) to deliver accurate per-item calorie and macronutrient estimates.

---

## Screenshots

### Upload & Analyze
![Upload Tab](docs/images/upload_tab.png)

### Dashboard
![Dashboard Tab](docs/images/dashboard_tab.png)

### Architecture
![Architecture Flow](docs/images/architecture_flow.png)

---

## Quick Start

### macOS / Linux

```bash
chmod +x start.sh
./start.sh              # Gradio UI (default)
./start.sh --fallback   # Force fallback HTML interface
```

### Windows

```bat
start.bat               # Gradio UI (default)
start.bat --fallback    # Force fallback HTML interface
```

Both scripts automatically create a virtual environment, install all dependencies, and launch the app.

---

## Features

### Detection & Classification
- **3-stage detection pipeline**: YOLO (YOLOv8s default) → HuggingFace BEiT refinement per crop → Sub-region scan for missed items
- **Texture-based count refinement**: OpenCV contour analysis improves quantity estimation for dense foods
- **Quantity estimation**: Data-driven `COUNT_RULES` tables with aspect-ratio correction for wide bounding boxes (side-by-side items)
- **100+ per-unit weights** in `UNIT_WEIGHTS` covering proteins, breads, pizza, Mexican, Asian, breakfast, snacks, fruits, desserts, sandwiches, and burgers
- **20 food family hierarchies** in `FOOD_HIERARCHY` for deduplication (e.g., chicken → wings/drumstick/breast/thigh/nuggets/tenders)
- **`HF_TO_DB_MAP`** (34-entry mapping): translates HuggingFace classifier labels to local nutrition database keys
- **Fuzzy matching** via `difflib.get_close_matches` + substring fallback for robust food name resolution

### Nutrition Lookup
- **USDA FoodData Central API** integration (requires free API key)
- **Open Food Facts** public API as secondary source
- **Local `NUTRITION_DB`** fallback (50+ foods with per-100g nutrition data)
- **7-day JSON cache** (`nutrition_cache.json`) with automatic expiry
- 5-step fallback chain: Cache → USDA → Open Food Facts → Local DB → None

### User Interface
- **Full-screen Gradio UI** with 5 tabs (Upload, Dashboard, Food Log, Tips, Settings)
- **Fallback HTML/JS interface** (`fallback_ui.html`) that auto-activates if Gradio is unavailable
- **`--fallback` CLI flag** to force the HTML interface
- **Dark mode** with live toggle (persisted to config)
- **No auto-save**: analysis results require manual "Save Meal" button click
- **Delete entry** per row and **Clear All** in Food Log
- **Manual calorie entry** form when detection fails
- **HTML meal report export** for PDF printing

### Dashboard
- **4 Plotly charts**: Daily Calorie Intake (bar), Macronutrient Distribution (pie), Weekly Calorie Trend (line), Top 5 Foods Eaten (horizontal bar)
- **3 summary cards**: Total Meals, Total Calories, Avg per Meal
- **HH:MM time format** (seconds stripped from timestamps)

### Settings
- USDA API key management (save, test connection)
- Cache management (view size, clear)
- CSV data export
- PDF report export (HTML-based, print to PDF)
- Dark mode toggle

---

## Model Configuration

The YOLO model and confidence threshold are controlled via environment variables:

| Variable | Default | Description |
|---|---|---|
| `NUTRISNAP_YOLO_MODEL` | `yolov8s.pt` | YOLOv8 model file to use |
| `NUTRISNAP_YOLO_CONF` | `0.20` | Detection confidence threshold |

Set them before launching:

```bash
# macOS / Linux
export NUTRISNAP_YOLO_MODEL="yolov8m.pt"
export NUTRISNAP_YOLO_CONF="0.25"
./start.sh
```

```bat
REM Windows
set NUTRISNAP_YOLO_MODEL=yolov8m.pt
set NUTRISNAP_YOLO_CONF=0.25
start.bat
```

### Model Comparison

| Model | Parameters | mAP (COCO) | CPU Speed | Use Case |
|---|---|---|---|---|
| `yolov8n.pt` | 3.2M | 37.3 | ~80ms | Fastest; low-power devices, quick testing |
| **`yolov8s.pt`** | **11.2M** | **44.9** | **~130ms** | **Default; best balance of speed & accuracy** |
| `yolov8m.pt` | 25.9M | 50.2 | ~250ms | Highest accuracy; slower, needs more RAM |

If the configured model fails to load (e.g., download error), the app automatically falls back to `yolov8n.pt`.

---

## Detection Pipeline

```
Image → YOLO Detection (WHERE) → HF Classification per crop (WHAT)
      → Sub-region scan (missed items, skipped if ≥2 high-conf detections)
      → Quantity estimation (COUNT_RULES + aspect-ratio + texture refinement)
      → Nutrition lookup (cache → USDA → Open Food Facts → local DB)
      → Results display
```

### Stage 1: YOLO Detection (WHERE)
YOLOv8 runs on the full image to locate food items. Only COCO food classes (banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake) are kept. Detections below `NUTRISNAP_YOLO_CONF` are discarded.

### Stage 2: HF Classification (WHAT)
Each YOLO bounding box is cropped and passed to the HuggingFace `yvelos/beit-food-384` classifier. Top-3 predictions are evaluated; the highest-confidence label (≥0.15) that maps to a known food in `NUTRITION_DB` via `HF_TO_DB_MAP` or fuzzy matching wins. If no HF label maps, the YOLO/COCO label is kept.

### Stage 3: Sub-Region Scan
Skipped when ≥2 YOLO detections have confidence >0.6 (indicating YOLO already found the main items). Otherwise, the image is divided into a grid (2×2, then 3×3) and uncovered regions are classified with the HF model to find missed food items (confidence ≥0.3). Duplicate food names are avoided.

If no detections exist after all 3 stages, the full image is classified as a last resort.

### Stage 4: Quantity & Nutrition Estimation
- `COUNT_RULES` maps food names to area-ratio thresholds that estimate count (e.g., "6 wings" vs "2 wings")
- Aspect-ratio correction boosts the effective area ratio for wide bounding boxes (items side-by-side)
- `PORTION_RULES` handles non-countable foods (fries, rice, salad, pasta, etc.)
- `UNIT_WEIGHTS` provides per-item gram weights for 100+ food items
- Nutrition is calculated per-gram from the lookup source, then multiplied by portion weight

### Stage 5: Deduplication
- Removes generic items (e.g., "chicken") when a specific variant (e.g., "chicken_wings") overlaps with IoU > 0.3
- Merges same-food detections from different regions, combining counts and nutrition

### Stage 6: Output
- Annotated image with bounding boxes, labels, and calorie counts
- Cropped food thumbnails with 35% padding (resized to 200×200)
- Markdown summary table with per-item and total nutrition

---

## Nutrition Lookup Flow

For each detected food item, nutrition data is retrieved via a 5-step fallback:

1. **Cache** — Check `nutrition_cache.json` for data less than 7 days old
2. **USDA API** — Query USDA FoodData Central (if API key is configured in `nutri_config.json`)
3. **Open Food Facts** — Query the free, no-key Open Food Facts API
4. **Local DB** — Fall back to the built-in `NUTRITION_DB` (50+ foods, per-100g values)
5. **None** — Item gets 0 calories/macros if all sources fail

Successful lookups from USDA or OFF are cached for 7 days to avoid repeated API calls.

---

## Installation & Setup

### Prerequisites
- **Python 3.8+** (3.10+ recommended)
- **pip** package manager
- **4 GB+ RAM** (for model inference)
- Internet connection (for model downloads and API calls)

### Manual Setup

```bash
# Clone or download the project
cd NutriSnapAI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

# Launch
python app.py                  # Gradio UI
python app.py --fallback       # Fallback HTML UI
```

The first run downloads YOLOv8 and HuggingFace model weights automatically.

---

## Configuration

### USDA API Key
Get a free key at [data.nal.usda.gov/api-key-signup](https://fdc.nal.usda.gov/api-key-signup.html).

Enter the key in the **Settings** tab of either the Gradio UI or the fallback HTML interface. Keys are stored in `nutri_config.json`.

### Configuration File: `nutri_config.json`

Auto-generated on first settings save. Structure:

```json
{
  "usda_api_key": "YOUR_KEY_HERE",
  "dark_mode": false
}
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NUTRISNAP_YOLO_MODEL` | `yolov8s.pt` | YOLO model file |
| `NUTRISNAP_YOLO_CONF` | `0.20` | YOLO confidence threshold |

---

## Usage

### Tab 1: Upload & Analyze
Upload a meal photo (JPG, PNG, WEBP). Click **Analyze Food** to run the 3-stage detection pipeline. Review detected items, nutrition table, and annotated image. Click **Save Meal** to log results to CSV. Use the **Manual Entry** form if detection fails.

### Tab 2: Dashboard
View 4 Plotly charts and 3 summary cards showing your meal history trends. Click **Refresh Dashboard** to reload data.

### Tab 3: Food Log
Browse all logged meals in a sortable table. Delete individual entries by row index, or click **Clear All** to wipe the log.

### Tab 4: Nutrition Tips
Reference guide with Recommended Daily Intake (RDI) values, tracking best practices, dashboard interpretation tips, and healthy eating reminders.

### Tab 5: Settings
- **USDA API Key**: Enter, save, and test your USDA FoodData Central API key
- **Cache Management**: View cached item count and clear the cache
- **Export Data**: Download meal log as CSV or generate an HTML meal report for PDF printing
- **Appearance**: Toggle dark mode (persisted across sessions)

### Save / Delete Workflow
1. Analyze a photo → review results
2. Click **Save Meal** → each detected food item is appended to `meal_log.csv`
3. View saved meals in the **Food Log** tab
4. Delete individual rows by index, or **Clear All** to reset

---

## API Endpoints (Fallback Server)

When running with `--fallback`, the app starts a lightweight HTTP server on port `7860` with the following REST API:

### GET Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Serve `fallback_ui.html` |
| `GET /api/log` | Return all meal log entries as JSON |
| `GET /api/dashboard` | Return dashboard chart data (stats, daily, macros, weekly, top foods) |
| `GET /api/settings` | Return current settings (API key, dark mode, cache size) |
| `GET /api/export/csv` | Download `meal_log.csv` as file attachment |

### POST Endpoints

| Endpoint | Body | Description |
|---|---|---|
| `POST /api/analyze` | `multipart/form-data` (image field) | Run detection pipeline on uploaded image |
| `POST /api/log/manual` | `{"food", "calories", "protein", "carbs", "fat"}` | Log a manual food entry |
| `POST /api/settings` | `{"usda_api_key", "dark_mode"}` | Save settings |
| `POST /api/settings/test` | `{"api_key"}` | Test USDA API connection |
| `POST /api/cache/clear` | — | Clear nutrition cache |

---

## File Structure

```
NutriSnapAI/
├── app.py                  # Main application (2453 lines) — all logic in one file
├── fallback_ui.html        # Standalone HTML/JS fallback interface (885 lines)
├── requirements.txt        # Python dependencies
├── start.sh                # macOS/Linux startup script
├── start.bat               # Windows startup script
├── docs/
│   └── images/
│       ├── upload_tab.png          # Screenshot: Upload & Analyze tab
│       ├── dashboard_tab.png       # Screenshot: Dashboard tab
│       └── architecture_flow.png   # Architecture diagram
│
├── [auto-generated at runtime]
│   ├── meal_log.csv            # Meal history (Date, Time, Food, Calories, Protein, Carbs, Fat, Portion, Confirmed)
│   ├── nutrition_cache.json    # 7-day nutrition lookup cache
│   ├── nutri_config.json       # App settings (USDA API key, dark mode)
│   └── meal_report.html        # HTML meal report (generated via Export PDF)
└── venv/                       # Virtual environment (created by start scripts)
```

---

## Architecture

`app.py` is organized into clearly delimited sections:

| Section | Line Range (approx.) | Description |
|---|---|---|
| **Configuration & Constants** | 1–55 | File paths, theme colors, YOLO model config, global state |
| **Nutrition Database** | 56–125 | `NUTRITION_DB` — 50+ foods with per-100g calories/protein/carbs/fat/typical_g |
| **Unit Weights** | 127–268 | `UNIT_WEIGHTS` — 100+ entries mapping food names to grams-per-unit |
| **Food Hierarchy** | 270–291 | `FOOD_HIERARCHY` — 20 food families for deduplication |
| **Count Rules** | 293–420 | `COUNT_RULES` + `PORTION_RULES` — area-ratio-based quantity estimation |
| **COCO & HF Mapping** | 422–486 | `COCO_FOOD_CLASSES`, `HF_TO_DB_MAP` (34 entries), `fuzzy_match_food()` |
| **Nutrition API Functions** | 488–555 | `search_usda_food()`, `search_openfoodfacts()` |
| **Cache System** | 557–620 | Load/save/query `nutrition_cache.json` with 7-day expiry |
| **Settings Functions** | 622–663 | `load_config()`, `save_config()`, `test_usda_connection()`, `export_csv_file()` |
| **AI Detection** | 666–710 | `load_yolo()` (with yolov8n fallback), `load_hf_classifier()` (yvelos/beit-food-384) |
| **CSV Logging** | 712–775 | `ensure_csv()`, `log_meal()`, `read_log()`, `delete_log_entry()`, `clear_all_log()` |
| **Analysis Pipeline** | 777–1402 | `calculate_nutrition()`, `estimate_portion()`, `estimate_item_count()`, `deduplicate_results()`, `detect_with_yolo()`, `classify_with_hf()`, `draw_annotations()`, `analyze_image()` |
| **Dashboard** | 1404–1471 | `build_dashboard()` — 4 Plotly charts + 3 HTML summary cards |
| **Gradio UI** | 1474–2064 | CSS, tips markdown, `generate_meal_report()`, `build_ui()` — all 5 tabs and event handlers |
| **Fallback HTTP Server** | 2067–2427 | `start_fallback_server()` — lightweight REST API + HTML UI server on port 7860 |
| **Main Entry Point** | 2430–2453 | Model loading, `--fallback` flag check, Gradio launch with auto-fallback |

---

## Dependencies

All dependencies are listed in `requirements.txt`:

| Package | Purpose |
|---|---|
| `gradio` | Web UI framework (main interface) |
| `torch` | PyTorch — deep learning inference (YOLO, HuggingFace) |
| `torchvision` | Image transforms for model preprocessing |
| `transformers` | HuggingFace `AutoImageProcessor` + `AutoModelForImageClassification` for BEiT food classifier |
| `pillow` | Image loading and manipulation (PIL) |
| `pandas` | CSV read/write, dataframe operations for meal log and dashboard |
| `matplotlib` | Chart rendering backend (Agg backend for headless use) |
| `plotly` | Interactive dashboard charts (bar, pie, line, horizontal bar) |
| `ultralytics` | YOLOv8 object detection model |
| `opencv-python-headless` | Image processing, bounding box drawing, contour analysis |
| `requests>=2.28.0` | HTTP client for USDA API and Open Food Facts API calls |
| `numpy` | Numerical array operations for image processing |

---

## Troubleshooting

### macOS

1. **"Python 3 is not installed"** — Install via Homebrew: `brew install python3`
2. **`chmod: operation not permitted`** — Run `chmod +x start.sh` first
3. **`pip install torch` fails on M1/M2 Macs** — Try: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
4. **Port 7860 already in use** — Kill the existing process: `lsof -ti:7860 | xargs kill`

### Windows

5. **`start.bat` exits immediately** — Right-click → Run as Administrator, or run from an open Command Prompt
6. **`python` not recognized** — Install Python from [python.org](https://www.python.org/downloads/) and check "Add to PATH" during installation
7. **`venv\Scripts\activate.bat` not found** — Delete the `venv` folder and re-run `start.bat`
8. **`torch` CUDA errors** — Force CPU-only: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

### General

9. **YOLO model download hangs** — Check internet connection; models are downloaded on first run (~25 MB for yolov8s)
10. **"No food items detected"** — Use a clearer photo with good lighting; all food items should be fully visible
11. **Gradio `share=True` link expires** — The public link expires after 72 hours; restart the app for a new link
12. **USDA API returns 401** — Your API key may be invalid; get a new one at [data.nal.usda.gov](https://fdc.nal.usda.gov/api-key-signup.html)
13. **Nutrition values seem wrong** — Clear the cache in Settings to force fresh API lookups; verify portion sizes visually
14. **Dashboard shows no data** — Save at least one meal via the "Save Meal" button first; data persists in `meal_log.csv`
15. **Fallback UI "Failed to save meal"** — The fallback server's `/api/log/save` endpoint may not be implemented; use the Gradio UI for save/delete/clear operations
16. **Slow first analysis** — First run downloads YOLO and HuggingFace model weights; subsequent runs load from cache
17. **High RAM usage** — `yolov8m.pt` requires ~4 GB RAM; switch to `yolov8s.pt` (default) or `yolov8n.pt` for lower memory usage
18. **CSV export is empty** — Log at least one meal before exporting

---

## License

MIT
