# 🍔 PlateGenie AI

**Snap a photo of your meal and instantly unlock calories, macros, and personalized nutrition insights—your AI-powered guide to smarter, healthier eating.**

PlateGenie AI is a single-file food tracking application that uses computer vision to automatically detect, classify, and estimate nutrition from meal photos. It combines YOLOv8 object detection with a multi-model HuggingFace ensemble classifier, YOLO11n-based item counting (with OpenCV fallback), and a multi-source nutrition lookup chain (USDA API → Open Food Facts → local database) to deliver accurate per-item calorie and macronutrient estimates.

---

## Overview

PlateGenie AI turns a meal photo into structured nutrition data in seconds:

1. **Upload** a photo of your meal
2. **Detect** food items using YOLOv8 object detection
3. **Classify** each item using a 3-model ensemble with majority voting
4. **Count** items using YOLO11n object counting (OpenCV watershed fallback)
5. **Look up** nutrition data from USDA, Open Food Facts, or the built-in database
6. **Edit** food names, quantities, and portions before saving
7. **Save** to CSV and track trends on the interactive dashboard

---

## Features

### Detection & Classification
- **Multi-model ensemble food classification** — 3 parallel HuggingFace models with majority voting and confidence-weighted fusion
- **Confidence-weighted fusion** with improved scoring (70% max + 30% weighted avg, gentle majority boost capped at 1.4×) and top candidate display showing vote counts per model
- **Upgraded YOLOv8 detection** — configurable model selection (yolov8n/s/m) with automatic fallback to yolov8n on load failure
- **Advanced item counting** — YOLO11n primary counting model with OpenCV watershed fallback; 3-signal pipeline (watershed, morphological, Canny contours) as secondary fallback
- **AI-powered item counting (YOLO11n)** — model counts detected objects in each food crop, overriding heuristic estimates
- **Quantity estimation** — data-driven `COUNT_RULES` tables with aspect-ratio correction for wide bounding boxes (side-by-side items)
- **100+ per-unit weights** in `UNIT_WEIGHTS` covering proteins, breads, pizza, Mexican, Asian, breakfast, snacks, fruits, desserts, sandwiches, and burgers
- **20 food family hierarchies** in `FOOD_HIERARCHY` for deduplication (e.g., chicken → wings/drumstick/breast/thigh/nuggets/tenders)
- **34-entry `HF_TO_DB_MAP`** translating HuggingFace classifier labels to local nutrition database keys
- **Fuzzy matching** via `difflib.get_close_matches` with stricter substring rules + fallback for robust food name resolution

### User-Editable Quantities & Food Names
- **User-correctable food names with auto-recalculation** — fix misidentified foods and recompute all nutrition values instantly
- **Gradio UI**: interactive dataframe appears after analysis — edit Food, Quantity, and Grams columns; click Recalculate to look up the corrected food name in the database and recompute all nutrition values, then Save
- **HTML fallback UI**: food name is an editable text input alongside inline number inputs for quantity/grams; client-side Recalculate button looks up the new food and recomputes all values
- Edited values (including corrected food names) override AI estimates in CSV log and dashboard calculations

### User Interface
- **6-tab Gradio interface**: Upload & Analyze, Dashboard, Food Log, Nutrition Tips, Settings, AI Assistant
- **Fallback HTML/JS interface** (`fallback_ui.html`) that auto-activates if Gradio is unavailable
- **`--fallback` CLI flag** to force the HTML interface
- **Dark mode** with live toggle (persisted to config)
- **No auto-save**: analysis results require manual "Save Meal" button click
- **Delete entry** per row and **Clear All** in Food Log
- **Manual calorie entry** form when detection fails
- **HTML meal report export** for PDF printing

### Nutrition Lookup
- **USDA FoodData Central API** integration (requires free API key)
- **Open Food Facts** public API as secondary source
- **Local `NUTRITION_DB`** fallback (50+ foods with per-100g nutrition data)
- **7-day JSON cache** (`nutrition_cache.json`) with automatic expiry
- 5-step fallback chain: Cache → USDA → Open Food Facts → Local DB → None

### Dashboard
- **4 Plotly charts**: Daily Calorie Intake (bar), Macronutrient Distribution (pie), Weekly Calorie Trend (line), Top 5 Foods Eaten (horizontal bar)
- **3 summary cards**: Total Meals, Total Calories, Avg per Meal
- **HH:MM time format** (seconds stripped from timestamps)

### AI Assistant
- **AI-powered conversational assistant (Groq API, Llama 3.1)** — dual-mode chatbot integrated into the Gradio UI
- **Dual-mode chatbot: application help + personalized nutrition advisor** — keyword-based intent classification routes questions to the appropriate mode
- **Meal history analysis with daily/weekly summaries and recommendations** — reads last 7 days from `meal_log.csv` to provide data-driven insights
- **Legal disclaimer for nutrition advice liability protection** — automatically appended to all nutrition-related responses

### Settings
- USDA API key management (save, test connection)
- Cache management (view size, clear)
- CSV data export
- PDF report export (HTML-based, print to PDF)
- Dark mode toggle

---

## Screenshots

### Upload & Analyze
![Upload Tab](docs/images/upload_tab.png)

### Dashboard
![Dashboard Tab](docs/images/dashboard_tab.png)

---

## Architecture / Detection Pipeline

```
                         ┌──────────────────────┐
                         │   Input Meal Photo    │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │  Stage 1: YOLOv8 Detection   │
                     │  (COCO food classes, WHERE)   │
                     └──────────────┬──────────────┘
                                    │
               ┌────────────────────▼────────────────────┐
               │  Stage 2: Multi-Model Ensemble           │
               │  Classification (3 HF models → majority  │
               │  voting → confidence fusion, WHAT)       │
               └────────────────────┬────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  Stage 3: Sub-Region Scan      │
                    │  (missed items, skipped if ≥2  │
                    │   high-confidence detections)  │
                    └───────────────┬───────────────┘
                                    │
               ┌────────────────────▼────────────────────┐
               │  Stage 4: Advanced Counting              │
               │  (YOLO11n primary → OpenCV watershed     │
               │   fallback if model fails)               │
               └────────────────────┬────────────────────┘
                                    │
               ┌────────────────────▼────────────────────┐
               │  Stage 5: Nutrition Lookup               │
               │  (USDA API → OpenFoodFacts → local DB)   │
               └────────────────────┬────────────────────┘
                                    │
               ┌────────────────────▼────────────────────┐
               │  Stage 6: User Food Name & Quantity Editing │
               │  → Recalculate → Save to CSV             │
               └─────────────────────────────────────────┘
```

![Architecture Flow](docs/images/architecture_flow.png)

### Stage 1: YOLOv8 Detection (WHERE)
YOLOv8 runs on the full image to locate food items. Only COCO food classes (banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake) are kept. Detections below `PLATEGENIE_YOLO_CONF` are discarded.

### Stage 2: Multi-Model Ensemble Classification (WHAT)
Each YOLO bounding box is cropped and passed to 3 HuggingFace models in parallel:
- `yvelos/beit-food-384` (BEiT, weight 1.0) — primary high-accuracy classifier
- `nateraw/food` (ViT, weight 1.0) — Food-101 specialized
- `Kaludi/food-category-classification-v2.0` (ViT, weight 0.7) — broader food categories

Predictions are fused via **majority voting** (≥2 models agree) with an improved **confidence-weighted scoring** algorithm:
- **Score composition**: 70% max confidence + 30% weighted average across models
- **Gentle majority boost**: +20% score per additional model vote, capped at 1.4× (replaces the aggressive 1.5× multiplier)
- **Food group hierarchy merge**: prevents duplicates by consolidating related labels (e.g., "chicken" and "chicken_wings" are merged into a single entry using `FOOD_HIERARCHY`)
- **Improved fuzzy matching**: stricter substring rules via `difflib.get_close_matches` reduce false positive label collisions

Labels are resolved through `HF_TO_DB_MAP` and fuzzy matching against `NUTRITION_DB`. If the ensemble produces no valid result, the system falls back to the single primary model.

### Stage 3: Sub-Region Scan
Skipped when ≥2 YOLO detections have confidence >0.6 (indicating YOLO already found the main items). Otherwise, the image is divided into a grid (2×2, then 3×3) and uncovered regions are classified with the ensemble/single model to find missed food items (confidence ≥0.3). Duplicate food names are avoided.

If no detections exist after all 3 stages, the full image is classified as a last resort.

### Stage 4: Advanced Counting (HOW MANY)
PlateGenie uses **YOLO11n** as the primary counting model — it counts detected objects within each food crop and the result overrides the heuristic entirely. If the counting model fails, the system falls back to an improved OpenCV approach with 3 independent signals:

1. **Canny edge contours** — Gaussian blur → Canny edge detection → external contour counting (filtered by minimum area)
2. **Watershed segmentation** — Otsu threshold → morphological opening → distance transform → connected components as markers → component count
3. **Morphological separation** — adaptive threshold → morphological close + open → connected component labeling

The three signals are **fused via median** (middle value of 3), then clamped to food-specific maximums from `MAX_COUNTS`.

- **Accurate mode** (default): all 3 signals, median fusion, 50/50 blend with heuristic rules
- **Fast mode** (Canny-only): single signal, 60/40 blend favoring heuristics — use when analysis is slow

### Stage 5: Nutrition Lookup
For each detected food item, nutrition data is retrieved via a 5-step fallback:
1. **Cache** — `nutrition_cache.json` entries less than 7 days old
2. **USDA API** — FoodData Central (if API key configured)
3. **Open Food Facts** — free public API, no key required
4. **Local DB** — built-in `NUTRITION_DB` (50+ foods, per-100g values)
5. **None** — item gets 0 calories/macros if all sources fail

Successful lookups are cached for 7 days.

### Stage 6: User Quantity & Food Name Editing → Save
After analysis, an editable table displays each detected item with Food name, Quantity, Grams, Calories, Protein, Carbs, and Fat. Users can correct the food name (not just quantities), adjust gram weights, and click **Recalculate** to look up the corrected food in the database and recompute all nutrition from per-100g rates, then **Save Meal** to append to `meal_log.csv`. Edited values override AI estimates in the CSV log and dashboard.

### Deduplication
- Removes generic items (e.g., "chicken") when a specific variant (e.g., "chicken_wings") overlaps with IoU > 0.3
- Merges same-food detections from different regions, combining counts and nutrition

---

## Model Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PLATEGENIE_YOLO_MODEL` | `yolov8m.pt` | YOLO detection model file |
| `PLATEGENIE_YOLO_CONF` | `0.20` | Detection confidence threshold |
| `PLATEGENIE_ENSEMBLE_ENABLED` | `true` | Enable/disable multi-model ensemble |
| `PLATEGENIE_COUNT_MODEL` | `yolo11n.pt` | YOLO model for item counting within crops |
| `PLATEGENIE_COUNT_MODE` | `accurate` | Counting mode: `accurate` (watershed + morphological + contour) or `fast` (Canny-only) |

Set them before launching:

```bash
# macOS / Linux
export PLATEGENIE_YOLO_MODEL="yolov8m.pt"
export PLATEGENIE_YOLO_CONF="0.25"
export PLATEGENIE_ENSEMBLE_ENABLED="true"
export PLATEGENIE_COUNT_MODEL="yolo11n.pt"
export PLATEGENIE_COUNT_MODE="accurate"
./start.sh
```

```bat
REM Windows
set PLATEGENIE_YOLO_MODEL=yolov8m.pt
set PLATEGENIE_YOLO_CONF=0.25
set PLATEGENIE_ENSEMBLE_ENABLED=true
set PLATEGENIE_COUNT_MODEL=yolo11n.pt
set PLATEGENIE_COUNT_MODE=accurate
start.bat
```

### YOLO Model Comparison

| Model | Size | Speed | Accuracy | Use Case |
|---|---|---|---|---|
| `yolov8n.pt` | ~6 MB | Fastest | Lower | Low-resource fallback, quick testing |
| `yolov8s.pt` | ~22 MB | Fast | Good | Quick analysis, lower resource usage |
| **`yolov8m.pt`** | **~50 MB** | **Medium** | **Best** | **Default (recommended)** |

If the configured model fails to load (e.g., download error), the app automatically falls back to `yolov8n.pt`.

### Ensemble Models

| Model | Type | Weight | Notes |
|---|---|---|---|
| `yvelos/beit-food-384` | BEiT | 1.0 | Primary classifier, high accuracy on food images |
| `nateraw/food` | ViT | 1.0 | Trained on Food-101 dataset, specialized for food recognition |
| `Kaludi/food-category-classification-v2.0` | ViT | 0.7 | Broader food categories, lower weight in fusion |

All 3 models run in parallel on each YOLO crop. Predictions are resolved through label normalization, fuzzy matching, and majority voting with an improved fusion algorithm: score = 70% max confidence + 30% weighted average, with a gentle majority boost (+20% per additional vote, capped at 1.4×). The food group hierarchy merge prevents duplicate entries (e.g., "chicken" and "chicken_wings" are consolidated). Models that fail to load are silently skipped — the ensemble degrades gracefully to however many models are available.

---

## Counting Approach

PlateGenie uses **YOLO11n** as the primary counting model to count detected objects within each food crop. The model is configured via the `PLATEGENIE_COUNT_MODEL` environment variable (defaults to `yolo11n.pt`). The counting model result **overrides the heuristic entirely** when successful.

When the counting model fails (e.g., load error or inference failure), PlateGenie falls back to an improved OpenCV approach that combines 3 independent counting signals:

| Signal | Technique | Strengths |
|---|---|---|
| **Canny contours** | Edge detection → external contours | Fast, good for well-separated items |
| **Watershed segmentation** | Otsu → morphological opening → distance transform → connected components | Handles touching/overlapping items |
| **Morphological separation** | Adaptive threshold → close + open → connected components | Robust to varying lighting |

**Fusion**: The median of all 3 signals is taken (robust to outliers), then clamped to the food-specific maximum from `MAX_COUNTS`.

**Blending with heuristics** (fallback only):
- **Accurate mode** (default): 50% heuristic + 50% texture count — best overall accuracy
- **Fast mode**: 60% heuristic + 40% texture (Canny only) — faster but less refined

Set `PLATEGENIE_COUNT_MODE=fast` if analysis is too slow on your hardware.

---

## User-Editable Quantities & Food Names

### Gradio UI
After analysis, an interactive dataframe appears below the results with columns: **Food**, **Quantity**, **Grams**, **Calories**, **Protein (g)**, **Carbs (g)**, **Fat (g)**. The Food column is editable — users can correct misidentified foods, not just quantities:
1. Edit the Food name, Quantity, or Grams values directly in the table
2. Click **🔄 Recalculate Nutrition** — the corrected food name is looked up in the nutrition database and all nutrition values are recomputed from per-100g rates
3. Click **💾 Save Meal** to log the edited values to CSV

### HTML Fallback UI
After analysis, each detected food item shows with an editable text input for the food name alongside inline number inputs for quantity and grams:
1. Correct the food name in the text input, and adjust quantity/grams as needed
2. Click **🔄 Recalculate Nutrition** — client-side JavaScript looks up the new food in the database and recomputes calories and macros using per-100g rates
3. Click **💾 Save Meal** to log the adjusted values

Edited values (including corrected food names) override AI estimates in the CSV log and are reflected in dashboard charts and statistics.

---

## AI Assistant

PlateGenie AI includes a conversational chatbot (Tab 6) powered by the **Groq API** (free tier) using the **Llama 3.3 70B** model. It operates in two modes and automatically classifies user intent via keyword detection.

### Overview

The AI Assistant is a dual-mode chatbot:
- **App Knowledge mode** — answers questions about PlateGenie AI features, configuration, troubleshooting, and how the pipeline works
- **Nutrition Advisor mode** — analyzes your logged meal history to provide personalized dietary insights, patterns, and recommendations

### Setup

1. Get a free Groq API key at [console.groq.com](https://console.groq.com/keys)
2. Enter the key in the **Settings** tab (Gradio UI) — it is saved to `plategenie_config.json`
3. Alternatively, set the environment variable before launching:

```bash
export PLATEGENIE_GROQ_KEY="gsk_your_key_here"
python app.py
```

### Modes

The chatbot classifies each message into one of three intents:

| Mode | Trigger | Behavior |
|---|---|---|
| `app_help` | Keywords: app, feature, config, setting, install, tab, model, error, export, etc. | Answers from built-in PlateGenie AI knowledge base |
| `nutrition` | Keywords: calorie, protein, carb, meal, diet, weight, eat, daily, breakfast, etc. | Reads meal history and provides personalized nutrition insights |
| `general` | Fallback when neither mode scores high enough | Combines both capabilities, asks for clarification if needed |

### Meal Analysis

When in nutrition mode, the chatbot reads the **last 7 days** of entries from `meal_log.csv` and provides:
- Daily average calories, protein, carbs, and fat
- Most frequently logged foods
- Today's totals vs. weekly averages
- Actionable recommendations based on logged patterns (e.g., "your protein intake is low on weekdays")

The meal summary is cached for 60 seconds to avoid repeated disk reads.

### Legal Disclaimer

The following disclaimer is automatically appended to every nutrition-related response:

> ⚠️ **Disclaimer**: This AI assistant provides general nutritional information based on your logged meal data. It is NOT a substitute for professional medical or dietary advice. Always consult a qualified healthcare provider or registered dietitian for personalized nutrition guidance. PlateGenieAI and its developers assume no liability for dietary decisions made based on this chatbot's responses.

### Rate Limits

The Groq free tier allows **30 requests per minute**. If the limit is reached, the chatbot displays a rate-limit message and the user can retry after a brief wait. Conversation history is capped at the last 10 turns (20 messages) to stay within token limits.

---

## Installation & Quick Start

### Prerequisites
- **Python 3.8+** (3.10+ recommended)
- **pip** package manager
- **4 GB+ RAM** (~2 GB without ensemble, ~4–6 GB with all 3 ensemble models)
- Internet connection (for model downloads and API calls on first run)

### Quick Start (Recommended)

**macOS / Linux:**
```bash
chmod +x start.sh
./start.sh              # Gradio UI (default)
./start.sh --fallback   # Force fallback HTML interface
```

**Windows:**
```bat
start.bat               # Gradio UI (default)
start.bat --fallback    # Force fallback HTML interface
```

Both scripts automatically create a virtual environment, install all dependencies, and launch the app.

### Manual Setup

```bash
# Clone or download the project
cd PlateGenieAI

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

### With Custom Configuration

```bash
# Use higher-accuracy YOLO model + ensemble
export PLATEGENIE_YOLO_MODEL="yolov8m.pt"
export PLATEGENIE_ENSEMBLE_ENABLED="true"
export PLATEGENIE_COUNT_MODEL="yolo11n.pt"
export PLATEGENIE_COUNT_MODE="accurate"
python app.py
```

```bash
# Low-resource mode (faster, less accurate)
export PLATEGENIE_YOLO_MODEL="yolov8n.pt"
export PLATEGENIE_ENSEMBLE_ENABLED="false"
export PLATEGENIE_COUNT_MODE="fast"
python app.py
```

The first run downloads YOLOv8 and HuggingFace model weights automatically (~200 MB total with all 3 ensemble models).

---

## File Structure

```
PlateGenieAI/
├── app.py                  # Main application — all logic in one file
├── chatbot.py              # AI assistant module (Groq integration)
├── fallback_ui.html        # Standalone HTML/JS fallback interface
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
├── start.sh                # macOS/Linux launch script
├── start.bat               # Windows launch script
├── docs/
│   └── images/
│       ├── architecture_flow.png   # Architecture diagram
│       ├── upload_tab.png          # Screenshot: Upload & Analyze tab
│       └── dashboard_tab.png       # Screenshot: Dashboard tab
│
├── [auto-generated at runtime]
│   ├── meal_log.csv            # Meal history (Date, Time, Food, Calories, Protein, Carbs, Fat, Portion, Confirmed)
│   ├── nutrition_cache.json    # 7-day nutrition lookup cache
│   ├── plategenie_config.json       # App settings (USDA API key, dark mode)
│   └── meal_report.html        # HTML meal report (generated via Export PDF)
└── venv/                       # Virtual environment (created by start scripts)
```

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
| `POST /api/analyze` | `multipart/form-data` (image field) | Run detection pipeline on uploaded image. Returns items with nutrition, `per_100g` rates, `candidates`, `quantity` |
| `POST /api/log/manual` | `{"food", "calories", "protein", "carbs", "fat"}` | Log a manual food entry |
| `POST /api/settings` | `{"usda_api_key", "dark_mode"}` | Save settings |
| `POST /api/settings/test` | `{"api_key"}` | Test USDA API connection |
| `POST /api/cache/clear` | — | Clear nutrition cache |

### Analyze Response Format

```json
{
  "items": [
    {
      "food": "Chicken Wings",
      "portion": "6 items (refined)",
      "grams": 360,
      "calories": 594,
      "protein": 111.6,
      "carbs": 0.0,
      "fat": 13.0,
      "confidence": 0.92,
      "quantity": 6,
      "per_100g": { "calories": 165, "protein": 31.0, "carbs": 0.0, "fat": 3.6 },
      "candidates": [
        { "label": "chicken_wings", "confidence": 0.92, "votes": 3 },
        { "label": "chicken", "confidence": 0.78, "votes": 2 }
      ]
    }
  ],
  "totals": { "calories": 594, "protein": 111.6, "carbs": 0.0, "fat": 13.0 },
  "messages": ["✅ Stage 1: Found 1 food item(s)", "🔍 Stage 2: Refining with AI classifier..."]
}
```

---

## Usage

### Tab 1: Upload & Analyze
Upload a meal photo (JPG, PNG, WEBP). Click **Analyze Food** to run the multi-stage detection pipeline. Review detected items, nutrition table, annotated image, and candidate predictions. Correct food names, edit quantities in the interactive table if needed, click **Recalculate Nutrition**, then **Save Meal** to log results to CSV. Use the **Manual Entry** form if detection fails.

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

### Tab 6: AI Assistant
Conversational chatbot with two modes — ask about app features/configuration/troubleshooting, or get personalized nutrition insights from your meal history. Powered by Groq API (free tier). The chatbot classifies your intent automatically: questions about the app are answered from the built-in knowledge base, while nutrition-related questions trigger analysis of your last 7 days of logged meals.

### Save / Edit / Delete Workflow
1. Analyze a photo → review results
2. Correct food names, edit quantities in the interactive table → Recalculate (optional)
3. Click **Save Meal** → each detected food item is appended to `meal_log.csv`
4. View saved meals in the **Food Log** tab
5. Delete individual rows by index, or **Clear All** to reset

---

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PLATEGENIE_YOLO_MODEL` | `yolov8m.pt` | YOLO model file (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`) |
| `PLATEGENIE_YOLO_CONF` | `0.20` | YOLO detection confidence threshold (0.0–1.0) |
| `PLATEGENIE_ENSEMBLE_ENABLED` | `true` | Enable multi-model ensemble (`true`/`false`) |
| `PLATEGENIE_COUNT_MODEL` | `yolo11n.pt` | YOLO model for item counting within crops |
| `PLATEGENIE_COUNT_MODE` | `accurate` | Counting mode: `accurate` (3-signal median) or `fast` (Canny-only) |
| `PLATEGENIE_GROQ_KEY` | *(none)* | Groq API key for AI Assistant chatbot |

### Configuration File: `plategenie_config.json`

Auto-generated on first settings save. Structure:

```json
{
  "usda_api_key": "YOUR_KEY_HERE",
  "dark_mode": false
}
```

### USDA API Key
Get a free key at [fdc.nal.usda.gov/api-key-signup](https://fdc.nal.usda.gov/api-key-signup.html).

Enter the key in the **Settings** tab of either the Gradio UI or the fallback HTML interface. Keys are stored in `plategenie_config.json`.

---

## Dependencies

All dependencies are listed in `requirements.txt`:

| Package | Purpose |
|---|---|
| `gradio` | Web UI framework (main interface) |
| `torch` | PyTorch — deep learning inference (YOLO, HuggingFace models) |
| `torchvision` | Image transforms for model preprocessing |
| `transformers` | HuggingFace `AutoImageProcessor` + `AutoModelForImageClassification` for ensemble classifiers |
| `pillow` | Image loading and manipulation (PIL) |
| `pandas` | CSV read/write, dataframe operations for meal log and dashboard |
| `matplotlib` | Chart rendering backend (Agg backend for headless use) |
| `plotly` | Interactive dashboard charts (bar, pie, line, horizontal bar) |
| `ultralytics` | YOLOv8 object detection model |
| `opencv-python-headless` | Image processing: watershed, morphological ops, contour analysis, bounding box drawing |
| `requests>=2.28.0` | HTTP client for USDA API and Open Food Facts API calls |
| `numpy` | Numerical array operations for image processing |
| `groq>=0.4.0` | Groq API SDK for AI chatbot |

---

## Troubleshooting

### macOS

1. **"Python 3 is not installed"** — Install via Homebrew: `brew install python3`
2. **`chmod: operation not permitted`** — Run `chmod +x start.sh` first
3. **`pip install torch` fails on M1/M2/M3 Macs** — Try: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
4. **Port 7860 already in use** — Kill the existing process: `lsof -ti:7860 | xargs kill`

### Windows

5. **`start.bat` exits immediately** — Right-click → Run as Administrator, or run from an open Command Prompt
6. **`python` not recognized** — Install Python from [python.org](https://www.python.org/downloads/) and check "Add to PATH" during installation
7. **`venv\Scripts\activate.bat` not found** — Delete the `venv` folder and re-run `start.bat`
8. **`torch` CUDA errors** — Force CPU-only: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

### Model Downloads

9. **YOLO model download hangs** — Check internet connection and firewall; models are downloaded on first run (~22 MB for yolov8s, ~50 MB for yolov8m)
10. **HuggingFace model download fails** — Ensemble models total ~150 MB; ensure sufficient disk space and no firewall blocking `huggingface.co`
11. **Slow first analysis** — First run downloads all model weights; subsequent runs load from local cache

### Performance & Resources

12. **High RAM usage** — ~4–6 GB with full ensemble (3 models); switch to `PLATEGENIE_ENSEMBLE_ENABLED=false` for ~2 GB usage
13. **Analysis is slow** — Set `PLATEGENIE_COUNT_MODE=fast` to skip watershed/morphological counting; or use `yolov8n.pt` for faster detection
14. **GPU availability** — runs on CPU by default; CUDA is auto-detected by PyTorch if available for faster inference

### Detection Issues

15. **"No food items detected"** — Use a clearer photo with good lighting; all food items should be fully visible and not heavily occluded
16. **Nutrition values seem wrong** — Clear the cache in Settings to force fresh API lookups; verify portion sizes visually; edit quantities manually
17. **Wrong food identification** — Check candidate predictions in the results; ensemble majority voting usually corrects single-model errors

### General

18. **Gradio `share=True` link expires** — The public link expires after 72 hours; restart the app for a new link
19. **Fallback UI activation** — Use `--fallback` flag or the app auto-fallbacks if Gradio is unavailable (import error, port conflict)
20. **USDA API returns 401** — Your API key may be invalid; get a new one at [data.nal.usda.gov](https://fdc.nal.usda.gov/api-key-signup.html)
21. **Dashboard shows no data** — Save at least one meal via the "Save Meal" button first; data persists in `meal_log.csv`
22. **CSV export is empty** — Log at least one meal before exporting

### AI Assistant

23. **Chatbot not responding** — Check that your Groq API key is entered in the Settings tab; verify the key is valid at [console.groq.com](https://console.groq.com/keys)
24. **Rate limit error** — The Groq free tier allows 30 requests/minute; wait a moment and retry
25. **`groq` not installed** — Run `pip install groq>=0.4.0` and restart the application

---

## Architecture (Code Organization)

`app.py` is organized into clearly delimited sections:

| Section | Description |
|---|---|
| **Configuration & Constants** | File paths, theme colors, YOLO/ensemble/count config, global state |
| **Nutrition Database** | `NUTRITION_DB` — 50+ foods with per-100g calories/protein/carbs/fat/typical_g |
| **Unit Weights** | `UNIT_WEIGHTS` — 100+ entries mapping food names to grams-per-unit |
| **Food Hierarchy** | `FOOD_HIERARCHY` — 20 food families for deduplication |
| **Count Rules** | `COUNT_RULES` + `PORTION_RULES` — area-ratio-based quantity estimation |
| **COCO & HF Mapping** | `COCO_FOOD_CLASSES`, `HF_TO_DB_MAP` (34 entries), `fuzzy_match_food()` |
| **Nutrition API Functions** | `search_usda_food()`, `search_openfoodfacts()` |
| **Cache System** | Load/save/query `nutrition_cache.json` with 7-day expiry |
| **Settings Functions** | `load_config()`, `save_config()`, `test_usda_connection()`, `export_csv_file()` |
| **AI Detection** | `load_yolo()` (with yolov8n fallback), `load_hf_classifier()`, `EnsembleClassifier` class |
| **CSV Logging** | `ensure_csv()`, `log_meal()`, `read_log()`, `delete_log_entry()`, `clear_all_log()` |
| **Analysis Pipeline** | `analyze_image()` — 6-stage pipeline with ensemble, counting, nutrition, deduplication |
| **Dashboard** | `build_dashboard()` — 4 Plotly charts + 3 HTML summary cards |
| **Gradio UI** | CSS, tips, `generate_meal_report()`, `build_ui()` — all 6 tabs and event handlers |
| **Fallback HTTP Server** | `start_fallback_server()` — lightweight REST API + HTML UI server on port 7860 |
| **Main Entry Point** | Model loading, `--fallback` flag check, Gradio launch with auto-fallback |
| **AI Assistant (`chatbot.py`)** | `classify_intent()`, `summarize_meal_log()`, `build_system_prompt()`, `chat()`, `_call_groq()` — dual-mode chatbot with Groq API |

---

## License

MIT
