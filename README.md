# 🍔 NutriSnap AI

**Snap. Identify. Track. Eat Better.**

NutriSnap AI is an AI-powered food tracking web application built with Gradio. Upload a photo of your meal, and the app automatically detects food items, estimates portion sizes, and retrieves nutritional data from multiple sources — all logged into an interactive dashboard with charts and trends.

---

## Features

- **YOLOv8 Multi-Food Detection** — Detects multiple food items in a single image using COCO-trained food classes (banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake)
- **HuggingFace Classifier Fallback** — Uses [yvelos/beit-food-384](https://huggingface.co/yvelos/beit-food-384) image classifier when YOLOv8 finds no matches
- **Portion Estimation** — Estimates Small / Medium / Large portions based on bounding box area relative to the image
- **USDA FoodData Central API** — Primary nutrition data source (requires a free API key)
- **Open Food Facts API** — Automatic fallback with no API key required
- **Local Nutrition Database** — 60+ foods as a last-resort offline fallback
- **JSON-Based Nutrition Cache** — Caches API results with a 7-day expiry to minimize repeated lookups
- **CSV Meal Logging** — Every analyzed meal is logged with date, time, food name, calories, macros, and portion
- **Interactive Plotly Dashboard** — 4 charts: daily calorie bar, macronutrient pie, weekly calorie trend line, and top foods horizontal bar
- **Dashboard Summary Cards** — Total meals, total calories, and average calories per meal
- **Manual Calorie Entry** — Log food manually when detection fails
- **Dark Mode Toggle** — Switch between light and dark themes
- **CSV / PDF Export** — Download your meal log (PDF requires `reportlab`)
- **Green Health Theme UI** — Cohesive green color palette with gradient header

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| UI Framework | Gradio |
| Object Detection | Ultralytics YOLOv8 |
| Image Classification | HuggingFace Transformers (BEiT) |
| Deep Learning | PyTorch, TorchVision |
| Charts | Plotly, Matplotlib |
| Data Processing | Pandas |
| Image Processing | OpenCV (headless), Pillow |
| API Client | Requests |
| Config | python-dotenv |

---

## Installation

```bash
# Clone or navigate to the project directory
cd NutriSnapAI

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The app will launch a local Gradio server and print a URL (typically `http://127.0.0.1:7860`). A public share link is also generated automatically via `share=True`.

> **Note:** On first run, YOLOv8n weights (~6 MB) and the HuggingFace classifier model will be downloaded automatically.

---

## Configuration

### USDA API Key (Optional but Recommended)

The app works **without** an API key by falling back to Open Food Facts and the local database. For the most accurate nutrition data:

1. Sign up for a free USDA FoodData Central API key at [fdc.nal.usda.gov/api-key-signup.html](https://fdc.nal.usda.gov/api-key-signup.html)
2. Open the **Settings** tab in the app
3. Paste your API key into the **USDA API Key** field
4. Click **Test Connection** to verify
5. Click **Save Key** to persist it to `nutri_config.json`

---

## UI Tabs

### 📸 Upload & Analyze
Upload a meal photo and click **Analyze Food**. The app runs YOLOv8 detection, falls back to the HuggingFace classifier if needed, estimates portions, retrieves nutrition data, draws annotated bounding boxes on the image, and displays a summary table with calories and macros. A manual entry form is available below for foods that aren't detected.

### 📊 Dashboard
Interactive analytics with 4 Plotly charts:
- **Daily Calorie Intake** — Bar chart of total calories per day
- **Macronutrient Distribution** — Pie chart of protein, carbs, and fat
- **Weekly Calorie Trend** — Line chart of weekly totals
- **Top Foods Eaten** — Horizontal bar chart of your most logged foods

Summary cards show total meals, total calories, and average calories per meal.

### 📋 Food Log
A tabular view of every logged meal entry with date, time, food name, calories, protein, carbs, fat, portion size, and confirmation status.

### 💡 Nutrition Tips
Static reference page with recommended daily intake values, tracking best practices, dashboard interpretation guide, and healthy eating reminders.

### ⚙️ Settings
- **USDA API Key** — Enter, test, and save your FoodData Central key
- **Cache Management** — View cached item count and clear the nutrition cache
- **Export Data** — Download meal log as CSV (PDF export requires `reportlab`)
- **Appearance** — Toggle dark mode (requires app restart for full effect)

---

## Nutrition Lookup Flow

When analyzing a food item, the app follows this fallback chain:

```
Cache (7-day expiry)
  ↓ miss
USDA FoodData Central API (if key configured)
  ↓ miss / no key
Open Food Facts API (no key needed)
  ↓ miss
Local Nutrition DB (60+ foods)
  ↓ miss
No data found
```

Each successful lookup (except cache hits) is written to the local JSON cache for faster future access.

---

## File Structure

```
NutriSnapAI/
├── app.py                  # Main application (single-file, ~1000 lines)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── meal_log.csv            # Auto-generated — meal history log
├── nutrition_cache.json    # Auto-generated — cached API nutrition data
└── nutri_config.json       # Auto-generated — saved settings (API key, dark mode)
```

---

## Dependencies

```
gradio
torch
torchvision
transformers
pillow
pandas
matplotlib
plotly
ultralytics
opencv-python-headless
requests>=2.28.0
python-dotenv>=1.0.0
```

---

## License

MIT License — use freely.
