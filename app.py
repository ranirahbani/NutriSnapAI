"""
NutriSnap AI - Single-file food tracking application.
Uses YOLOv8 for detection, HuggingFace classifier as fallback,
Gradio for UI, Plotly/Matplotlib for dashboard charts.
Run: python app.py
"""

import os
import csv
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import gradio as gr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Nutrition Database (50+ foods, per 100g values)
# ---------------------------------------------------------------------------

NUTRITION_DB = {
    "apple": {"calories": 52, "protein": 0.3, "carbs": 14.0, "fat": 0.2, "typical_g": 180},
    "banana": {"calories": 89, "protein": 1.1, "carbs": 23.0, "fat": 0.3, "typical_g": 120},
    "orange": {"calories": 47, "protein": 0.9, "carbs": 12.0, "fat": 0.1, "typical_g": 150},
    "broccoli": {"calories": 34, "protein": 2.8, "carbs": 7.0, "fat": 0.4, "typical_g": 150},
    "carrot": {"calories": 41, "protein": 0.9, "carbs": 10.0, "fat": 0.2, "typical_g": 80},
    "tomato": {"calories": 18, "protein": 0.9, "carbs": 3.9, "fat": 0.2, "typical_g": 120},
    "pizza": {"calories": 266, "protein": 11.0, "carbs": 33.0, "fat": 10.0, "typical_g": 250},
    "hamburger": {"calories": 295, "protein": 17.0, "carbs": 24.0, "fat": 14.0, "typical_g": 200},
    "sandwich": {"calories": 250, "protein": 12.0, "carbs": 30.0, "fat": 9.0, "typical_g": 200},
    "hot_dog": {"calories": 290, "protein": 11.0, "carbs": 24.0, "fat": 17.0, "typical_g": 150},
    "donut": {"calories": 452, "protein": 5.0, "carbs": 51.0, "fat": 25.0, "typical_g": 80},
    "cake": {"calories": 350, "protein": 5.0, "carbs": 50.0, "fat": 14.0, "typical_g": 120},
    "rice": {"calories": 130, "protein": 2.7, "carbs": 28.0, "fat": 0.3, "typical_g": 200},
    "pasta": {"calories": 131, "protein": 5.0, "carbs": 25.0, "fat": 1.1, "typical_g": 250},
    "chicken": {"calories": 165, "protein": 31.0, "carbs": 0.0, "fat": 3.6, "typical_g": 150},
    "steak": {"calories": 271, "protein": 26.0, "carbs": 0.0, "fat": 18.0, "typical_g": 200},
    "salmon": {"calories": 208, "protein": 20.0, "carbs": 0.0, "fat": 13.0, "typical_g": 150},
    "egg": {"calories": 155, "protein": 13.0, "carbs": 1.1, "fat": 11.0, "typical_g": 50},
    "bread": {"calories": 265, "protein": 9.0, "carbs": 49.0, "fat": 3.2, "typical_g": 30},
    "cheese": {"calories": 402, "protein": 25.0, "carbs": 1.3, "fat": 33.0, "typical_g": 40},
    "salad": {"calories": 20, "protein": 1.5, "carbs": 3.5, "fat": 0.2, "typical_g": 150},
    "lettuce": {"calories": 15, "protein": 1.4, "carbs": 2.9, "fat": 0.2, "typical_g": 100},
    "spinach": {"calories": 23, "protein": 2.9, "carbs": 3.6, "fat": 0.4, "typical_g": 100},
    "potato": {"calories": 77, "protein": 2.0, "carbs": 17.0, "fat": 0.1, "typical_g": 200},
    "fries": {"calories": 312, "protein": 3.4, "carbs": 41.0, "fat": 15.0, "typical_g": 150},
    "avocado": {"calories": 160, "protein": 2.0, "carbs": 8.5, "fat": 15.0, "typical_g": 150},
    "grapes": {"calories": 69, "protein": 0.7, "carbs": 18.0, "fat": 0.2, "typical_g": 150},
    "strawberry": {"calories": 32, "protein": 0.7, "carbs": 7.7, "fat": 0.3, "typical_g": 150},
    "blueberry": {"calories": 57, "protein": 0.7, "carbs": 14.0, "fat": 0.3, "typical_g": 100},
    "watermelon": {"calories": 30, "protein": 0.6, "carbs": 7.6, "fat": 0.2, "typical_g": 300},
    "mushroom": {"calories": 22, "protein": 3.1, "carbs": 3.3, "fat": 0.3, "typical_g": 100},
    "corn": {"calories": 86, "protein": 3.3, "carbs": 19.0, "fat": 1.4, "typical_g": 150},
    "onion": {"calories": 40, "protein": 1.1, "carbs": 9.3, "fat": 0.1, "typical_g": 100},
    "bell_pepper": {"calories": 31, "protein": 1.0, "carbs": 6.0, "fat": 0.3, "typical_g": 150},
    "cucumber": {"calories": 16, "protein": 0.7, "carbs": 3.6, "fat": 0.1, "typical_g": 200},
    "shrimp": {"calories": 99, "protein": 24.0, "carbs": 0.2, "fat": 0.3, "typical_g": 100},
    "tofu": {"calories": 76, "protein": 8.0, "carbs": 1.9, "fat": 4.8, "typical_g": 150},
    "beans": {"calories": 132, "protein": 8.9, "carbs": 22.0, "fat": 0.5, "typical_g": 150},
    "nuts": {"calories": 607, "protein": 21.0, "carbs": 22.0, "fat": 52.0, "typical_g": 50},
    "yogurt": {"calories": 59, "protein": 10.0, "carbs": 3.6, "fat": 0.4, "typical_g": 200},
    "milk": {"calories": 42, "protein": 3.4, "carbs": 5.0, "fat": 1.0, "typical_g": 250},
    "sushi": {"calories": 150, "protein": 6.0, "carbs": 22.0, "fat": 4.0, "typical_g": 200},
    "soup": {"calories": 50, "protein": 3.0, "carbs": 6.0, "fat": 1.5, "typical_g": 300},
    "chocolate": {"calories": 546, "protein": 5.0, "carbs": 61.0, "fat": 31.0, "typical_g": 50},
    "ice_cream": {"calories": 207, "protein": 3.5, "carbs": 24.0, "fat": 11.0, "typical_g": 100},
    "cereal": {"calories": 379, "protein": 8.0, "carbs": 77.0, "fat": 3.5, "typical_g": 60},
    "taco": {"calories": 226, "protein": 10.0, "carbs": 20.0, "fat": 12.0, "typical_g": 180},
    "bacon": {"calories": 541, "protein": 37.0, "carbs": 1.4, "fat": 42.0, "typical_g": 30},
    "sausage": {"calories": 301, "protein": 18.0, "carbs": 2.0, "fat": 25.0, "typical_g": 80},
    "pepperoni": {"calories": 494, "protein": 22.0, "carbs": 2.0, "fat": 44.0, "typical_g": 30},
    "lemon": {"calories": 29, "protein": 1.1, "carbs": 9.3, "fat": 0.3, "typical_g": 60},
    "pear": {"calories": 57, "protein": 0.4, "carbs": 15.0, "fat": 0.1, "typical_g": 180},
    "peach": {"calories": 39, "protein": 0.9, "carbs": 10.0, "fat": 0.3, "typical_g": 150},
    "mango": {"calories": 60, "protein": 0.8, "carbs": 15.0, "fat": 0.4, "typical_g": 200},
    "pineapple": {"calories": 50, "protein": 0.5, "carbs": 13.0, "fat": 0.1, "typical_g": 150},
    "coconut": {"calories": 354, "protein": 3.3, "carbs": 15.0, "fat": 33.0, "typical_g": 100},
    "celery": {"calories": 16, "protein": 0.7, "carbs": 3.0, "fat": 0.2, "typical_g": 100},
    "cauliflower": {"calories": 25, "protein": 1.9, "carbs": 5.0, "fat": 0.3, "typical_g": 150},
}

# COCO class ID -> food name mapping (YOLOv8 trained on COCO)
COCO_FOOD_CLASSES = {
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot_dog", 53: "pizza",
    54: "donut", 55: "cake",
}

CSV_FILE = "meal_log.csv"
CSV_COLUMNS = ["Date", "Time", "Food", "Calories", "Protein (g)", "Carbs (g)", "Fat (g)", "Portion", "Confirmed"]


# ---------------------------------------------------------------------------
# Model Loading (with graceful fallbacks)
# ---------------------------------------------------------------------------

yolo_model = None
hf_classifier = None
hf_processor = None


def load_yolo():
    """Load YOLOv8n model for food detection."""
    global yolo_model
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")
        print("[NutriSnap] YOLOv8n loaded successfully")
        return True
    except Exception as e:
        print(f"[NutriSnap] YOLO failed to load: {e}")
        return False


def load_hf_classifier():
    """Load HuggingFace food classifier as fallback."""
    global hf_classifier, hf_processor
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        model_name = "yvelos/beit-food-384"
        hf_processor = AutoImageProcessor.from_pretrained(model_name)
        hf_classifier = AutoModelForImageClassification.from_pretrained(model_name)
        hf_classifier.eval()
        print(f"[NutriSnap] HF classifier ({model_name}) loaded successfully")
        return True
    except Exception as e:
        print(f"[NutriSnap] HF classifier failed to load: {e}")
        return False


# ---------------------------------------------------------------------------
# CSV Logging
# ---------------------------------------------------------------------------

def ensure_csv():
    """Create CSV file with headers if it doesn't exist."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def log_meal(food, calories, protein, carbs, fat, portion, confirmed=True):
    """Log a meal entry to CSV."""
    ensure_csv()
    now = datetime.now()
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
            food, calories, protein, carbs, fat, portion, confirmed
        ])


def read_log():
    """Read all meal log entries."""
    ensure_csv()
    try:
        df = pd.read_csv(CSV_FILE)
        # Auto-migrate if columns changed
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[CSV_COLUMNS]
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=CSV_COLUMNS)


def calculate_nutrition(food_name, grams):
    """Calculate nutrition for a given portion."""
    key = food_name.lower().strip().replace(" ", "_")
    if key not in NUTRITION_DB:
        return None
    info = NUTRITION_DB[key]
    factor = grams / 100.0
    return {
        "calories": round(info["calories"] * factor, 1),
        "protein": round(info["protein"] * factor, 1),
        "carbs": round(info["carbs"] * factor, 1),
        "fat": round(info["fat"] * factor, 1),
    }


# ---------------------------------------------------------------------------
# Food Analysis Pipeline
# ---------------------------------------------------------------------------

def estimate_portion(bbox, img_shape):
    """Estimate portion size from bounding box area relative to image."""
    x1, y1, x2, y2 = bbox
    box_area = (x2 - x1) * (y2 - y1)
    img_area = img_shape[0] * img_shape[1]
    ratio = box_area / img_area
    if ratio < 0.05:
        return "Small", 0.5
    elif ratio < 0.15:
        return "Medium", 1.0
    else:
        return "Large", 1.5


def detect_with_yolo(image_np):
    """Run YOLOv8 detection and extract food items."""
    if yolo_model is None:
        return []
    try:
        results = yolo_model(image_np, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                if cls_id in COCO_FOOD_CLASSES and conf > 0.25:
                    bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    food_name = COCO_FOOD_CLASSES[cls_id]
                    detections.append({
                        "food": food_name,
                        "bbox": bbox,
                        "confidence": round(conf, 2),
                    })
        return detections
    except Exception as e:
        print(f"[NutriSnap] YOLO detection error: {e}")
        return []


def classify_with_hf(image_pil):
    """Use HuggingFace classifier as fallback."""
    if hf_classifier is None or hf_processor is None:
        return []
    try:
        import torch
        inputs = hf_processor(images=image_pil, return_tensors="pt")
        with torch.no_grad():
            outputs = hf_classifier(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top5 = torch.topk(probs, 5, dim=-1)
        results = []
        for i in range(5):
            idx = top5.indices[0][i].item()
            score = top5.values[0][i].item()
            label = hf_classifier.config.id2label[idx].lower().replace("-", "_").replace(" ", "_")
            # Match to our DB
            for db_key in NUTRITION_DB:
                if db_key in label or label in db_key:
                    results.append({"food": db_key, "confidence": round(score, 2)})
                    break
        return results[:3] if results else []
    except Exception as e:
        print(f"[NutriSnap] HF classification error: {e}")
        return []


def draw_annotations(image_np, detections):
    """Draw bounding boxes and labels on the image."""
    annotated = image_np.copy()
    colors = [(0, 255, 0), (255, 165, 0), (0, 165, 255), (255, 0, 255),
              (0, 255, 255), (255, 0, 0), (128, 255, 0), (0, 128, 255)]
    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = f"{det['food'].replace('_', ' ').title()} ({det.get('calories', '?')} cal)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 6), font, 0.6, (255, 255, 255), 2)
    return annotated


def analyze_image(image_path):
    """
    Full analysis pipeline:
    1. Try YOLOv8 for multi-food detection
    2. If no foods found, try HuggingFace classifier
    3. Calculate nutrition, estimate portions, log results
    """
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_shape = img_np.shape[:2]

    # Step 1: YOLO detection
    detections = detect_with_yolo(img_bgr)

    # Step 2: Fallback to HF classifier
    if not detections:
        hf_results = classify_with_hf(img_pil)
        if hf_results:
            # Assign full-image bbox for classifier results
            h, w = img_shape
            for r in hf_results:
                r["bbox"] = [int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)]
            detections = hf_results

    if not detections:
        return None, "No food items detected. Try a clearer photo of a meal.", None

    # Step 3: Calculate nutrition and portions
    results = []
    total_cal, total_pro, total_carb, total_fat = 0, 0, 0, 0
    for det in detections:
        food = det["food"]
        portion_label, portion_mult = estimate_portion(det["bbox"], img_shape)
        typical_g = NUTRITION_DB.get(food, {}).get("typical_g", 150)
        grams = round(typical_g * portion_mult)
        nutr = calculate_nutrition(food, grams)
        if nutr:
            det.update(nutr)
            det["portion"] = portion_label
            det["grams"] = grams
            results.append(det)
            total_cal += nutr["calories"]
            total_pro += nutr["protein"]
            total_carb += nutr["carbs"]
            total_fat += nutr["fat"]
            log_meal(food.replace("_", " ").title(), nutr["calories"],
                     nutr["protein"], nutr["carbs"], nutr["fat"], portion_label)

    # Step 4: Annotate image
    annotated = draw_annotations(img_bgr, results)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Step 5: Build summary markdown
    summary_lines = [f"### Detected {len(results)} food item(s)\n"]
    summary_lines.append("| Food | Portion | Grams | Calories | Protein | Carbs | Fat |")
    summary_lines.append("|------|---------|-------|----------|---------|-------|-----|")
    for r in results:
        name = r["food"].replace("_", " ").title()
        summary_lines.append(
            f"| {name} | {r['portion']} | {r['grams']}g "
            f"| {r['calories']} | {r['protein']}g | {r['carbs']}g | {r['fat']}g |"
        )
    summary_lines.append(f"\n**Totals: {round(total_cal, 1)} cal | "
                         f"{round(total_pro, 1)}g protein | "
                         f"{round(total_carb, 1)}g carbs | "
                         f"{round(total_fat, 1)}g fat**")

    return annotated_rgb, "\n".join(summary_lines), results


# ---------------------------------------------------------------------------
# Dashboard Charts
# ---------------------------------------------------------------------------

def build_dashboard():
    """Generate dashboard charts from meal log."""
    df = read_log()
    if df.empty or len(df) == 0:
        return (None, None, None, None,
                "**No data yet.** Upload and analyze a meal to see trends here.")

    df["Calories"] = pd.to_numeric(df["Calories"], errors="coerce").fillna(0)
    df["Protein (g)"] = pd.to_numeric(df["Protein (g)"], errors="coerce").fillna(0)
    df["Carbs (g)"] = pd.to_numeric(df["Carbs (g)"], errors="coerce").fillna(0)
    df["Fat (g)"] = pd.to_numeric(df["Fat (g)"], errors="coerce").fillna(0)

    # 1. Daily calorie intake (bar chart)
    daily = df.groupby("Date")["Calories"].sum().reset_index()
    fig_daily = px.bar(daily, x="Date", y="Calories",
                       title="Daily Calorie Intake",
                       color_discrete_sequence=["#003366"])
    fig_daily.update_layout(template="plotly_white", height=350)

    # 2. Macro distribution (pie chart)
    macro_totals = {
        "Protein": df["Protein (g)"].sum(),
        "Carbs": df["Carbs (g)"].sum(),
        "Fat": df["Fat (g)"].sum(),
    }
    fig_macro = px.pie(names=list(macro_totals.keys()), values=list(macro_totals.values()),
                       title="Macronutrient Distribution",
                       color_discrete_sequence=["#003366", "#0066cc", "#66b2ff"])
    fig_macro.update_layout(template="plotly_white", height=350)

    # 3. Weekly calorie trend (line chart)
    df["DateObj"] = pd.to_datetime(df["Date"], errors="coerce")
    weekly = df.dropna(subset=["DateObj"]).set_index("DateObj").resample("W")["Calories"].sum().reset_index()
    weekly.columns = ["Week", "Calories"]
    fig_weekly = px.line(weekly, x="Week", y="Calories",
                         title="Weekly Calorie Trend",
                         color_discrete_sequence=["#003366"])
    fig_weekly.update_layout(template="plotly_white", height=350)

    # 4. Top foods eaten (horizontal bar)
    top = df["Food"].value_counts().head(8).reset_index()
    top.columns = ["Food", "Count"]
    fig_top = px.bar(top, y="Food", x="Count", orientation="h",
                     title="Top Foods Eaten",
                     color_discrete_sequence=["#0066cc"])
    fig_top.update_layout(template="plotly_white", height=350, yaxis={"categoryorder": "total ascending"})

    # Summary stats
    total_cal = round(df["Calories"].sum(), 1)
    total_meals = len(df)
    avg_cal = round(df["Calories"].mean(), 1)
    stats = (f"**Total meals logged:** {total_meals} | "
             f"**Total calories:** {total_cal} | "
             f"**Avg per meal:** {avg_cal} cal")

    return fig_daily, fig_macro, fig_weekly, fig_top, stats


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
#title { text-align: center; padding: 16px 0 4px; }
#title h1 { color: #003366; font-size: 2.2em; margin: 0; }
#subtitle { text-align: center; color: #666; margin-bottom: 16px; }
.gradio-container { max-width: 960px !important; }
.card { background: #ffffff; border-radius: 12px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
footer { display: none; }
"""

TIPS_MD = """
## Nutrition Tips & Guidelines

### Recommended Daily Intake (RDI)

| Nutrient | Adult Male | Adult Female |
|----------|-----------|-------------|
| Calories | 2,500 kcal | 2,000 kcal |
| Protein | 56 g | 46 g |
| Carbs | 300 g | 250 g |
| Fat | 80 g | 65 g |

### Tracking Best Practices

1. **Log immediately** after eating for best accuracy
2. **Take clear photos** with good lighting - avoid blurry or dark images
3. **Include the full plate** so all foods are visible
4. **Review portion sizes** - the app estimates based on image area; adjust mentally if needed

### Understanding Your Dashboard

- **Daily Calories**: Compare against your RDI to avoid over/under-eating
- **Macro Split**: A balanced diet is roughly 25% protein, 50% carbs, 25% fat
- **Weekly Trend**: Look for consistency rather than perfection
- **Top Foods**: Helps identify if your diet lacks variety

### Healthy Eating Reminders

- Aim for **5+ servings of vegetables** daily
- Choose **whole grains** over refined
- Stay hydrated: **8 glasses of water** per day
- Limit **processed foods** and added sugars
- Include **lean protein** in every meal
"""


def build_ui():
    """Build the complete Gradio interface."""
    with gr.Blocks(css=CSS, title="NutriSnap AI", theme=gr.themes.Soft()) as demo:
        # Header
        gr.Markdown('<div id="title"><h1>NutriSnap AI</h1></div>', elem_id="title")
        gr.Markdown('<div id="subtitle">AI-Powered Food Tracking & Nutrition Analysis</div>', elem_id="subtitle")

        with gr.Tabs():
            # ---- Tab 1: Upload & Analyze ----
            with gr.TabItem("📸 Upload & Analyze"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.File(label="Upload Meal Photo", file_types=["image"],
                                               type="filepath")
                        analyze_btn = gr.Button("🔍 Analyze Food", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Annotated Result", type="numpy")
                with gr.Row():
                    output_md = gr.Markdown(value="Upload a photo and click **Analyze Food** to see results.")

            # ---- Tab 2: Dashboard ----
            with gr.TabItem("📊 Dashboard"):
                dash_stats = gr.Markdown()
                with gr.Row():
                    chart_daily = gr.Plot(label="Daily Calories")
                    chart_macro = gr.Plot(label="Macro Distribution")
                with gr.Row():
                    chart_weekly = gr.Plot(label="Weekly Trend")
                    chart_top = gr.Plot(label="Top Foods")
                refresh_btn = gr.Button("🔄 Refresh Dashboard")

            # ---- Tab 3: Food Log ----
            with gr.TabItem("📋 Food Log"):
                log_table = gr.Dataframe(headers=CSV_COLUMNS, label="Meal History", interactive=False)
                log_refresh = gr.Button("🔄 Refresh Log")

            # ---- Tab 4: Tips ----
            with gr.TabItem("💡 Nutrition Tips"):
                gr.Markdown(TIPS_MD)

        # ---- Event Handlers ----
        def on_analyze(file):
            if file is None:
                return None, "Please upload an image first.", None
            annotated, summary, detections = analyze_image(file)
            return annotated, summary

        def on_refresh_dashboard():
            return build_dashboard()

        def on_refresh_log():
            return read_log()

        analyze_btn.click(fn=on_analyze, inputs=[input_image],
                          outputs=[output_image, output_md])
        refresh_btn.click(fn=on_refresh_dashboard,
                          outputs=[chart_daily, chart_macro, chart_weekly, chart_top, dash_stats])
        log_refresh.click(fn=on_refresh_log, outputs=[log_table])

        # Load dashboard on tab select
        demo.load(fn=on_refresh_dashboard,
                 outputs=[chart_daily, chart_macro, chart_weekly, chart_top, dash_stats])
        demo.load(fn=on_refresh_log, outputs=[log_table])

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[NutriSnap] Loading models...")
    load_yolo()
    load_hf_classifier()
    print("[NutriSnap] Starting Gradio app...")
    demo = build_ui()
    demo.launch(share=True)
