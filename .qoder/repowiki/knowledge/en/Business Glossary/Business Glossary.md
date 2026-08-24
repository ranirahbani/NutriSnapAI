---
kind: business_term
name: Business Glossary
category: business_term
scope:
    - '**'
---

### NutriSnap AI
- Definition：The project's internal product name for the single-file Python web application that accepts meal photos, detects food items via YOLOv8 (with a HuggingFace classifier fallback), estimates portion sizes from bounding-box area ratios, calculates nutrition from an embedded per-100g database, logs results to CSV, and renders interactive dashboards via Gradio.
- Aliases：NutriSnap

### meal_log.csv
- Definition：The persistent local CSV file where every analyzed meal entry is appended with columns Date, Time, Food, Calories, Protein (g), Carbs (g), Fat (g), Portion, Confirmed. It is auto-created on first write and read back to power the dashboard charts and food log tab.
- Aliases：CSV_FILE、meal log

### portion estimation
- Definition：The heuristic that infers a small/medium/large portion label and gram multiplier (0.5x / 1.0x / 1.5x) by comparing the detected food bounding-box area to the total image area, then scales the food's typical gram weight from the nutrition database to compute calories and macros.
- Aliases：portion size、grams estimate

### COCO food classes
- Definition：A hard-coded mapping from COCO object-detection class IDs (46–55) to food names recognized by the YOLOv8 model (banana, apple, sandwich, orange, broccoli, carrot, hot_dog, pizza, donut, cake). Only detections whose class ID appears in this map are treated as food.
- Aliases：COCO_FOOD_CLASSES

### nutrition database
- Definition：An in-memory dictionary of ~50 foods, each keyed by a normalized name and storing per-100g values for calories, protein, carbs, fat, plus a typical gram weight used to derive portioned nutrition after detection.
- Aliases：NUTRITION_DB
