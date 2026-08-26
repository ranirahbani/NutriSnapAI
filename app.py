"""
NutriSnap AI - Single-file food tracking application.
Uses YOLOv8 for detection, HuggingFace classifier as second-stage refinement,
Gradio for UI, Plotly/Matplotlib for dashboard charts.
USDA & Open Food Facts APIs for nutrition data with local fallback.
Run: python app.py
"""

import os
import csv
import json
import time
import warnings
import difflib
from datetime import datetime, timedelta
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
import requests

warnings.filterwarnings("ignore")

# ============================================
# CONFIGURATION & CONSTANTS
# ============================================

CSV_FILE = "meal_log.csv"
CSV_COLUMNS = ["Date", "Time", "Food", "Calories", "Protein (g)", "Carbs (g)", "Fat (g)", "Portion", "Confirmed"]
CACHE_FILE = "nutrition_cache.json"
CONFIG_FILE = "nutri_config.json"
CACHE_EXPIRY_DAYS = 7

# State: last analysis results for Save Meal button
_last_analysis_results = None

# Color theme
THEME_PRIMARY = "#2C7A4A"
THEME_SECONDARY = "#1A5A3A"
THEME_ACCENT = "#4CAF50"
THEME_BG = "#F5F9F8"
THEME_CHART_COLORS = ["#2C7A4A", "#4CAF50", "#81C784", "#A5D6A7"]

# ============================================
# NUTRITION DATABASE (last-resort fallback)
# ============================================

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
    "chicken_wings": {"calories": 203, "protein": 18.0, "carbs": 0.0, "fat": 14.0, "typical_g": 150},
    "eggs": {"calories": 155, "protein": 13.0, "carbs": 1.1, "fat": 11.0, "typical_g": 100},
    "pancakes": {"calories": 227, "protein": 6.0, "carbs": 28.0, "fat": 10.0, "typical_g": 150},
    "waffles": {"calories": 291, "protein": 8.0, "carbs": 33.0, "fat": 14.0, "typical_g": 150},
    "tacos": {"calories": 226, "protein": 10.0, "carbs": 20.0, "fat": 12.0, "typical_g": 180},
    "nachos": {"calories": 346, "protein": 9.0, "carbs": 36.0, "fat": 19.0, "typical_g": 200},
}

# Per-unit weights for countable food items (grams per unit)
UNIT_WEIGHTS = {
    # --- Proteins ---
    "chicken_wings": 30,      # grams per wing
    "chicken wings": 30,
    "chicken_drumstick": 75,  # grams per drumstick
    "drumstick": 75,
    "chicken_breast": 175,    # grams per breast
    "chicken_thigh": 110,     # grams per thigh
    "chicken_nugget": 18,     # grams per nugget
    "chicken_nuggets": 18,
    "nugget": 18,
    "nuggets": 18,
    "meatball": 30,           # grams per meatball
    "meatballs": 30,
    "shrimp": 10,             # grams per shrimp
    "prawns": 10,
    "fish_fillet": 140,       # grams per fillet
    "steak": 225,             # grams per steak
    "lamb_chop": 100,         # grams per chop
    "lamb chop": 100,
    "pork_chop": 150,         # grams per chop
    "pork chop": 150,
    "sausage": 70,            # grams per sausage
    "sausages": 70,
    "bacon_strip": 8,         # grams per strip
    "bacon": 8,
    "hot_dog": 50,            # grams per hot dog
    "hotdog": 50,
    # --- Bread / Pastry ---
    "bread_slice": 30,        # grams per slice
    "bread": 30,
    "croissant": 60,          # grams per croissant
    "donut": 60,              # grams per donut
    "donuts": 60,
    "doughnut": 60,
    "muffin": 115,            # grams per muffin
    "cookie": 30,             # grams per cookie
    "cookies": 30,
    "biscuit": 15,            # grams per biscuit
    "biscuits": 15,
    "pancake": 75,            # grams per pancake
    "pancakes": 75,
    "waffle": 90,             # grams per waffle
    "waffles": 90,
    "bagel": 100,             # grams per bagel
    "toast": 30,              # grams per slice of toast
    "roll": 50,               # grams per roll
    "dinner_roll": 50,
    "cupcake": 65,            # grams per cupcake
    "cupcakes": 65,
    "macaron": 15,            # grams per macaron
    "macarons": 15,
    "brownie": 55,            # grams per brownie
    "brownies": 55,
    # --- Pizza / Italian ---
    "pizza": 110,             # grams per slice
    "pizza_slice": 110,
    "ravioli": 15,            # grams per piece
    "mozzarella_stick": 28,   # grams per stick
    "mozzarella_sticks": 28,
    "garlic_bread": 45,       # grams per piece
    "bruschetta": 40,         # grams per piece
    # --- Mexican ---
    "taco": 80,               # grams per taco
    "tacos": 80,
    "burrito": 250,           # grams per burrito
    "enchilada": 150,         # grams per enchilada
    "enchiladas": 150,
    "quesadilla": 180,        # grams per quesadilla
    "nacho": 8,               # grams per nacho chip
    "nachos": 8,
    "empanada": 90,           # grams per empanada
    "empanadas": 90,
    "churro": 40,             # grams per churro
    "churros": 40,
    # --- Asian ---
    "sushi": 30,              # grams per piece
    "sushi_roll": 30,
    "dumpling": 25,           # grams per dumpling
    "dumplings": 25,
    "spring_roll": 60,        # grams per spring roll
    "spring_rolls": 60,
    "egg_roll": 85,           # grams per egg roll
    "egg_rolls": 85,
    "dim_sum": 30,            # grams per piece
    "gyoza": 25,              # grams per gyoza
    "samosa": 50,             # grams per samosa
    "samosas": 50,
    "tempura": 30,            # grams per piece
    "wonton": 20,             # grams per wonton
    "wontons": 20,
    # --- Breakfast ---
    "egg": 50,                # grams per egg
    "eggs": 50,
    "boiled_egg": 50,
    "fried_egg": 55,
    "sausage_patty": 45,      # grams per patty
    "hash_brown": 60,         # grams per hash brown
    "hash_browns": 60,
    "french_toast": 65,       # grams per slice
    # --- Snacks / Sides ---
    "fries": 130,             # grams per portion
    "french_fries": 130,
    "onion_ring": 15,         # grams per ring
    "onion_rings": 15,
    "chicken_tender": 40,     # grams per tender
    "chicken_tenders": 40,
    "jalapeno_popper": 25,    # grams per popper
    "corn_on_cob": 150,       # grams per cob
    "potato_wedge": 30,       # grams per wedge
    "potato_wedges": 30,
    # --- Fruits ---
    "apple": 180,             # grams per apple
    "banana": 120,            # grams per banana
    "orange": 150,            # grams per orange
    "strawberry": 12,         # grams per strawberry
    "strawberries": 12,
    "grape": 5,               # grams per grape
    "grapes": 5,
    "cherry": 8,              # grams per cherry
    "cherries": 8,
    "blueberry": 1.5,         # grams per blueberry
    "blueberries": 1.5,
    # --- Desserts ---
    "ice_cream_scoop": 70,    # grams per scoop
    "ice_cream": 70,
    "cake_slice": 100,        # grams per slice
    "cake": 100,
    "pie_slice": 125,         # grams per slice
    "chocolate_bar": 40,      # grams per bar
    "truffle": 15,            # grams per truffle
    "truffles": 15,
    # --- Sandwiches / Burgers ---
    "hamburger": 200,         # grams per burger
    "burger": 200,
    "cheeseburger": 220,      # grams per cheeseburger
    "sandwich": 180,          # grams per sandwich
    "slider": 85,             # grams per slider
    "sliders": 85,
    "wrap": 200,              # grams per wrap
}

# Food hierarchy for deduplication: generic -> [specific variants]
FOOD_HIERARCHY = {
    "chicken": ["chicken_wings", "chicken_drumstick", "chicken_breast", "chicken_thigh", "chicken_nuggets", "chicken_tenders", "fried_chicken", "chicken"],
    "fish": ["salmon", "tuna", "fish_fillet", "grilled_fish", "fried_fish", "fish"],
    "potato": ["fries", "french_fries", "hash_browns", "potato_wedges", "baked_potato", "mashed_potato", "potato"],
    "bread": ["toast", "bread_slice", "garlic_bread", "baguette", "roll", "dinner_roll", "bread"],
    "pasta": ["spaghetti", "penne", "ravioli", "lasagna", "fettuccine", "macaroni", "pasta"],
    "rice": ["fried_rice", "white_rice", "brown_rice", "sushi_rice", "rice"],
    "egg": ["fried_egg", "boiled_egg", "scrambled_eggs", "omelette", "eggs", "egg"],
    "pork": ["pork_chop", "bacon", "ham", "pulled_pork", "pork"],
    "beef": ["steak", "hamburger", "meatballs", "beef"],
    "shrimp": ["prawns", "tempura_shrimp", "shrimp"],
    "cake": ["cupcake", "cheesecake", "chocolate_cake", "cake_slice", "cake"],
    "pizza": ["pizza_slice", "pepperoni_pizza", "margherita", "pizza"],
    "sandwich": ["club_sandwich", "grilled_cheese", "blt", "sub", "sandwich"],
    "salad": ["caesar_salad", "greek_salad", "garden_salad", "coleslaw", "salad"],
    "soup": ["tomato_soup", "chicken_soup", "miso_soup", "chowder", "soup"],
    "ice_cream": ["gelato", "frozen_yogurt", "sorbet", "ice_cream_scoop", "ice_cream"],
    "donut": ["doughnut", "donuts", "donut"],
    "taco": ["tacos", "taco"],
    "sushi": ["sushi_roll", "nigiri", "sashimi", "maki", "sushi"],
}

# COCO class ID -> food name mapping (YOLOv8 trained on COCO)
COCO_FOOD_CLASSES = {
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot_dog", 53: "pizza",
    54: "donut", 55: "cake",
}

# HuggingFace model output -> NUTRITION_DB key mapping
HF_TO_DB_MAP = {
    "french_fries": "fries",
    "hamburger": "hamburger",
    "cheeseburger": "hamburger",
    "hot_dog": "hot_dog",
    "pizza": "pizza",
    "caesar_salad": "salad",
    "greek_salad": "salad",
    "grilled_salmon": "salmon",
    "fried_rice": "rice",
    "chicken_wings": "chicken_wings",
    "ice_cream": "ice_cream",
    "chocolate_cake": "cake",
    "spaghetti_bolognese": "pasta",
    "spaghetti_carbonara": "pasta",
    "lasagna": "pasta",
    "sushi": "sushi",
    "steak": "steak",
    "fried_chicken": "chicken",
    "fish_and_chips": "fries",
    "club_sandwich": "sandwich",
    "pulled_pork_sandwich": "sandwich",
    "lobster_bisque": "soup",
    "miso_soup": "soup",
    "omelette": "eggs",
    "scrambled_eggs": "eggs",
    "pancakes": "pancakes",
    "waffles": "waffles",
    "donuts": "donut",
    "apple_pie": "cake",
    "cheesecake": "cake",
    "tacos": "tacos",
    "nachos": "nachos",
    "guacamole": "avocado",
    "bread_pudding": "bread",
    "garlic_bread": "bread",
}


def fuzzy_match_food(name, threshold=0.6):
    """Try fuzzy matching a food name against NUTRITION_DB keys."""
    name_clean = name.lower().strip().replace(" ", "_").replace("-", "_")
    # Direct match first
    if name_clean in NUTRITION_DB:
        return name_clean
    # Check HF map
    if name_clean in HF_TO_DB_MAP:
        return HF_TO_DB_MAP[name_clean]
    # Fuzzy match against DB keys
    matches = difflib.get_close_matches(name_clean, list(NUTRITION_DB.keys()), n=1, cutoff=threshold)
    if matches:
        return matches[0]
    # Also try partial matching: if any DB key is a substring
    for db_key in NUTRITION_DB:
        if db_key in name_clean or name_clean in db_key:
            return db_key
    return name_clean

# ============================================
# NUTRITION API FUNCTIONS
# ============================================

def search_usda_food(query, api_key):
    """Search USDA FoodData Central for nutrition info."""
    try:
        url = "https://api.nal.usda.gov/fdc/v1/foods/search"
        params = {"api_key": api_key, "query": query, "pageSize": 3}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            foods = data.get("foods", [])
            if not foods:
                return None
            food = foods[0]
            nutrients = {}
            for n in food.get("foodNutrients", []):
                name = n.get("nutrientName", "").lower()
                val = n.get("value", 0)
                if "energy" in name and n.get("unitName", "") == "KCAL":
                    nutrients["calories"] = val
                elif "protein" in name:
                    nutrients["protein"] = val
                elif "carbohydrate" in name and "by difference" in name:
                    nutrients["carbs"] = val
                elif name == "total lipid (fat)":
                    nutrients["fat"] = val
            if "calories" in nutrients:
                nutrients.setdefault("protein", 0)
                nutrients.setdefault("carbs", 0)
                nutrients.setdefault("fat", 0)
                nutrients["source"] = "USDA"
                return nutrients
        return None
    except Exception as e:
        print(f"[NutriSnap] USDA API error: {e}")
        return None


def search_openfoodfacts(query):
    """Search Open Food Facts for nutrition info."""
    try:
        url = "https://world.openfoodfacts.org/cgi/search.pl"
        params = {"search_terms": query, "json": 1, "page_size": 3}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            products = data.get("products", [])
            if not products:
                return None
            product = products[0]
            nutriments = product.get("nutriments", {})
            calories = nutriments.get("energy-kcal_100g") or nutriments.get("energy-kcal")
            if calories is None:
                return None
            return {
                "calories": float(calories),
                "protein": float(nutriments.get("proteins_100g", 0) or 0),
                "carbs": float(nutriments.get("carbohydrates_100g", 0) or 0),
                "fat": float(nutriments.get("fat_100g", 0) or 0),
                "source": "OpenFoodFacts",
            }
        return None
    except Exception as e:
        print(f"[NutriSnap] Open Food Facts API error: {e}")
        return None


# ============================================
# CACHE SYSTEM
# ============================================

def _load_cache():
    """Load the nutrition cache from disk."""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_cache(cache):
    """Save the nutrition cache to disk."""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"[NutriSnap] Cache save error: {e}")


def cache_nutrition(food_name, data):
    """Store nutrition data in cache with timestamp."""
    cache = _load_cache()
    cache[food_name.lower().strip()] = {
        "data": data,
        "timestamp": datetime.now().isoformat(),
    }
    _save_cache(cache)


def get_cached_nutrition(food_name):
    """Retrieve cached nutrition data if < 7 days old."""
    cache = _load_cache()
    key = food_name.lower().strip()
    entry = cache.get(key)
    if entry:
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            if datetime.now() - ts < timedelta(days=CACHE_EXPIRY_DAYS):
                return entry["data"]
        except Exception:
            pass
    return None


def get_cache_size():
    """Return number of items in the nutrition cache."""
    cache = _load_cache()
    return len(cache)


def clear_cache():
    """Delete the nutrition cache file."""
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            return "Cache cleared successfully."
        return "Cache was already empty."
    except Exception as e:
        return f"Error clearing cache: {e}"


# ============================================
# SETTINGS FUNCTIONS
# ============================================

def load_config():
    """Load app configuration from disk."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {"usda_api_key": "", "dark_mode": False}


def save_config(config):
    """Save app configuration to disk."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"[NutriSnap] Config save error: {e}")
        return False


def test_usda_connection(api_key):
    """Test USDA API connection with a sample query."""
    if not api_key or not api_key.strip():
        return "Please enter an API key first."
    result = search_usda_food("apple", api_key.strip())
    if result:
        return f"Connection successful! Found: Apple - {result['calories']} kcal/100g"
    return "Connection failed. Check your API key and try again."


def export_csv_file():
    """Return the CSV file path for download."""
    if os.path.exists(CSV_FILE):
        return CSV_FILE
    return None


# ============================================
# AI DETECTION (EXISTING - keep as-is)
# ============================================

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


# ============================================
# CSV LOGGING
# ============================================

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
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df = df[CSV_COLUMNS]
        # Format Time column to HH:MM (strip seconds)
        if "Time" in df.columns:
            df["Time"] = df["Time"].astype(str).str[:5]
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=CSV_COLUMNS)


def delete_log_entry(row_index):
    """Delete a single row from the CSV by 0-based index."""
    ensure_csv()
    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty or row_index < 0 or row_index >= len(df):
            return False
        df = df.drop(index=row_index).reset_index(drop=True)
        df.to_csv(CSV_FILE, index=False)
        return True
    except Exception:
        return False


def clear_all_log():
    """Wipe all entries from the CSV."""
    ensure_csv()
    try:
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(CSV_FILE, index=False)
        return True
    except Exception:
        return False


# ============================================
# ANALYSIS PIPELINE (MODIFIED with API lookup)
# ============================================

def calculate_nutrition(food_name, grams, status_callback=None):
    """
    Calculate nutrition for a given portion using the fallback chain:
    1. Check local cache
    2. Query USDA API (if key configured)
    3. Query Open Food Facts
    4. Fallback to local NUTRITION_DB
    5. Return None if all fail
    """
    key = food_name.lower().strip().replace(" ", "_")
    display_name = food_name.replace("_", " ").title()

    def _notify(msg):
        if status_callback:
            status_callback(msg)

    # 1. Check cache
    cached = get_cached_nutrition(key)
    if cached:
        _notify(f"✅ Using cached data for {display_name}")
        factor = grams / 100.0
        return {
            "calories": round(cached["calories"] * factor, 1),
            "protein": round(cached["protein"] * factor, 1),
            "carbs": round(cached["carbs"] * factor, 1),
            "fat": round(cached["fat"] * factor, 1),
        }

    # 2. Try USDA API
    config = load_config()
    api_key = config.get("usda_api_key", "")
    if api_key and api_key.strip():
        _notify(f"🔍 Searching USDA for {display_name}...")
        usda_result = search_usda_food(food_name, api_key.strip())
        if usda_result:
            _notify(f"✅ Found in USDA: {display_name}")
            cache_nutrition(key, usda_result)
            factor = grams / 100.0
            return {
                "calories": round(usda_result["calories"] * factor, 1),
                "protein": round(usda_result["protein"] * factor, 1),
                "carbs": round(usda_result["carbs"] * factor, 1),
                "fat": round(usda_result["fat"] * factor, 1),
            }

    # 3. Try Open Food Facts
    _notify(f"🔍 Searching Open Food Facts for {display_name}...")
    off_result = search_openfoodfacts(food_name)
    if off_result:
        _notify(f"✅ Found in Open Food Facts: {display_name}")
        cache_nutrition(key, off_result)
        factor = grams / 100.0
        return {
            "calories": round(off_result["calories"] * factor, 1),
            "protein": round(off_result["protein"] * factor, 1),
            "carbs": round(off_result["carbs"] * factor, 1),
            "fat": round(off_result["fat"] * factor, 1),
        }

    # 4. Fallback to local DB
    if key in NUTRITION_DB:
        _notify(f"⚠️ Using fallback local data for {display_name}")
        info = NUTRITION_DB[key]
        factor = grams / 100.0
        return {
            "calories": round(info["calories"] * factor, 1),
            "protein": round(info["protein"] * factor, 1),
            "carbs": round(info["carbs"] * factor, 1),
            "fat": round(info["fat"] * factor, 1),
        }

    # 5. All failed
    _notify(f"❌ No nutrition data found for {display_name}")
    return None


def estimate_portion(bbox, img_shape, food_name=""):
    """
    Estimate portion size from bounding box area relative to image.
    Returns (portion_label, portion_mult, grams_estimate).
    Uses fries-specific weights when food_name contains 'fries'.
    """
    x1, y1, x2, y2 = bbox
    box_area = (x2 - x1) * (y2 - y1)
    img_area = img_shape[0] * img_shape[1]
    ratio = box_area / img_area if img_area > 0 else 0.0

    is_fries = "fries" in food_name.lower() if food_name else False

    if ratio > 0.20:
        label = "Large portion"
        if is_fries:
            grams = 180
        else:
            grams = None  # will use typical_g * 1.5
        mult = 1.5
    elif ratio >= 0.08:
        label = "Medium portion"
        if is_fries:
            grams = 130
        else:
            grams = None
        mult = 1.0
    else:
        label = "Small portion"
        if is_fries:
            grams = 80
        else:
            grams = None
        mult = 0.5

    return label, mult, grams


def estimate_item_count(food_name, bbox, img_shape):
    """
    Estimate number of individual items based on food type and bbox size.
    Returns (count, description) tuple.
    """
    img_area = img_shape[0] * img_shape[1]
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    area_ratio = bbox_area / img_area if img_area > 0 else 0.0
    
    food_lower = food_name.lower().replace(" ", "_")
    
    # Pizza: distinguish between slice vs full pizza
    if food_lower in ["pizza"]:
        if area_ratio > 0.4:  # Takes up most of image = full pizza
            return 8, "Full pizza (8 slices)"
        elif area_ratio > 0.2:  # Medium area = half pizza or 3-4 slices
            return 4, "Half pizza (4 slices)"
        elif area_ratio > 0.1:  # Smaller = 2-3 slices
            return 2, "2 slices"
        else:  # Small = single slice
            return 1, "1 slice"
    
    # Chicken wings: count individual wings
    elif food_lower in ["chicken_wings", "chicken wings"]:
        if area_ratio > 0.35:
            return 6, "6 wings"
        elif area_ratio > 0.2:
            return 4, "4 wings"
        elif area_ratio > 0.1:
            return 3, "3 wings"
        elif area_ratio > 0.05:
            return 2, "2 wings"
        else:
            return 1, "1 wing"
    
    # Fries: portion-based, not countable
    elif food_lower in ["fries", "french_fries"]:
        if area_ratio > 0.3:
            return 1, "Large portion"
        elif area_ratio > 0.15:
            return 1, "Medium portion"
        else:
            return 1, "Small portion"
    
    # Donuts: count individual
    elif food_lower in ["donut", "donuts"]:
        if area_ratio > 0.3:
            return 3, "3 donuts"
        elif area_ratio > 0.15:
            return 2, "2 donuts"
        else:
            return 1, "1 donut"
    
    # Tacos: count individual
    elif food_lower in ["tacos", "taco"]:
        if area_ratio > 0.3:
            return 3, "3 tacos"
        elif area_ratio > 0.15:
            return 2, "2 tacos"
        else:
            return 1, "1 taco"
    
    # Sushi: count pieces
    elif food_lower in ["sushi"]:
        if area_ratio > 0.3:
            return 8, "8 pieces"
        elif area_ratio > 0.15:
            return 6, "6 pieces"
        elif area_ratio > 0.08:
            return 4, "4 pieces"
        else:
            return 2, "2 pieces"
    
    # Eggs: count individual
    elif food_lower in ["eggs", "egg"]:
        if area_ratio > 0.2:
            return 3, "3 eggs"
        elif area_ratio > 0.1:
            return 2, "2 eggs"
        else:
            return 1, "1 egg"
    
    # Pancakes/waffles: count individual
    elif food_lower in ["pancakes"]:
        if area_ratio > 0.25:
            return 3, "3 pancakes"
        elif area_ratio > 0.12:
            return 2, "2 pancakes"
        else:
            return 1, "1 pancake"
    
    elif food_lower in ["waffles"]:
        if area_ratio > 0.25:
            return 2, "2 waffles"
        else:
            return 1, "1 waffle"
    
    # Nachos: usually a portion, not counted individually
    elif food_lower in ["nachos"]:
        if area_ratio > 0.3:
            return 1, "Large portion"
        elif area_ratio > 0.15:
            return 1, "Medium portion"
        else:
            return 1, "Small portion"
    
    # Default: single item
    else:
        return 1, "1 serving"


def deduplicate_results(results):
    """
    Merge duplicates and resolve hierarchy conflicts.
    - If a specific item (chicken_wings) and generic (chicken) both detected
      with overlapping bboxes, remove the generic one.
    - If same food detected in multiple regions, merge into one entry
      with combined count and nutrition.
    """
    if not results:
        return results
    
    # Build reverse hierarchy: specific -> generic
    specific_to_generic = {}
    for generic, specifics in FOOD_HIERARCHY.items():
        for specific in specifics:
            if specific != generic:  # Don't map generic to itself
                specific_to_generic[specific] = generic
    
    # Step 1: Remove generic items if a more specific version overlaps
    filtered = []
    for item in results:
        food = item["food"].lower().replace(" ", "_")
        is_generic = food in FOOD_HIERARCHY
        
        if is_generic:
            # Check if any specific variant overlaps with this generic item
            has_specific_overlap = False
            for other in results:
                if other is item:
                    continue
                other_food = other["food"].lower().replace(" ", "_")
                # Check if other is a specific variant of this generic
                if specific_to_generic.get(other_food) == food:
                    # Check bbox overlap
                    iou = bbox_iou(item["bbox"], other["bbox"])
                    if iou > 0.2:  # Significant overlap
                        has_specific_overlap = True
                        break
            
            if has_specific_overlap:
                # Skip this generic item, keep the specific one
                continue
        
        filtered.append(item)
    
    # Step 2: Merge same food items (non-overlapping regions)
    merged = {}
    for item in filtered:
        food_key = item["food"].lower().replace(" ", "_")
        
        if food_key in merged:
            # Merge: add counts and nutrition
            existing = merged[food_key]
            existing["count"] = existing.get("count", 1) + item.get("count", 1)
            existing["calories"] = existing.get("calories", 0) + item.get("calories", 0)
            existing["protein"] = existing.get("protein", 0) + item.get("protein", 0)
            existing["carbs"] = existing.get("carbs", 0) + item.get("carbs", 0)
            existing["fat"] = existing.get("fat", 0) + item.get("fat", 0)
            existing["grams"] = existing.get("grams", 0) + item.get("grams", 0)
            # Keep the larger bbox for display
            if item.get("bbox"):
                existing_area = (existing["bbox"][2] - existing["bbox"][0]) * (existing["bbox"][3] - existing["bbox"][1])
                new_area = (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1])
                if new_area > existing_area:
                    existing["bbox"] = item["bbox"]
            # Combine descriptions
            if item.get("count_description"):
                existing["count_description"] = item["count_description"]
        else:
            merged[food_key] = item.copy()
    
    return list(merged.values())


def bbox_iou(box_a, box_b):
    """Compute IoU between two bounding boxes [x1,y1,x2,y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def get_uncovered_regions(img_shape, existing_bboxes, grid=(3, 3), overlap_thresh=0.3):
    """
    Divide image into a grid and return regions not significantly overlapping
    with existing YOLO detections.
    Returns list of [x1, y1, x2, y2] regions.
    """
    h, w = img_shape[:2]
    rows, cols = grid
    cell_h = h // rows
    cell_w = w // cols
    uncovered = []
    for r in range(rows):
        for c in range(cols):
            region = [c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h]
            # Check if this region overlaps significantly with any existing detection
            dominated = False
            for det_box in existing_bboxes:
                if bbox_iou(region, det_box) > overlap_thresh:
                    dominated = True
                    break
            if not dominated:
                uncovered.append(region)
    return uncovered


def _resolve_food_name(hf_label):
    """
    Map a raw HF classifier label to a NUTRITION_DB key using:
    1. HF_TO_DB_MAP
    2. Direct DB match
    3. Fuzzy match
    """
    if not hf_label:
        return None
    label = hf_label.lower().strip().replace("-", "_").replace(" ", "_")
    if label in HF_TO_DB_MAP:
        return HF_TO_DB_MAP[label]
    if label in NUTRITION_DB:
        return label
    return fuzzy_match_food(label)


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


def classify_with_hf(image_input, top_k=3):
    """
    Use HuggingFace classifier on an image (file path string OR PIL Image).
    Returns list of dicts with raw HF label + confidence:
      [{"label": "hamburger", "confidence": 0.82}, ...]
    """
    if hf_classifier is None or hf_processor is None:
        return []
    try:
        import torch
        if isinstance(image_input, str):
            image_input = Image.open(image_input).convert("RGB")
        elif not isinstance(image_input, Image.Image):
            image_input = Image.fromarray(image_input).convert("RGB")
        inputs = hf_processor(images=image_input, return_tensors="pt")
        with torch.no_grad():
            outputs = hf_classifier(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        topk = torch.topk(probs, min(top_k, probs.shape[-1]), dim=-1)
        results = []
        for i in range(topk.indices.shape[1]):
            idx = topk.indices[0][i].item()
            score = topk.values[0][i].item()
            label = hf_classifier.config.id2label[idx].lower().replace("-", "_").replace(" ", "_")
            results.append({"label": label, "confidence": round(score, 4)})
        return results
    except Exception as e:
        print(f"[NutriSnap] HF classification error: {e}")
        return []


def draw_annotations(image_np, detections):
    """Draw bounding boxes and labels on the image."""
    annotated = image_np.copy()
    h_img, w_img = annotated.shape[:2]
    colors = [(44, 122, 74), (76, 175, 80), (129, 199, 132), (255, 165, 0),
              (0, 255, 255), (255, 0, 0), (128, 255, 0), (0, 128, 255)]
    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = det["bbox"]
        # Clamp coordinates to image bounds so drawing never goes off-screen
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        # Build label with count information
        food_name = det['food'].replace('_', ' ').title()
        count = det.get('count', 1)
        count_desc = det.get('count_description', '')
        calories = det.get('calories', '?')
        
        # Format label: "Food x{count} ({cal} cal)" or "Food ({desc}) ({cal} cal)"
        if count > 1 and count_desc:
            label = f"{food_name} x{count} ({calories} cal)"
        elif count_desc and count_desc not in ['1 serving', '1 slice', '1 wing', '1 donut', '1 taco', '1 egg', '1 pancake', '1 waffle']:
            label = f"{food_name} ({count_desc}) ({calories} cal)"
        else:
            label = f"{food_name} ({calories} cal)"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th_text), _ = cv2.getTextSize(label, font, 0.6, 2)
        label_height = th_text + 12
        # If there's room above the box, draw label above; otherwise draw inside
        if y1 - label_height >= 0:
            label_y_top = y1
            label_y_bottom = y1 - label_height
        else:
            label_y_top = y1 + label_height
            label_y_bottom = y1
        cv2.rectangle(annotated, (x1, label_y_bottom), (x1 + tw + 6, label_y_top), color, -1)
        cv2.putText(annotated, label, (x1 + 3, label_y_top - 6), font, 0.6, (255, 255, 255), 2)
    return annotated


def analyze_image(image_path, status_callback=None):
    """
    Multi-stage food detection pipeline:
      Stage 1: YOLO detection (WHERE food is)
      Stage 2: HF classifier on each YOLO crop (WHAT it is)
      Stage 3: Sub-region scan for missed items
      Stage 4: Nutrition calculation & logging
    Returns: (annotated_image, summary_text, results_list, thumbnails)
    """
    if status_callback:
        status_callback("📸 Loading image...")

    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_shape = img_np.shape[:2]  # (h, w)

    if status_callback:
        status_callback("🔍 Stage 1: YOLO detection...")

    # ── Stage 1: YOLO detection ──────────────────────────────────────────────
    yolo_detections = detect_with_yolo(img_bgr)
    if status_callback:
        status_callback(f"  YOLO found {len(yolo_detections)} item(s)")

    # ── Stage 2: Refine each YOLO box with HF classifier ─────────────────────
    detections = []
    if status_callback:
        status_callback("🔍 Stage 2: Refining with AI classifier...")

    for det in yolo_detections:
        x1, y1, x2, y2 = det["bbox"]
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_np.shape[1], x2), min(img_np.shape[0], y2)
        coco_name = det["food"]

        if x2 > x1 and y2 > y1:
            crop_pil = img_pil.crop((x1, y1, x2, y2))
            hf_results = classify_with_hf(crop_pil, top_k=3)
            resolved = None
            for hf_r in hf_results:
                mapped = _resolve_food_name(hf_r["label"])
                if mapped and mapped in NUTRITION_DB:
                    resolved = mapped
                    break
            food_name = resolved if resolved else coco_name
        else:
            food_name = coco_name

        det["food"] = food_name
        detections.append(det)

    # ── Stage 3: Sub-region scan for missed items ────────────────────────────
    if len(detections) < 3:
        if status_callback:
            status_callback("🔍 Stage 3: Scanning for missed items...")
        existing_bboxes = [d["bbox"] for d in detections]
        # Try 2x2 grid first, then 3x3 if still few detections
        for grid in [(2, 2), (3, 3)]:
            uncovered = get_uncovered_regions(img_shape, existing_bboxes, grid=grid)
            for region in uncovered:
                x1, y1, x2, y2 = region
                crop_pil = img_pil.crop((x1, y1, x2, y2))
                hf_results = classify_with_hf(crop_pil, top_k=3)
                for hf_r in hf_results:
                    if hf_r["confidence"] < 0.3:
                        continue
                    mapped = _resolve_food_name(hf_r["label"])
                    if not mapped or mapped not in NUTRITION_DB:
                        continue
                    # Avoid duplicate foods
                    existing_foods = [d["food"] for d in detections]
                    if mapped in existing_foods:
                        continue
                    detections.append({
                        "food": mapped,
                        "bbox": region,
                        "confidence": round(hf_r["confidence"], 2),
                        "source": "sub_region",
                    })
                    existing_bboxes.append(region)
                    if status_callback:
                        status_callback(f"  ➕ Detected: {mapped.replace('_', ' ').title()} (sub-region)")
                    break  # one food per region
            if len(detections) >= 3:
                break

    if not detections:
        # Last resort: run HF on full image
        if status_callback:
            status_callback("🔍 Fallback: classifying full image...")
        hf_results = classify_with_hf(img_pil, top_k=3)
        for hf_r in hf_results:
            mapped = _resolve_food_name(hf_r["label"])
            if mapped and mapped in NUTRITION_DB:
                h, w = img_shape
                detections.append({
                    "food": mapped,
                    "bbox": [int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)],
                    "confidence": round(hf_r["confidence"], 2),
                    "source": "full_image_hf",
                })
                break

    if not detections:
        return None, "No food items detected. Try a clearer photo of a meal.", None, []

    # ── Stage 4: Calculate nutrition & portions ─────────────────────────────
    if status_callback:
        status_callback("🧮 Calculating nutrition...")
    results = []
    status_messages = []

    for det in detections:
        food = det["food"]
        food_key = food.lower().replace(" ", "_")
        
        # Estimate item count based on bbox size
        count, count_description = estimate_item_count(food, det["bbox"], img_shape)
        det["count"] = count
        det["count_description"] = count_description
        
        # Calculate grams: use unit weight for countable items, otherwise portion estimation
        if food_key in UNIT_WEIGHTS:
            grams = count * UNIT_WEIGHTS[food_key]
            portion_label = count_description
        else:
            portion_label, portion_mult, fries_grams = estimate_portion(det["bbox"], img_shape, food_name=food)
            if fries_grams is not None:
                grams = fries_grams
            else:
                typical_g = NUTRITION_DB.get(food, {}).get("typical_g", 150)
                grams = round(typical_g * portion_mult)

        nutr = calculate_nutrition(food, grams, status_callback=status_messages.append)
        if nutr:
            det.update(nutr)
        else:
            # Ensure item still gets annotated even if nutrition lookup fails
            det.update({"calories": 0, "protein": 0, "carbs": 0, "fat": 0})
        det["portion"] = portion_label
        det["grams"] = grams
        results.append(det)

    # ── Stage 4b: Deduplicate results ──────────────────────────────────────
    results = deduplicate_results(results)
    
    # Calculate totals after deduplication
    total_cal, total_pro, total_carb, total_fat = 0, 0, 0, 0
    for r in results:
        total_cal += r.get("calories", 0)
        total_pro += r.get("protein", 0)
        total_carb += r.get("carbs", 0)
        total_fat += r.get("fat", 0)

    if status_callback:
        for msg in status_messages:
            status_callback(msg)
        status_callback(f"✅ Analysis complete! Found {len(results)} item(s)")

    # ── Stage 5: Annotate image ─────────────────────────────────────────────
    annotated = draw_annotations(img_bgr, results)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # ── Stage 5b: Crop thumbnails ──────────────────────────────────────────
    cropped_thumbnails = []
    for det in results:
        x1, y1, x2, y2 = det["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_np.shape[1], x2), min(img_np.shape[0], y2)
        if x2 > x1 and y2 > y1:
            # Add 30% padding around bounding box for natural-looking thumbnails
            pad_x = int((x2 - x1) * 0.3)
            pad_y = int((y2 - y1) * 0.3)
            crop_y1 = max(0, y1 - pad_y)
            crop_y2 = min(img_np.shape[0], y2 + pad_y)
            crop_x1 = max(0, x1 - pad_x)
            crop_x2 = min(img_np.shape[1], x2 + pad_x)
            crop = img_np[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            crop_resized = cv2.resize(crop, (200, 200), interpolation=cv2.INTER_CUBIC)
            label = det["food"].replace("_", " ").title()
            cropped_thumbnails.append((crop_resized, label))

    # ── Stage 6: Build summary markdown ─────────────────────────────────────
    summary_lines = [f"### Detected {len(results)} food item(s)\n"]
    summary_lines.append("| Food | Quantity | Portion (g) | Calories | Protein | Carbs | Fat |")
    summary_lines.append("|------|----------|-------------|----------|---------|-------|-----|")
    for r in results:
        name = r["food"].replace("_", " ").title()
        count = r.get('count', 1)
        count_desc = r.get('count_description', r.get('portion', '1 serving'))
        # Use count_description for quantity column if available
        quantity = count_desc if count_desc else f"{count} item(s)"
        summary_lines.append(
            f"| {name} | {quantity} | {r['grams']}g "
            f"| {r['calories']} | {r['protein']}g | {r['carbs']}g | {r['fat']}g |"
        )
    summary_lines.append(f"\n**Totals: {round(total_cal, 1)} cal | "
                         f"{round(total_pro, 1)}g protein | "
                         f"{round(total_carb, 1)}g carbs | "
                         f"{round(total_fat, 1)}g fat**")

    if status_messages:
        summary_lines.append("\n---\n**Lookup Status:**")
        for msg in status_messages:
            summary_lines.append(f"  \n{msg}")

    return annotated_rgb, "\n".join(summary_lines), results, cropped_thumbnails


# ============================================
# DASHBOARD (MODIFIED)
# ============================================

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

    # 1. Daily calorie intake
    daily = df.groupby("Date")["Calories"].sum().reset_index()
    fig_daily = px.bar(daily, x="Date", y="Calories",
                       title="Daily Calorie Intake",
                       color_discrete_sequence=[THEME_PRIMARY])
    fig_daily.update_layout(template="plotly_white", height=380, margin=dict(b=60, t=50))
    fig_daily.update_xaxes(tickformat="%Y-%m-%d", tickangle=-30)

    # 2. Macro distribution (pie chart)
    macro_totals = {
        "Protein": df["Protein (g)"].sum(),
        "Carbs": df["Carbs (g)"].sum(),
        "Fat": df["Fat (g)"].sum(),
    }
    fig_macro = px.pie(names=list(macro_totals.keys()), values=list(macro_totals.values()),
                       title="Macronutrient Distribution",
                       color_discrete_sequence=[THEME_PRIMARY, THEME_ACCENT, "#81C784"])
    fig_macro.update_layout(template="plotly_white", height=380, margin=dict(b=40, t=50))

    # 3. Weekly calorie trend
    df["DateObj"] = pd.to_datetime(df["Date"], errors="coerce")
    weekly = df.dropna(subset=["DateObj"]).set_index("DateObj").resample("W")["Calories"].sum().reset_index()
    weekly.columns = ["Week", "Calories"]
    fig_weekly = px.line(weekly, x="Week", y="Calories",
                         title="Weekly Calorie Trend",
                         color_discrete_sequence=[THEME_PRIMARY])
    fig_weekly.update_layout(template="plotly_white", height=380, margin=dict(b=60, t=50))
    fig_weekly.update_xaxes(tickformat="%Y-%m-%d", tickangle=-30)

    # 4. Top foods eaten
    top = df["Food"].value_counts().head(5).reset_index()
    top.columns = ["Food", "Count"]
    fig_top = px.bar(top, y="Food", x="Count", orientation="h",
                     title="Top 5 Foods Eaten",
                     color_discrete_sequence=[THEME_ACCENT])
    fig_top.update_layout(template="plotly_white", height=380, yaxis={"categoryorder": "total ascending"}, margin=dict(l=120))

    # Summary cards
    total_cal = int(round(df["Calories"].sum()))
    total_meals = len(df)
    avg_cal = int(round(df["Calories"].mean()))
    stats = (f"<div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;'>"
             f"<div style='background:{THEME_PRIMARY};color:white;padding:14px 22px;border-radius:10px;flex:1;min-width:160px;text-align:center;'>"
             f"<div style='font-size:1.8em;font-weight:700;'>{total_meals}</div><div style='opacity:0.85;font-size:0.9em;'>📊 Total Meals</div></div>"
             f"<div style='background:{THEME_ACCENT};color:white;padding:14px 22px;border-radius:10px;flex:1;min-width:160px;text-align:center;'>"
             f"<div style='font-size:1.8em;font-weight:700;'>{total_cal:,}</div><div style='opacity:0.85;font-size:0.9em;'>🔥 Total Calories</div></div>"
             f"<div style='background:{THEME_SECONDARY};color:white;padding:14px 22px;border-radius:10px;flex:1;min-width:160px;text-align:center;'>"
             f"<div style='font-size:1.8em;font-weight:700;'>{avg_cal} cal</div><div style='opacity:0.85;font-size:0.9em;'>📈 Avg per Meal</div></div>"
             f"</div>")

    return fig_daily, fig_macro, fig_weekly, fig_top, stats


# ============================================
# GRADIO UI (MODIFIED)
# ============================================

CSS = """
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}
.main {
    max-width: 100% !important;
    width: 100% !important;
}
.contain {
    max-width: 100% !important;
}
.tabitem, .tab-nav + div {
    padding: 0 24px !important;
}
.header {
    background: linear-gradient(135deg, #2C7A4A, #1A5A3A);
    padding: 24px 32px;
    border-radius: 0;
    margin-bottom: 0;
    text-align: center;
}
.header h1 {
    color: white;
    margin: 0;
    font-size: 2em;
    font-weight: 700;
}
.header p {
    color: rgba(255,255,255,0.8);
    margin: 6px 0 0;
    font-size: 1em;
}
.card {
    background: #ffffff;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
footer { display: none; }
.status-box {
    padding: 10px 14px;
    border-radius: 8px;
    background: #f0f7f4;
    border-left: 4px solid #2C7A4A;
    font-size: 0.95em;
    min-height: 24px;
}
.settings-section {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.settings-section h3 {
    margin-top: 0;
    color: #2C7A4A;
    border-bottom: 2px solid #E8F5E9;
    padding-bottom: 8px;
}

/* Live dark mode toggle styles */
body.dark-mode,
body.dark-mode .gradio-container {
    background-color: #1a1a2e !important;
    color: #e0e0e0 !important;
}
body.dark-mode .gr-block,
body.dark-mode .gr-form,
body.dark-mode .gr-box,
body.dark-mode .settings-section,
body.dark-mode .card {
    background-color: #16213e !important;
    color: #e0e0e0 !important;
}
body.dark-mode .status-box {
    background: #1a2e1a !important;
    color: #c0e0c0 !important;
}
body.dark-mode .gr-input,
body.dark-mode .gr-text-input,
body.dark-mode .gr-textbox textarea {
    background-color: #0f3460 !important;
    color: #e0e0e0 !important;
    border-color: #2C7A4A !important;
}
body.dark-mode .gr-button {
    border-color: #2C7A4A !important;
}
body.dark-mode .gr-markdown {
    color: #e0e0e0 !important;
}
body.dark-mode table {
    color: #e0e0e0 !important;
}

/* Loading spinner */
.analysis-spinner {
    display: none;
    text-align: center;
    padding: 24px;
    margin: 12px 0;
    background: linear-gradient(135deg, #f0f7f4, #e8f5e9);
    border-radius: 12px;
    border: 2px solid #2C7A4A;
}
.analysis-spinner.active {
    display: block;
}
.analysis-spinner .spinner-icon {
    font-size: 2em;
    animation: spin-pulse 1.2s ease-in-out infinite;
}
.analysis-spinner .spinner-text {
    font-size: 1.1em;
    color: #2C7A4A;
    font-weight: 600;
    margin-top: 8px;
}
@keyframes spin-pulse {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.2); opacity: 0.6; }
    100% { transform: scale(1); opacity: 1; }
}

/* Food thumbnail gallery */
.food-gallery-section {
    background: #f8faf9;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #e0ece6;
}
.food-gallery-section h3 {
    color: #2C7A4A;
    margin-top: 0;
    margin-bottom: 12px;
}
.results-section {
    background: #ffffff;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
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

HEADER_HTML = """
<div class="header">
    <h1>🍔 NutriSnap AI</h1>
    <p>Snap. Identify. Track. Eat Better.</p>
</div>
"""


def generate_meal_report(detections):
    """Generate a nicely formatted HTML meal report for PDF printing."""
    if not detections:
        return None
    now = datetime.now()
    rows_html = ""
    total_cal, total_pro, total_carb, total_fat = 0, 0, 0, 0
    for det in detections:
        name = det.get("food", "").replace("_", " ").title()
        count = det.get("count", 1)
        count_desc = det.get("count_description", det.get("portion", "N/A"))
        quantity = count_desc if count_desc else f"{count} item(s)"
        grams = det.get("grams", 0)
        cal = det.get("calories", 0)
        pro = det.get("protein", 0)
        carb = det.get("carbs", 0)
        fat = det.get("fat", 0)
        total_cal += cal
        total_pro += pro
        total_carb += carb
        total_fat += fat
        rows_html += (
            f"<tr><td>{name}</td><td>{quantity}</td><td>{grams}g</td>"
            f"<td>{cal}</td><td>{pro}g</td><td>{carb}g</td><td>{fat}g</td></tr>\n"
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NutriSnap AI - Meal Report</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 700px; margin: 40px auto; padding: 20px; color: #333; }}
  .report-header {{ background: linear-gradient(135deg, #2C7A4A, #1A5A3A); color: white; padding: 24px; border-radius: 12px; text-align: center; margin-bottom: 24px; }}
  .report-header h1 {{ margin: 0; font-size: 1.6em; }}
  .report-header p {{ margin: 4px 0 0; opacity: 0.85; font-size: 0.9em; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
  th {{ background: #2C7A4A; color: white; padding: 10px 12px; text-align: left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #e0e0e0; }}
  tr:nth-child(even) {{ background: #f8faf9; }}
  .totals {{ background: #e8f5e9; padding: 16px; border-radius: 8px; margin-top: 16px; font-weight: 600; }}
  .totals span {{ display: inline-block; margin-right: 20px; }}
  .footer {{ margin-top: 24px; text-align: center; font-size: 0.85em; color: #888; }}
  @media print {{ body {{ margin: 0; }} }}
</style>
</head>
<body>
<div class="report-header">
  <h1>NutriSnap AI - Meal Report</h1>
  <p>Generated on {now.strftime("%Y-%m-%d %H:%M:%S")}</p>
</div>
<h2>Detected Foods ({len(detections)} item(s))</h2>
<table>
<tr><th>Food</th><th>Quantity</th><th>Portion (g)</th><th>Calories</th><th>Protein</th><th>Carbs</th><th>Fat</th></tr>
{rows_html}
</table>
<div class="totals">
  <strong>Totals:</strong>
  <span>{round(total_cal, 1)} cal</span>
  <span>{round(total_pro, 1)}g protein</span>
  <span>{round(total_carb, 1)}g carbs</span>
  <span>{round(total_fat, 1)}g fat</span>
</div>
<div class="footer">Generated by NutriSnap AI</div>
</body>
</html>"""
    report_path = "meal_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


def build_ui():
    """Build the complete Gradio interface."""
    config = load_config()

    # JavaScript for live dark mode toggle and spinner control
    JS_HEAD = """
    <script>
    // Apply saved dark mode preference on load
    (function() {
        const isDark = """ + str(config.get("dark_mode", False)).lower() + """;
        if (isDark) document.body.classList.add('dark-mode');
    })();

    function toggleDarkMode(enabled) {
        if (enabled) {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
    }

    function showSpinner() {
        var el = document.getElementById('analysis-spinner');
        if (el) { el.classList.add('active'); }
    }

    function hideSpinner() {
        var el = document.getElementById('analysis-spinner');
        if (el) { el.classList.remove('active'); }
    }
    </script>
    """

    full_css = CSS

    with gr.Blocks(css=full_css, title="NutriSnap AI", theme=gr.themes.Soft(), head=JS_HEAD) as demo:
        # Header
        gr.HTML(HEADER_HTML)

        with gr.Tabs():
            # ---- Tab 1: Upload & Analyze ----
            with gr.TabItem("📸 Upload & Analyze"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📷 Upload Your Meal Photo")
                        input_image = gr.File(label="Upload Meal Photo", file_types=["image"],
                                               type="filepath")
                        analyze_btn = gr.Button("🔍 Analyze Food", variant="primary", size="lg")
                        status_display = gr.Markdown(value="", elem_classes=["status-box"])
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🖼️ Annotated Result")
                        output_image = gr.Image(label="Annotated Result", type="numpy")

                # Loading spinner (hidden by default, shown via JS)
                gr.HTML("""
                <div id="analysis-spinner" class="analysis-spinner">
                    <div class="spinner-icon">🔍</div>
                    <div class="spinner-text">Analyzing your meal...</div>
                </div>
                """)

                # Detected food items gallery
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🍽️ Detected Food Items")
                        food_gallery = gr.Gallery(
                            label="Detected Food Items",
                            columns=4,
                            rows=1,
                            height=240,
                            object_fit="cover",
                            show_label=False,
                        )

                # Results table
                with gr.Row():
                    with gr.Column(elem_classes=["results-section"]):
                        output_md = gr.Markdown(
                            value="Upload a photo and click **Analyze Food** to see results."
                        )

                # Save Meal button (below results)
                with gr.Row():
                    save_meal_btn = gr.Button("💾 Save Meal", variant="secondary")
                    save_meal_status = gr.Markdown("")

                # Manual calorie entry section
                with gr.Row():
                    gr.Markdown("---")
                with gr.Row():
                    gr.Markdown("#### ✏️ Manual Entry (if detection fails)")
                with gr.Row():
                    manual_food = gr.Textbox(label="Food name", placeholder="e.g. Chicken breast")
                    manual_cal = gr.Number(label="Calories", value=0)
                    manual_protein = gr.Number(label="Protein (g)", value=0)
                    manual_carbs = gr.Number(label="Carbs (g)", value=0)
                    manual_fat = gr.Number(label="Fat (g)", value=0)
                with gr.Row():
                    manual_log_btn = gr.Button("📝 Log Manual Entry", variant="secondary")
                    manual_status = gr.Markdown("")

            # ---- Tab 2: Dashboard ----
            with gr.TabItem("📊 Dashboard"):
                dash_stats = gr.HTML()
                with gr.Row():
                    chart_daily = gr.Plot(label="Daily Calories")
                    chart_macro = gr.Plot(label="Macro Distribution")
                with gr.Row():
                    chart_weekly = gr.Plot(label="Weekly Trend")
                    chart_top = gr.Plot(label="Top 5 Foods")
                refresh_btn = gr.Button("🔄 Refresh Dashboard")

            # ---- Tab 3: Food Log ----
            with gr.TabItem("📋 Food Log"):
                log_table = gr.Dataframe(headers=CSV_COLUMNS, label="Meal History", interactive=False)
                log_refresh = gr.Button("🔄 Refresh Log")
                gr.Markdown("---")
                gr.Markdown("#### 🗑️ Manage Entries")
                with gr.Row():
                    delete_row_input = gr.Number(label="Row index to delete (0-based)", value=0, precision=0)
                    delete_entry_btn = gr.Button("🗑️ Delete Entry", variant="secondary")
                with gr.Row():
                    clear_all_btn = gr.Button("🗑️ Clear All Log", variant="stop")
                    log_delete_status = gr.Markdown("")

            # ---- Tab 4: Tips ----
            with gr.TabItem("💡 Nutrition Tips"):
                gr.Markdown(TIPS_MD)

            # ---- Tab 5: Settings ----
            with gr.TabItem("⚙️ Settings"):
                # USDA API Key section
                with gr.Group():
                    gr.Markdown("### 🔑 USDA FoodData Central API")
                    gr.Markdown("Get a free API key at [data.nal.usda.gov](https://fdc.nal.usda.gov/api-key-signup.html)")
                    with gr.Row():
                        usda_key_input = gr.Textbox(
                            label="USDA API Key",
                            type="password",
                            value=config.get("usda_api_key", ""),
                            placeholder="Enter your USDA API key..."
                        )
                    with gr.Row():
                        test_conn_btn = gr.Button("🔌 Test Connection")
                        save_key_btn = gr.Button("💾 Save Key")
                    conn_status = gr.Markdown("")

                # Cache management section
                with gr.Group():
                    gr.Markdown("### 🗃️ Cache Management")
                    cache_info = gr.Markdown(f"Cached items: **{get_cache_size()}**")
                    clear_cache_btn = gr.Button("🗑️ Clear Cache")
                    cache_status = gr.Markdown("")

                # Export data section
                with gr.Group():
                    gr.Markdown("### 📤 Export Data")
                    with gr.Row():
                        export_csv_btn = gr.Button("📊 Export CSV")
                        export_pdf_btn = gr.Button("📄 Export PDF Report")
                    export_status = gr.Markdown("")
                    export_file = gr.File(label="Download", visible=False)

                # Dark mode toggle
                with gr.Group():
                    gr.Markdown("### 🎨 Appearance")
                    dark_mode_toggle = gr.Checkbox(
                        label="Enable Dark Mode",
                        value=config.get("dark_mode", False)
                    )
                    dark_mode_status = gr.Markdown("")

        # ---- Event Handlers ----
        def on_analyze(file):
            global _last_analysis_results
            if file is None:
                return None, "Please upload an image first.", "", []
            status_msgs = []
            def collect_status(msg):
                status_msgs.append(msg)
            annotated, summary, detections, thumbnails = analyze_image(file, status_callback=collect_status)
            status_text = "\n".join(status_msgs) if status_msgs else ""
            # Store detections for Save Meal button
            _last_analysis_results = detections if detections else None
            return annotated, summary, status_text, thumbnails

        def on_analyze_click():
            """Show spinner when analyze button is clicked."""
            return gr.update(visible=True)

        def on_manual_log(food_name, cal, protein, carbs, fat):
            if not food_name or not food_name.strip():
                return "⚠️ Please enter a food name."
            food_name = food_name.strip()
            log_meal(food_name, cal, protein, carbs, fat, "Manual")
            return f"✅ Logged: {food_name} - {cal} cal"

        def on_test_connection(api_key):
            return test_usda_connection(api_key)

        def on_save_key(api_key):
            cfg = load_config()
            cfg["usda_api_key"] = api_key.strip() if api_key else ""
            if save_config(cfg):
                return "💾 API key saved successfully."
            return "❌ Failed to save API key."

        def on_clear_cache():
            result = clear_cache()
            return result, f"Cached items: **{get_cache_size()}**"

        def on_export_csv():
            path = export_csv_file()
            if path:
                return "✅ CSV file ready for download.", gr.update(value=path, visible=True)
            return "⚠️ No meal log data to export.", gr.update(visible=False)

        def on_export_pdf():
            """Generate an HTML meal report that can be printed as PDF."""
            df = read_log()
            if df.empty:
                return "⚠️ No meal log data to export.", gr.update(visible=False)
            # Build detections from recent log entries for the report
            recent = df.tail(10)
            mock_detections = []
            for _, row in recent.iterrows():
                mock_detections.append({
                    "food": str(row.get("Food", "unknown")),
                    "portion": str(row.get("Portion", "N/A")),
                    "grams": 0,
                    "calories": float(row.get("Calories", 0)),
                    "protein": float(row.get("Protein (g)", 0)),
                    "carbs": float(row.get("Carbs (g)", 0)),
                    "fat": float(row.get("Fat (g)", 0)),
                })
            report_path = generate_meal_report(mock_detections)
            if report_path:
                return (f"✅ Meal report generated! Open `{report_path}` in your browser and use **Print → Save as PDF**.",
                        gr.update(value=report_path, visible=True))
            return "⚠️ Failed to generate report.", gr.update(visible=False)

        def on_dark_mode_change(enabled):
            """Toggle dark mode live via JS + save config."""
            cfg = load_config()
            cfg["dark_mode"] = enabled
            save_config(cfg)
            if enabled:
                return "🌙 Dark mode enabled."
            return "☀️ Light mode enabled."

        def on_refresh_dashboard():
            return build_dashboard()

        def on_refresh_log():
            return read_log()

        def on_save_meal():
            global _last_analysis_results
            if _last_analysis_results is None:
                return "⚠️ No meal to save. Analyze a photo first."
            for det in _last_analysis_results:
                food = det.get("food", "").replace("_", " ").title()
                log_meal(
                    food,
                    det.get("calories", 0),
                    det.get("protein", 0),
                    det.get("carbs", 0),
                    det.get("fat", 0),
                    det.get("portion", ""),
                )
            _last_analysis_results = None
            return "✅ Meal saved!"

        def on_delete_entry(row_idx):
            row_idx = int(row_idx)
            if delete_log_entry(row_idx):
                return "✅ Entry deleted", read_log()
            return "⚠️ Invalid row index or empty log.", read_log()

        def on_clear_all():
            clear_all_log()
            return "✅ All log entries cleared.", read_log()

        # Wire up events
        analyze_btn.click(
            fn=on_analyze, inputs=[input_image],
            outputs=[output_image, output_md, status_display, food_gallery],
            js="(file) => { var el = document.getElementById('analysis-spinner'); if (el) el.classList.add('active'); return file; }"
        )
        # Hide spinner when analysis completes (via output change)
        output_image.change(
            fn=None, inputs=[output_image], outputs=None,
            js="(img) => { var el = document.getElementById('analysis-spinner'); if (el) el.classList.remove('active'); return img; }"
        )

        manual_log_btn.click(fn=on_manual_log,
                             inputs=[manual_food, manual_cal, manual_protein, manual_carbs, manual_fat],
                             outputs=[manual_status])
        save_meal_btn.click(fn=on_save_meal, outputs=[save_meal_status])
        refresh_btn.click(fn=on_refresh_dashboard,
                          outputs=[chart_daily, chart_macro, chart_weekly, chart_top, dash_stats])
        log_refresh.click(fn=on_refresh_log, outputs=[log_table])
        delete_entry_btn.click(fn=on_delete_entry,
                               inputs=[delete_row_input],
                               outputs=[log_delete_status, log_table])
        clear_all_btn.click(fn=on_clear_all,
                            outputs=[log_delete_status, log_table])
        test_conn_btn.click(fn=on_test_connection, inputs=[usda_key_input], outputs=[conn_status])
        save_key_btn.click(fn=on_save_key, inputs=[usda_key_input], outputs=[conn_status])
        clear_cache_btn.click(fn=on_clear_cache, outputs=[cache_status, cache_info])
        export_csv_btn.click(fn=on_export_csv, outputs=[export_status, export_file])
        export_pdf_btn.click(fn=on_export_pdf, outputs=[export_status, export_file])
        dark_mode_toggle.change(
            fn=on_dark_mode_change,
            inputs=[dark_mode_toggle],
            outputs=[dark_mode_status],
            js="(function(val) { if (val) { document.body.classList.add('dark-mode'); } else { document.body.classList.remove('dark-mode'); } })"
        )

        # Load data on startup
        demo.load(fn=on_refresh_dashboard,
                 outputs=[chart_daily, chart_macro, chart_weekly, chart_top, dash_stats])
        demo.load(fn=on_refresh_log, outputs=[log_table])

    return demo


# ============================================
# FALLBACK HTTP SERVER
# ============================================

def start_fallback_server(port=7860):
    """Start a lightweight HTTP server serving the fallback HTML UI + REST API."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse
    import io
    import tempfile

    HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fallback_ui.html")

    def parse_multipart(body, boundary):
        """Simple multipart/form-data parser (replaces deprecated cgi.FieldStorage)."""
        parts = {}
        boundary_bytes = boundary.encode("utf-8") if isinstance(boundary, str) else boundary
        delimiter = b"--" + boundary_bytes
        sections = body.split(delimiter)
        for section in sections[1:]:  # skip preamble
            if section.startswith(b"--"):
                break  # end marker
            # Split headers from content at double newline
            header_end = section.find(b"\r\n\r\n")
            if header_end == -1:
                header_end = section.find(b"\n\n")
                if header_end == -1:
                    continue
                content = section[header_end + 2:]
            else:
                content = section[header_end + 4:]
            # Strip trailing \r\n
            if content.endswith(b"\r\n"):
                content = content[:-2]
            headers_raw = section[:header_end].decode("utf-8", errors="replace")
            # Extract name and filename from Content-Disposition
            name = None
            filename = None
            for line in headers_raw.split("\n"):
                if "content-disposition" in line.lower():
                    for part in line.split(";"):
                        part = part.strip()
                        if part.startswith("name="):
                            name = part.split("=", 1)[1].strip('"').strip("'")
                        elif part.startswith("filename="):
                            filename = part.split("=", 1)[1].strip('"').strip("'")
            if name:
                parts[name] = {"data": content, "filename": filename}
        return parts

    class FallbackHandler(BaseHTTPRequestHandler):
        """HTTP handler for fallback UI + API endpoints."""

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _json_response(self, data, status=200):
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self):
            length = int(self.headers.get("Content-Length", 0))
            if length:
                return json.loads(self.rfile.read(length))
            return {}

        def _error(self, msg, status=400):
            self._json_response({"error": msg}, status)

        def log_message(self, fmt, *args):
            print(f"[FallbackServer] {fmt % args}")

        # ── OPTIONS (CORS preflight) ──
        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        # ── GET ──
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")

            if path == "" or path == "/":
                self._serve_html()
            elif path == "/api/log":
                self._handle_get_log()
            elif path == "/api/dashboard":
                self._handle_get_dashboard()
            elif path == "/api/settings":
                self._handle_get_settings()
            elif path == "/api/export/csv":
                self._handle_export_csv()
            else:
                self._error("Not found", 404)

        # ── POST ──
        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")

            if path == "/api/analyze":
                self._handle_analyze()
            elif path == "/api/log/manual":
                self._handle_manual_log()
            elif path == "/api/settings":
                self._handle_save_settings()
            elif path == "/api/settings/test":
                self._handle_test_api_key()
            elif path == "/api/cache/clear":
                self._handle_clear_cache()
            else:
                self._error("Not found", 404)

        # ── HTML ──
        def _serve_html(self):
            try:
                with open(HTML_PATH, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self._cors()
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self._error("fallback_ui.html not found", 500)

        # ── API: Analyze Image ──
        def _handle_analyze(self):
            try:
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    self._error("Expected multipart/form-data")
                    return

                # Extract boundary from Content-Type
                boundary = None
                for part in content_type.split(";"):
                    part = part.strip()
                    if part.startswith("boundary="):
                        boundary = part.split("=", 1)[1].strip('"')
                if not boundary:
                    self._error("No boundary in Content-Type")
                    return

                # Parse multipart using manual parser
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                parts = parse_multipart(body, boundary)

                image_field = parts.get("image")
                if image_field is None:
                    self._error("No image file uploaded")
                    return

                # Save to temp file
                filename = image_field.get("filename") or ".jpg"
                suffix = os.path.splitext(filename)[1] or ".jpg"
                tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                tmp.write(image_field["data"])
                tmp.close()

                try:
                    status_msgs = []
                    def collect_status(msg):
                        status_msgs.append(msg)

                    annotated, summary, detections, thumbnails = analyze_image(
                        tmp.name, status_callback=collect_status
                    )

                    if detections is None:
                        self._json_response({
                            "items": [],
                            "totals": {"calories": 0, "protein": 0, "carbs": 0, "fat": 0},
                            "messages": status_msgs + ["No food items detected."],
                        })
                        return

                    items = []
                    totals = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
                    for det in detections:
                        item = {
                            "food": det.get("food", "").replace("_", " ").title(),
                            "portion": det.get("portion", ""),
                            "grams": det.get("grams", 0),
                            "calories": det.get("calories", 0),
                            "protein": det.get("protein", 0),
                            "carbs": det.get("carbs", 0),
                            "fat": det.get("fat", 0),
                            "confidence": det.get("confidence", 0),
                        }
                        items.append(item)
                        totals["calories"] += item["calories"]
                        totals["protein"] += item["protein"]
                        totals["carbs"] += item["carbs"]
                        totals["fat"] += item["fat"]

                    totals = {k: round(v, 1) for k, v in totals.items()}
                    self._json_response({
                        "items": items, "totals": totals, "messages": status_msgs
                    })
                finally:
                    os.unlink(tmp.name)

            except Exception as e:
                print(f"[FallbackServer] Analyze error: {e}")
                self._error(str(e), 500)

        # ── API: Get Log ──
        def _handle_get_log(self):
            try:
                df = read_log()
                entries = df.values.tolist()
                self._json_response({"entries": entries, "columns": CSV_COLUMNS})
            except Exception as e:
                self._error(str(e), 500)

        # ── API: Dashboard ──
        def _handle_get_dashboard(self):
            try:
                df = read_log()
                if df.empty:
                    self._json_response({
                        "stats": None, "daily": None, "macros": None,
                        "weekly": None, "top_foods": None
                    })
                    return

                df["Calories"] = pd.to_numeric(df["Calories"], errors="coerce").fillna(0)
                df["Protein (g)"] = pd.to_numeric(df["Protein (g)"], errors="coerce").fillna(0)
                df["Carbs (g)"] = pd.to_numeric(df["Carbs (g)"], errors="coerce").fillna(0)
                df["Fat (g)"] = pd.to_numeric(df["Fat (g)"], errors="coerce").fillna(0)

                total_cal = round(df["Calories"].sum(), 1)
                total_meals = len(df)
                avg_cal = round(df["Calories"].mean(), 1)

                daily = df.groupby("Date")["Calories"].sum().reset_index()
                macro_totals = {
                    "Protein": round(df["Protein (g)"].sum(), 1),
                    "Carbs": round(df["Carbs (g)"].sum(), 1),
                    "Fat": round(df["Fat (g)"].sum(), 1),
                }
                df["DateObj"] = pd.to_datetime(df["Date"], errors="coerce")
                weekly = df.dropna(subset=["DateObj"]).set_index("DateObj").resample("W")["Calories"].sum().reset_index()
                weekly.columns = ["Week", "Calories"]
                top = df["Food"].value_counts().head(5).reset_index()
                top.columns = ["Food", "Count"]

                self._json_response({
                    "stats": {"total_meals": total_meals, "total_calories": total_cal, "avg_per_meal": avg_cal},
                    "daily": {"labels": daily["Date"].tolist(), "values": daily["Calories"].tolist()},
                    "macros": {"labels": list(macro_totals.keys()), "values": list(macro_totals.values())},
                    "weekly": {
                        "labels": [w.strftime("%Y-%m-%d") for w in weekly["Week"]],
                        "values": weekly["Calories"].tolist()
                    },
                    "top_foods": {"labels": top["Food"].tolist(), "values": top["Count"].tolist()},
                })
            except Exception as e:
                self._error(str(e), 500)

        # ── API: Manual Log ──
        def _handle_manual_log(self):
            try:
                data = self._read_json_body()
                food = data.get("food", "").strip()
                if not food:
                    self._error("Food name is required")
                    return
                calories = float(data.get("calories", 0))
                protein = float(data.get("protein", 0))
                carbs = float(data.get("carbs", 0))
                fat = float(data.get("fat", 0))
                log_meal(food, calories, protein, carbs, fat, "Manual")
                self._json_response({"success": True, "message": f"Logged: {food} - {calories} cal"})
            except Exception as e:
                self._error(str(e), 500)

        # ── API: Get Settings ──
        def _handle_get_settings(self):
            try:
                cfg = load_config()
                self._json_response({
                    "usda_api_key": cfg.get("usda_api_key", ""),
                    "dark_mode": cfg.get("dark_mode", False),
                    "cache_size": get_cache_size(),
                })
            except Exception as e:
                self._error(str(e), 500)

        # ── API: Save Settings ──
        def _handle_save_settings(self):
            try:
                data = self._read_json_body()
                cfg = load_config()
                if "usda_api_key" in data:
                    cfg["usda_api_key"] = data["usda_api_key"].strip()
                if "dark_mode" in data:
                    cfg["dark_mode"] = bool(data["dark_mode"])
                save_config(cfg)
                self._json_response({"success": True, "message": "Settings saved."})
            except Exception as e:
                self._error(str(e), 500)

        # ── API: Test API Key ──
        def _handle_test_api_key(self):
            try:
                data = self._read_json_body()
                api_key = data.get("api_key", "").strip()
                result = test_usda_connection(api_key)
                success = "successful" in result.lower()
                self._json_response({"success": success, "message": result})
            except Exception as e:
                self._error(str(e), 500)

        # ── API: Clear Cache ──
        def _handle_clear_cache(self):
            try:
                result = clear_cache()
                self._json_response({"success": True, "message": result})
            except Exception as e:
                self._error(str(e), 500)

        # ── API: Export CSV ──
        def _handle_export_csv(self):
            try:
                if not os.path.exists(CSV_FILE):
                    self._error("No meal log to export", 404)
                    return
                with open(CSV_FILE, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", f'attachment; filename="{CSV_FILE}"')
                self._cors()
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self._error(str(e), 500)

    # ── Launch server ──
    print(f"\n{'='*50}")
    print(f"🌐 Fallback UI running at http://localhost:{port}")
    print(f"{'='*50}\n")
    server = HTTPServer(("0.0.0.0", port), FallbackHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[FallbackServer] Shutting down...")
        server.shutdown()


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import sys as _sys

    print("[NutriSnap] Loading models...")
    load_yolo()
    load_hf_classifier()

    if "--fallback" in _sys.argv:
        print("[NutriSnap] --fallback flag detected. Starting HTML interface...")
        start_fallback_server()
    else:
        try:
            import gradio as gr  # noqa: F811
            print("[NutriSnap] Starting Gradio app...")
            demo = build_ui()
            demo.launch(share=True)
        except (ImportError, Exception) as e:
            print(f"⚠️  Gradio unavailable ({e}). Launching fallback HTML interface...")
            start_fallback_server()
