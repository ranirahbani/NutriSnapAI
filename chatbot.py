"""
PlateGenieAI Conversational AI Assistant
Dual-mode chatbot: Application Knowledge + Personal Nutrition Advisor
Uses Groq API (free tier) with Llama 3.1 model.
"""

import os
import re
import time
import pandas as pd
from pathlib import Path

# ============================================
# CONSTANTS
# ============================================

DISCLAIMER = (
    "\n\n---\n⚠️ *Disclaimer: This AI assistant provides general nutritional information "
    "based on your logged meal data. It is NOT a substitute for professional medical or "
    "dietary advice. Always consult a qualified healthcare provider or registered dietitian "
    "for personalized nutrition guidance. PlateGenieAI and its developers assume no liability "
    "for dietary decisions made based on this chatbot's responses.*"
)

DISCLAIMER_BANNER = (
    "⚠️ **Disclaimer**: This AI assistant provides general nutritional information based on "
    "your logged meal data. It is NOT a substitute for professional medical or dietary advice. "
    "Always consult a qualified healthcare provider or registered dietitian for personalized "
    "nutrition guidance. PlateGenieAI and its developers assume no liability for dietary "
    "decisions made based on this chatbot's responses."
)

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_HISTORY_TURNS = 10
MAX_OUTPUT_TOKENS = 1024
CACHE_TTL = 60  # seconds
CSV_COLUMNS = ["Date", "Time", "Food", "Calories", "Protein (g)", "Carbs (g)", "Fat (g)", "Portion", "Confirmed"]

# App knowledge context
APP_KNOWLEDGE = """
# PlateGenie AI - Application Knowledge Base

## What is PlateGenie AI?
PlateGenie AI is a single-file food tracking application that uses AI to detect, classify, and track nutritional content from meal photos.

## How It Works (Pipeline)
1. **Upload** a meal photo
2. **YOLOv8m** detects food items (bounding boxes)
3. **Multi-model ensemble** (3 HuggingFace classifiers) refines food identification using majority voting + confidence-weighted fusion
4. **YOLO11n counting model** counts individual items (wings, slices, etc.)
5. **Nutrition lookup** via USDA API, Open Food Facts, or local database
6. **User can edit** food names, quantities, and grams before saving
7. **Dashboard** tracks calories, macros, weekly trends

## Tabs (6 total)
1. **Upload & Analyze** - Upload meal photo, get AI detection results
2. **Dashboard** - Charts for daily calories, macros, weekly trends, top foods
3. **Meal Log** - View/delete logged entries, manual entry option
4. **Nutrition Tips** - General healthy eating advice
5. **Settings** - API keys (USDA, Groq), cache management, export, dark mode
6. **AI Assistant** - This chatbot (app help + nutrition advice)

## Configuration (Environment Variables)
- `PLATEGENIE_YOLO_MODEL` (default: yolov8m.pt) - Detection model
- `PLATEGENIE_YOLO_CONF` (default: 0.20) - Detection confidence threshold
- `PLATEGENIE_ENSEMBLE_ENABLED` (default: true) - Multi-model ensemble on/off
- `PLATEGENIE_COUNT_MODE` (default: accurate) - Counting mode: "accurate" or "fast"
- `PLATEGENIE_COUNT_MODEL` (default: yolo11n.pt) - Item counting model
- `PLATEGENIE_GROQ_KEY` - Groq API key for AI Assistant

## Ensemble Models
- yvelos/beit-food-384 (primary, weight 1.0)
- nateraw/food (ViT Food-101, weight 1.0)
- Kaludi/food-category-classification-v2.0 (weight 0.7)

## Troubleshooting
- **Models not downloading**: Check internet connection, firewall, disk space (~4GB needed)
- **Out of memory**: Set PLATEGENIE_ENSEMBLE_ENABLED=false (saves ~4GB RAM)
- **Slow analysis**: Set PLATEGENIE_COUNT_MODE=fast
- **Wrong food detected**: Edit the food name in the results table and click Recalculate
- **Groq API errors**: Check API key in Settings, verify at console.groq.com
- **Dashboard empty**: Upload and save at least one meal first
- **App won't start**: Run `pip install -r requirements.txt` to install dependencies

## Data Storage
- Meals logged to `meal_log.csv` (Date, Time, Food, Calories, Protein, Carbs, Fat, Portion)
- Nutrition cache in `nutrition_cache.json` (7-day expiry)
- Settings in `plategenie_config.json`

## Running the App
- `python app.py` (Gradio UI on port 7860)
- `python app.py --fallback` (HTML UI on port 8080)
- `./start.sh` (Unix) or `start.bat` (Windows)
"""

# Intent classification keywords
NUTRITION_KEYWORDS = [
    r"\bcalori", r"\bprotein", r"\bcarb", r"\bfat\b", r"\bmacro",
    r"\bate\b", r"\beat\b", r"\bmeal", r"\bfood\b", r"\bdiet",
    r"\bnutri", r"\bweight", r"\bhealth", r"\bbalance",
    r"\byesterday", r"\btoday", r"\bweek", r"\bdaily", r"\baverage",
    r"\bbreakfast", r"\blunch", r"\bdinner", r"\bsnack",
    r"\bhow much", r"\bhow many", r"\bam i\b", r"\bshould i\b",
    r"\bsuggest", r"\brecommend", r"\bimprove", r"\bgoal",
    r"\btrack", r"\blog", r"\bhistory", r"\btrend",
]

APP_KEYWORDS = [
    r"\bapp\b", r"\bfeature", r"\bconfig", r"\bsetting",
    r"\binstall", r"\bsetup", r"\brun\b", r"\bstart\b",
    r"\btab\b", r"\bbutton", r"\bui\b", r"\binterface",
    r"\byolo\b", r"\bmodel\b", r"\bensemble", r"\bdetect",
    r"\bapi.?key", r"\benvironment", r"\benv\b", r"\bvariable",
    r"\btroubleshoot", r"\berror", r"\bbug\b", r"\bfix\b",
    r"\bexport", r"\bcsv\b", r"\bpdf\b", r"\bdashboard",
    r"\bhow.?to", r"\bhow.?does", r"\bwhat.?is",
    r"\bupload", r"\banalyze", r"\bsave\b",
]

# ============================================
# CSV SUMMARY CACHE
# ============================================

_summary_cache = {"text": "", "timestamp": 0}


# ============================================
# FUNCTIONS
# ============================================

def classify_intent(message):
    """Classify user message intent: 'nutrition', 'app_help', or 'general'."""
    msg_lower = message.lower()
    
    nutrition_score = sum(1 for pattern in NUTRITION_KEYWORDS if re.search(pattern, msg_lower))
    app_score = sum(1 for pattern in APP_KEYWORDS if re.search(pattern, msg_lower))
    
    if nutrition_score > app_score and nutrition_score >= 1:
        return "nutrition"
    elif app_score > nutrition_score and app_score >= 1:
        return "app_help"
    else:
        return "general"


def summarize_meal_log(csv_path):
    """
    Read meal_log.csv and produce a compact text summary for the LLM context.
    Caches for 60 seconds to avoid repeated disk reads.
    """
    global _summary_cache
    
    now = time.time()
    if _summary_cache["text"] and (now - _summary_cache["timestamp"]) < CACHE_TTL:
        return _summary_cache["text"]
    
    try:
        if not Path(csv_path).exists():
            return "No meal data logged yet. The user hasn't saved any meals."
        
        df = pd.read_csv(csv_path)
        if df.empty:
            return "No meal data logged yet. The user hasn't saved any meals."
        
        # Ensure expected columns
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        
        # Parse dates
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        
        if df.empty:
            return "No valid meal data found."
        
        # Convert numeric columns
        for col in ["Calories", "Protein (g)", "Carbs (g)", "Fat (g)"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        
        summary_parts = []
        
        # Overall stats
        total_entries = len(df)
        date_range = f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
        summary_parts.append(f"Total logged entries: {total_entries} (from {date_range})")
        
        # Last 7 days
        seven_days_ago = pd.Timestamp.now() - pd.Timedelta(days=7)
        recent = df[df["Date"] >= seven_days_ago]
        
        if not recent.empty:
            daily = recent.groupby(recent["Date"].dt.date).agg({
                "Calories": "sum",
                "Protein (g)": "sum",
                "Carbs (g)": "sum",
                "Fat (g)": "sum",
                "Food": "count"
            }).reset_index()
            
            avg_cal = daily["Calories"].mean()
            avg_protein = daily["Protein (g)"].mean()
            avg_carbs = daily["Carbs (g)"].mean()
            avg_fat = daily["Fat (g)"].mean()
            
            summary_parts.append(f"\nLast 7 days daily averages:")
            summary_parts.append(f"  - Calories: {avg_cal:.0f} kcal/day")
            summary_parts.append(f"  - Protein: {avg_protein:.1f}g/day")
            summary_parts.append(f"  - Carbs: {avg_carbs:.1f}g/day")
            summary_parts.append(f"  - Fat: {avg_fat:.1f}g/day")
            summary_parts.append(f"  - Items logged: {len(recent)} across {len(daily)} days")
            
            # Top foods
            top_foods = recent["Food"].value_counts().head(5)
            if not top_foods.empty:
                summary_parts.append(f"\nMost frequent foods (last 7 days):")
                for food, count in top_foods.items():
                    summary_parts.append(f"  - {food}: {count} times")
        
        # Latest 5 entries
        latest = df.sort_values("Date", ascending=False).head(5)
        summary_parts.append(f"\nLatest 5 logged items:")
        for _, row in latest.iterrows():
            date_str = row["Date"].strftime("%Y-%m-%d") if pd.notna(row["Date"]) else "?"
            summary_parts.append(
                f"  - {date_str} | {row['Food']} | {row['Calories']:.0f} cal | "
                f"P:{row['Protein (g)']:.0f}g C:{row['Carbs (g)']:.0f}g F:{row['Fat (g)']:.0f}g"
            )
        
        # Today's meals
        today = pd.Timestamp.now().normalize()
        today_meals = df[df["Date"].dt.normalize() == today]
        if not today_meals.empty:
            today_cal = today_meals["Calories"].sum()
            today_pro = today_meals["Protein (g)"].sum()
            summary_parts.append(f"\nToday's totals: {today_cal:.0f} cal, {today_pro:.0f}g protein, "
                               f"{today_meals['Carbs (g)'].sum():.0f}g carbs, "
                               f"{today_meals['Fat (g)'].sum():.0f}g fat "
                               f"({len(today_meals)} items)")
        else:
            summary_parts.append("\nNo meals logged today yet.")
        
        summary = "\n".join(summary_parts)
        _summary_cache = {"text": summary, "timestamp": now}
        return summary
        
    except Exception as e:
        return f"Error reading meal log: {e}"


def build_system_prompt(intent, meal_summary=""):
    """Build the appropriate system prompt based on classified intent."""
    
    if intent == "app_help":
        return (
            "You are PlateGenie AI's built-in assistant. Answer questions about the application "
            "using ONLY the following knowledge base. Be concise, friendly, and helpful. "
            "If the answer isn't in the knowledge base, say you don't know.\n\n"
            f"{APP_KNOWLEDGE}"
        )
    
    elif intent == "nutrition":
        return (
            "You are a friendly nutrition advisor for PlateGenie AI. Analyze the user's meal "
            "history data provided below to give personalized insights. Be specific about what "
            "the user actually ate — reference their logged foods by name. Provide actionable "
            "suggestions. Keep responses concise (2-3 paragraphs max). "
            "IMPORTANT: Base your advice ONLY on the provided data. Do not invent meals the user didn't log. "
            "If the data is insufficient, say so.\n\n"
            f"User's Meal History:\n{meal_summary}"
        )
    
    else:  # general
        return (
            "You are PlateGenie AI's assistant. You can help with:\n"
            "1. Questions about the app (features, settings, troubleshooting)\n"
            "2. Nutrition insights based on the user's logged meals\n\n"
            "Determine what the user needs and respond helpfully. If unclear, ask them to "
            "clarify whether they need app help or nutrition advice.\n\n"
            f"App info available. Meal data summary:\n{meal_summary}"
        )


def chat(message, history, api_key, csv_path="meal_log.csv"):
    """
    Main chatbot entry point.
    
    Args:
        message: User's message string
        history: List of {"role": "user"/"assistant", "content": "..."} dicts
        api_key: Groq API key
        csv_path: Path to meal_log.csv
    
    Returns:
        Response string from the LLM (with disclaimer appended for nutrition advice)
    """
    if not message.strip():
        return "Please type a message to get started!"
    
    # Classify intent
    intent = classify_intent(message)
    
    # Get meal summary for nutrition/general modes
    meal_summary = ""
    if intent in ("nutrition", "general"):
        meal_summary = summarize_meal_log(csv_path)
    
    # Build system prompt
    system_prompt = build_system_prompt(intent, meal_summary)
    
    # Build messages array with sliding window
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add history (last N turns)
    if history:
        recent_history = history[-(MAX_HISTORY_TURNS * 2):]  # Each turn = 2 messages
        for msg in recent_history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                if msg["role"] in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Call Groq API
    response = _call_groq(messages, api_key)
    
    # Append disclaimer for nutrition advice
    if intent == "nutrition":
        response += DISCLAIMER
    
    return response


def _call_groq(messages, api_key):
    """Call the Groq API and return the response text."""
    try:
        from groq import Groq
    except ImportError:
        return (
            "⚠️ The `groq` package is not installed. Please run:\n"
            "```\npip install groq\n```\n"
            "Then restart the application."
        )
    
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.7,
        )
        return completion.choices[0].message.content
    
    except Exception as e:
        error_str = str(e).lower()
        if "rate_limit" in error_str or "429" in error_str:
            return "⚠️ Rate limit reached. The free tier allows 30 requests/minute. Please wait a moment and try again."
        elif "authentication" in error_str or "401" in error_str or "invalid api key" in error_str:
            return "⚠️ Invalid Groq API key. Please check your key in the Settings tab or at [console.groq.com](https://console.groq.com/keys)."
        else:
            return f"⚠️ Error communicating with AI: {e}"
