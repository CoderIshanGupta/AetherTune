from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import joblib
import numpy as np
import math
import sqlite3
from datetime import datetime

from auth import router as auth_router
from spotify_service import get_current_track_features
from recommendation import recommend
from database import init_db

DB_NAME = "aethertune.db"

# FastAPI Initialization
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router) 

# Static + Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize DB
init_db()

# Load ML Model
model = joblib.load("models/activity_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Sigmoid Threshold with Decay
def get_dynamic_threshold(age: int, activity_name: str):

    LOWER = 0.55
    UPPER = 0.85
    midpoint = 40
    steepness = 0.08

    sigmoid = 1 / (1 + math.exp(-steepness * (age - midpoint)))
    base_threshold = LOWER + (UPPER - LOWER) * sigmoid

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT adjustment, last_updated
        FROM user_tolerance
        WHERE age=? AND activity=?
    """, (age, activity_name))

    row = cursor.fetchone()
    conn.close()

    adjustment = 0

    if row:
        adjustment, last_updated = row
        last_updated = datetime.fromisoformat(last_updated)

        days_passed = (datetime.now() - last_updated).days
        decay_factor = 0.98 ** days_passed
        adjustment *= decay_factor

    threshold = base_threshold + adjustment
    threshold = max(0.5, min(threshold, 0.9))

    return threshold

# Serve Frontend
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction Endpoint
@app.get("/predict")
def predict(activity: int, age: int):

    data = get_current_track_features()

    if data is None: 
        return {"message": "No song currently playing or user not logged in."}

    features = data["features"]
    track_name = data["track_name"]
    artist = data["artist"]
    cover = data["cover"]
    duration = data["duration"]
    progress = data["progress"]
    spotify_url = data["spotify_url"]

    input_data = np.array([[
        features["danceability"],
        features["energy"],
        features["loudness"],
        features["speechiness"],
        features["acousticness"],
        features["instrumentalness"],
        features["liveness"],
        features["valence"],
        features["tempo"],
        age
    ]])

    input_data = scaler.transform(input_data)
    probabilities = model.predict_proba(input_data)

    activity_map = {
        0: "studying",
        1: "driving",
        2: "meditating",
        3: "exercising"
    }

    selected_activity = activity_map.get(activity)

    if selected_activity is None:
        return {"error": "Invalid activity value."}

    # Extract probability for selected activity
    activity_index = activity
    selected_score = probabilities[activity_index][0][1]

    threshold = get_dynamic_threshold(age, selected_activity)

    current_track_info = {
        "name": track_name,
        "artist": artist,
        "cover": cover,
        "duration": duration,
        "progress": progress,
        "spotify_url": spotify_url
    }

    if selected_score >= threshold:
        return {
            "Current_Track": current_track_info,
            "message": f"Song is suitable for {selected_activity} ✅"
        }

    recommendations = recommend(activity)

    return {
        "Current_Track": current_track_info,
        "ALERT": f"Song is NOT suitable for {selected_activity} ❌",
        "Recommended_Tracks": recommendations
    }

# Feedback Endpoint
@app.post("/feedback")
def feedback(age: int, activity: int, liked: bool):

    activity_map = {
        0: "studying",
        1: "driving",
        2: "meditating",
        3: "exercising"
    }

    activity_name = activity_map.get(activity)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO user_feedback (age, activity, liked, timestamp)
        VALUES (?, ?, ?, ?)
    """, (age, activity_name, liked, datetime.now().isoformat()))

    cursor.execute("""
        SELECT adjustment FROM user_tolerance
        WHERE age=? AND activity=?
    """, (age, activity_name))

    row = cursor.fetchone()
    current_adjustment = row[0] if row else 0

    cursor.execute("""
        SELECT COUNT(*) FROM user_feedback
        WHERE age=? AND activity=?
    """, (age, activity_name))

    feedback_count = cursor.fetchone()[0]

    learning_rate = max(0.01, 0.05 * (1 / (1 + feedback_count / 10)))

    if liked:
        new_adjustment = current_adjustment - learning_rate
    else:
        new_adjustment = current_adjustment + learning_rate

    new_adjustment = max(-0.2, min(new_adjustment, 0.2))

    cursor.execute("""
        INSERT OR REPLACE INTO user_tolerance
        (age, activity, adjustment, last_updated)
        VALUES (?, ?, ?, ?)
    """, (age, activity_name, new_adjustment, datetime.now().isoformat()))

    conn.commit()
    conn.close()

    return {"message": "Feedback recorded"}