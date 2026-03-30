<div align="center">

<img src="static/logo.png" alt="AetherTune Logo" width="120"/>

#  AetherTune

### *Right Music. Every Moment.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.133-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Spotify API](https://img.shields.io/badge/Spotify%20API-enabled-1DB954?style=flat-square&logo=spotify&logoColor=white)](https://developer.spotify.com)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-MVP%20Development-orange?style=flat-square)]()

</div>

---

##  Overview

**AetherTune** is a context-aware music recommendation system that analyzes your **current mood, activity, and real-time listening context** to decide whether what you're playing right now is the right fit — and if not, suggests better alternatives instantly.

Unlike traditional platforms that rely on listening history, AetherTune focuses on **what you need right now**: evaluating your currently playing Spotify track against your activity using a trained machine learning model, then recommending replacements when the vibe doesn't match.

---

##  The Problem

Modern music platforms are **reactive, not intelligent**:

-  Recommendations are driven by past behaviour, not present context
-  Users waste time scrolling to find the right music for their mood
-  No platform truly understands *what you're doing right now*

> *Users often don't know what they want to listen to — they just know the current song isn't working.*

---

## The Solution

AetherTune introduces **moment-based personalization** through three pillars:

| Pillar | Description |
|---|---|
|  **Mood-Aware** | Evaluates audio features (valence, energy, acousticness) against your emotional state |
|  **Activity-Driven** | Tailors recommendations to studying, driving, meditating, or exercising |
|  **Adaptive Learning** | Adjusts its tolerance thresholds over time based on your feedback |

---

##  How It Works

```
User selects Activity + Age
        ↓
Spotify API fetches currently playing track
        ↓
ML Model evaluates audio features against activity profile
        ↓
Dynamic Threshold check (sigmoid + feedback decay)
        ↓
  ┌─────────────────────────────────┐
  │  Match?  → "Song is suitable" │
  │  No match? → 5 Recommendations│
  └─────────────────────────────────┘
        ↓
User submits feedback → Threshold adapts over time
```

### Adaptive Threshold System

AetherTune doesn't use a fixed confidence cutoff. Instead, it computes a **sigmoid-based dynamic threshold** by age group, then adjusts it through a time-decaying feedback loop:

- **More feedback = more personalised threshold**
- **Inactivity causes gradual decay** back to the base — keeping the model fresh
- Adjustments are capped at ±0.2 to prevent drift

---

##  Project Structure

```
AetherTune/
├── main.py                  # FastAPI app — /predict and /feedback endpoints
├── auth.py                  # Spotify OAuth2 login & callback
├── spotify_service.py       # Fetches currently playing track + audio features
├── recommendation.py        # Activity-based Spotify track search
├── ml_model.py              # Model training script (Random Forest)
├── database.py              # SQLite schema — feedback & tolerance tables
├── models/
│   ├── activity_model.pkl   # Trained MultiOutputClassifier
│   └── scaler.pkl           # StandardScaler for feature normalisation
├── data/
│   ├── spotify_dataset.csv          # Raw Spotify audio features dataset
│   └── spotify_dataset_labeled.csv  # Labelled dataset with activity targets
├── static/
│   ├── logo.png
│   └── style.css
└── templates/
    └── index.html           # Jinja2 web frontend
```

---

##  ML Model Details

| Property | Value |
|---|---|
| Model | `MultiOutputClassifier` wrapping `RandomForestClassifier` |
| Labels | `studying`, `driving`, `meditating`, `exercising` |
| Features | `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `age` |
| Scaler | `StandardScaler` |
| Train/Test Split | 80 / 20 |

**Label generation logic (rule-based for training):**

```python
studying   → energy < 0.6  AND speechiness < 0.4
driving    → energy > 0.5  AND tempo > 90
meditating → acousticness > 0.6 AND energy < 0.4
exercising → energy > 0.7  AND tempo > 120
```

---

##  Getting Started

### Prerequisites

- Python 3.10+
- A [Spotify Developer Account](https://developer.spotify.com/dashboard) with a registered app

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/AetherTune.git
cd AetherTune
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
```

>  **Never commit your `.env` file.** Add it to `.gitignore`.

### 4. (Optional) Retrain the Model

The pre-trained model is already included in `models/`. To retrain:

```bash
python ml_model.py
```

### 5. Run the App

```bash
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000` in your browser.

---

##  API Reference

### `GET /login`
Redirects to Spotify's OAuth2 authorization page.

### `GET /callback?code=`
Handles the OAuth2 callback and initialises the Spotify client.

### `GET /predict?activity={0-3}&age={int}`
Analyses the currently playing track and returns a suitability verdict.

**Activity codes:**

| Code | Activity |
|---|---|
| `0` | Studying |
| `1` | Driving |
| `2` | Meditating |
| `3` | Exercising |

**Example response (suitable):**
```json
{
  "Current_Track": {
    "name": "Lo-Fi Chill",
    "artist": "ChillHop Music",
    "cover": "https://...",
    "duration": 210000,
    "progress": 45000,
    "spotify_url": "https://open.spotify.com/track/..."
  },
  "message": "Song is suitable for studying ✅"
}
```

**Example response (not suitable):**
```json
{
  "Current_Track": { ... },
  "ALERT": "Song is NOT suitable for studying ❌",
  "Recommended_Tracks": [
    { "name": "...", "artist": "...", "id": "...", "url": "..." }
  ]
}
```

### `POST /feedback?age={int}&activity={0-3}&liked={bool}`
Submits user feedback to personalise the tolerance threshold.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Uvicorn |
| ML | Scikit-learn (Random Forest, MultiOutputClassifier) |
| Data | Pandas, NumPy |
| Spotify Integration | Spotipy |
| Database | SQLite |
| Frontend | Jinja2 Templates, HTML/CSS |
| Auth | Spotify OAuth2 (via Spotipy) |

---

## 🗺️ Roadmap

- [x] Spotify OAuth2 integration
- [x] Real-time track evaluation via ML model
- [x] Activity-based recommendation fallback
- [x] Adaptive feedback loop with time decay
- [ ] Replace simulated audio features with Spotify's live Audio Features API
- [ ] User accounts and persistent profiles
- [ ] Browser extension for passive, always-on context detection
- [ ] Mood inference from time-of-day and calendar context
- [ ] Expand to Apple Music and YouTube Music

---

##  Authors

| Name | Role |
|---|---|
| Riddhi Mishra | Co-creator |
| Ishan Gupta | Co-creator |
| Abhijeet Kumar | Co-creator |
| Yadvendra Tripathi | Co-creator |
| Avishi Sinha | Co-creator |

---

##  Notes

- Audio features in `spotify_service.py` are currently **simulated** with `random.uniform`. Replace with a live call to Spotify's [Audio Features endpoint](https://developer.spotify.com/documentation/web-api/reference/get-audio-features) for production use.
- The `spotify_client` in `auth.py` is stored in memory (global variable). For a production deployment, use a proper session or token store (e.g., Redis).

---

##  License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with 🎵 by the AetherTune Team</sub>
</div>
