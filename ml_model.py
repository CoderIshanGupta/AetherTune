import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib


# Load Dataset
df = pd.read_csv("data/spotify_dataset.csv", encoding="latin1")

# Simulated age data for training (development mode)
df["age"] = np.random.randint(15, 60, size=len(df))

# Generate Multi-Label Targets


def label_studying(row):
    return 1 if row["energy"] < 0.6 and row["speechiness"] < 0.4 else 0

def label_driving(row):
    return 1 if row["energy"] > 0.5 and row["tempo"] > 90 else 0

def label_meditating(row):
    return 1 if row["acousticness"] > 0.6 and row["energy"] < 0.4 else 0

def label_exercising(row):
    return 1 if row["energy"] > 0.7 and row["tempo"] > 120 else 0

df["studying"] = df.apply(label_studying, axis=1)
df["driving"] = df.apply(label_driving, axis=1)
df["meditating"] = df.apply(label_meditating, axis=1)
df["exercising"] = df.apply(label_exercising, axis=1)


# Features

features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "age"
]

X = df[features]
y = df[["studying", "driving", "meditating", "exercising"]]


# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Scaling

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Multi-Label Model

base_model = RandomForestClassifier()
model = MultiOutputClassifier(base_model)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Multi-label accuracy:", accuracy)


# Save Model + Scaler

joblib.dump(model, "models/activity_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Multi-label model saved successfully.")
