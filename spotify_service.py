import auth
import random

def get_current_track_features():

    if auth.spotify_client is None:
        return None

    current = auth.spotify_client.current_user_playing_track()

    if current is None or current.get("item") is None:
        return None

    track = current["item"]

    track_name = track["name"]
    artist = track["artists"][0]["name"]
    cover = track["album"]["images"][0]["url"]
    duration = track["duration_ms"]
    progress = current["progress_ms"]
    spotify_url = track["external_urls"]["spotify"]

    simulated_features = {
        "danceability": random.uniform(0.2, 0.9),
        "energy": random.uniform(0.2, 0.9),
        "loudness": random.uniform(-20, -3),
        "speechiness": random.uniform(0.02, 0.5),
        "acousticness": random.uniform(0.0, 1.0),
        "instrumentalness": random.uniform(0.0, 1.0),
        "liveness": random.uniform(0.0, 0.8),
        "valence": random.uniform(0.0, 1.0),
        "tempo": random.uniform(60, 180)
    }

    return {
        "features": simulated_features,
        "track_name": track_name,
        "artist": artist,
        "cover": cover,
        "duration": duration,
        "progress": progress,
        "spotify_url": spotify_url
    }