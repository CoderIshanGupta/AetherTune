from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from spotipy.oauth2 import SpotifyOAuth
import spotipy
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

router = APIRouter()

# Read credentials from environment variables
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri="http://127.0.0.1:8000/callback",
    scope="user-read-currently-playing user-read-playback-state"
)

# Temporary storage (development only)
spotify_client = None


@router.get("/login")
def login():
    auth_url = sp_oauth.get_authorize_url()
    return RedirectResponse(auth_url)


@router.get("/callback")
def callback(code: str):
    global spotify_client

    token_info = sp_oauth.get_access_token(code)

    spotify_client = spotipy.Spotify(
        auth=token_info["access_token"]
    )

    return {
        "message": "Login successful"
    }