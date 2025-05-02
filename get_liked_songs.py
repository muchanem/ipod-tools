#!/usr/bin/env python3
"""
export_liked_tracks_full.py – Fetch all your saved Spotify tracks’
metadata into a table with the same columns as the full exporter.
"""

import time
import uuid
import logging
from collections import OrderedDict

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException
import pandas as pd
from tqdm import tqdm

# -- Configuration ---------------------------------------------------------
SCOPE = "user-library-read"
OUTPUT_FEATHER = "favs.feather"

# -- Retry helper (429 / 5xx) ---------------------------------------------
MAX_RETRIES = 5
INITIAL_DELAY = 2

def call_spotify_api(api_func, *args, **kwargs):
    retries, delay = 0, INITIAL_DELAY
    while True:
        try:
            return api_func(*args, **kwargs)
        except SpotifyException as e:
            status = e.http_status
            # Rate‑limit (429)?
            if status == 429 and retries < MAX_RETRIES:
                wait = int(e.headers.get("Retry-After", delay))
                logging.warning("429 → sleeping %ds (retry %d/%d)", wait, retries+1, MAX_RETRIES)
                time.sleep(wait)
                retries += 1
                continue
            # Server error?
            if 500 <= status < 600 and retries < MAX_RETRIES:
                logging.warning("%d → backing off %ds (retry %d/%d)", status, delay, retries+1, MAX_RETRIES)
                time.sleep(delay)
                delay *= 2
                retries += 1
                continue
            raise

# -- Pagination for saved tracks ------------------------------------------
def fetch_all_saved_tracks(client):
    all_items = []
    limit = 50
    offset = 0
    while True:
        page = call_spotify_api(client.current_user_saved_tracks, limit=limit, offset=offset)
        items = page.get("items", [])
        if not items:
            break
        all_items.extend(items)
        offset += len(items)
    return all_items

# -- Extract metadata for each saved track -------------------------------
def extract_track(item):
    t = item["track"]
    album = t.get("album", {}) or {}
    release = album.get("release_date")
    year = release.split("-",1)[0] if release else None
    return {
        "uuid":               str(uuid.uuid4()),
        "spotify_id":         t.get("id"),
        "title":              t.get("name"),
        "artists":            [a["name"] for a in t.get("artists", [])],
        "album_name":         album.get("name"),
        "album_artists":      [a["name"] for a in album.get("artists", [])],
        "track_number":       t.get("track_number"),
        "disc_number":        t.get("disc_number"),
        "duration_ms":        t.get("duration_ms"),
        "explicit":           t.get("explicit"),
        "popularity":         t.get("popularity"),
        "spotify_url":        t.get("external_urls", {}).get("spotify"),
        "spotify_cover_url":  (album.get("images") or [{}])[0].get("url"),
        "isrc":               t.get("external_ids", {}).get("isrc"),
        "release_date":       release,
        "year":               year,
        "added_at":           item.get("added_at"),
        "album_spotify_id":   album.get("id"),
        "album_spotify_url":  album.get("external_urls", {}).get("spotify"),
        # placeholder for genres, filled below
        "album_genres":       []
    }

def main():
    logging.basicConfig(level=logging.INFO)
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SCOPE))
    logging.info("Fetching saved tracks…")
    saved = fetch_all_saved_tracks(sp)  # all liked songs :contentReference[oaicite:3]{index=3}
    logging.info("Processing %d tracks…", len(saved))

    tracks = OrderedDict()
    album_ids = set()

    # extract track data and collect album IDs
    for item in tqdm(saved, unit="trk"):
        data = extract_track(item)
        tracks[data["uuid"]] = data
        if data["album_spotify_id"]:
            album_ids.add(data["album_spotify_id"])

    # batch‑fetch album genres
    logging.info("Fetching genres for %d albums…", len(album_ids))
    album_ids = list(album_ids)
    for i in tqdm(range(0, len(album_ids), 20), unit="batch"):
        batch = album_ids[i:i+20]
        resp = call_spotify_api(sp.albums, batch)   # /albums endpoint :contentReference[oaicite:4]{index=4}
        for alb in resp.get("albums", []):
            for tr in tracks.values():
                if tr["album_spotify_id"] == alb.get("id"):
                    tr["album_genres"] = alb.get("genres", [])

    # build DataFrame
    df = pd.DataFrame(tracks.values())
    # ensure list-columns correct
    for col in ["artists","album_artists","album_genres"]:
        df[col] = df[col].apply(lambda v: v if isinstance(v,list) else ([] if pd.isna(v) else [v]))
    # reorder exactly as original
    df = df[[
        "uuid","spotify_id","title","artists","album_name","album_artists",
        "track_number","disc_number","duration_ms","explicit","popularity",
        "spotify_url","spotify_cover_url","isrc","release_date","year",
        "added_at","album_spotify_id","album_spotify_url","album_genres"
    ]]

    # export
    logging.info("Writing Feather → %s", OUTPUT_FEATHER)
    df.to_feather(OUTPUT_FEATHER, version=2)
    logging.info("Done.")

if __name__ == "__main__":
    main()

