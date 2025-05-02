#!/usr/bin/env python3
"""
spotify_library_exporter.py – Export your saved Spotify tracks, playlists,
and albums into three Feather v2 files for ID3 tagging / iTunes import.
Features:
  - Full metadata: titles, artists (list), album artists (list), year,
    genres, cover URLs, etc.
  - UUIDs for tracks, playlists, albums.
  - Robust rate-limit (429) and 5xx retry/backoff logic.
  - Progress bars via tqdm, detailed logging.
  - Sample mode (--sample) to smoke-test on a few items.
"""

import os
import time
import uuid
import logging
import argparse
from collections import OrderedDict

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException

import pandas as pd
from tqdm import tqdm

# Ensure pyarrow is present for Feather v2
try:
    import pyarrow  # noqa: F401
except ImportError:
    raise SystemExit("Error: pyarrow is required. Install via `pip install pyarrow`.")

# --- Configuration ---------------------------------------------------------

SCOPE = "user-library-read playlist-read-private playlist-read-collaborative"
SONGS_FILE     = "spotify_songs.feather"
PLAYLISTS_FILE = "spotify_playlists.feather"
ALBUMS_FILE    = "spotify_albums.feather"

MAX_RETRIES         = 5
INITIAL_RETRY_DELAY = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Helpers ---------------------------------------------------------------

def call_spotify_api(api_func, *args, **kwargs):
    """Call Spotify API with retry on 429 and 5xx errors."""
    retries, delay = 0, INITIAL_RETRY_DELAY
    while True:
        try:
            return api_func(*args, **kwargs)
        except SpotifyException as e:
            status = e.http_status
            # Rate limited?
            if status == 429 and retries < MAX_RETRIES:
                retry_after = int(e.headers.get("Retry-After", delay))
                retries += 1
                logging.warning(
                    "429 rate limit: sleeping %s s (%s/%s retries)",
                    retry_after, retries, MAX_RETRIES
                )
                for _ in tqdm(range(retry_after), desc="Rate-limit wait", unit="s", leave=False):
                    time.sleep(1)
                continue
            # Server error?
            if 500 <= status < 600 and retries < MAX_RETRIES:
                retries += 1
                logging.warning(
                    "Server %s error: backing off %s s (%s/%s retries)",
                    status, delay, retries, MAX_RETRIES
                )
                for _ in tqdm(range(delay), desc=f"Server {status} wait", unit="s", leave=False):
                    time.sleep(1)
                delay *= 2
                continue
            # Otherwise, give up
            logging.error("Spotify API error %s: %s", status, e)
            raise

def fetch_paginated_items(client, api_call, *args, tqdm_desc="Fetching", **kwargs):
    """Fetch all pages from a Spotify paginated endpoint with a progress bar."""
    items = []
    page = call_spotify_api(api_call, *args, **kwargs)
    if not page:
        return items

    total = page.get("total") or 0
    pbar = tqdm(total=total, desc=tqdm_desc, unit="item")
    while page:
        chunk = page.get("items", [])
        items.extend(chunk)
        pbar.update(len(chunk))
        next_url = page.get("next")
        page = call_spotify_api(client.next, page) if next_url else None
    pbar.close()
    return items

def ensure_lists(df: pd.DataFrame, cols):
    """Convert nulls or scalars into empty/singleton lists for Arrow list<> columns."""
    for c in cols:
        df[c] = df[c].apply(lambda v: v if isinstance(v, list) else ([] if pd.isna(v) else [v]))
    return df

def maybe_slice(seq, max_items):
    """Return only the first max_items if max_items>0, else return the full seq."""
    return seq[:max_items] if max_items and len(seq) > max_items else seq

def extract_track_data(item, track_map, id_map):
    """
    Given a Spotify track-item dict (from saved tracks or a playlist/album),
    generate/lookup a UUID, extract metadata (artists list, album_artists list,
    release_date, year, etc.), and store in track_map.
    """
    if not item or "track" not in item or not item["track"] or not item["track"].get("id"):
        return None
    t = item["track"]
    tid = t["id"]
    if tid in id_map:
        return id_map[tid]

    t_uuid = str(uuid.uuid4())
    id_map[tid] = t_uuid

    album = t.get("album", {}) or {}
    release_date = album.get("release_date")
    year = release_date.split("-", 1)[0] if release_date else None

    track_map[t_uuid] = {
        "uuid":               t_uuid,
        "spotify_id":         tid,
        "title":              t.get("name"),
        "artists":            [a["name"] for a in t.get("artists", []) if a.get("name")],
        "album_name":         album.get("name"),
        "album_artists":      [a["name"] for a in album.get("artists", []) if a.get("name")],
        "track_number":       t.get("track_number"),
        "disc_number":        t.get("disc_number"),
        "duration_ms":        t.get("duration_ms"),
        "explicit":           t.get("explicit"),
        "popularity":         t.get("popularity"),
        "spotify_url":        t.get("external_urls", {}).get("spotify"),
        "spotify_cover_url":  (album.get("images") or [{}])[0].get("url"),
        "isrc":               t.get("external_ids", {}).get("isrc"),
        "release_date":       release_date,
        "year":               year,
        "added_at":           item.get("added_at"),
        "album_spotify_id":   album.get("id"),
        "album_spotify_url":  album.get("external_urls", {}).get("spotify"),
        # genre field filled later
        "album_genres":       []
    }
    return t_uuid

# --- Main ------------------------------------------------------------------

def main():
    # CLI: sample mode?
    parser = argparse.ArgumentParser(description="Export Spotify library to Feather")
    parser.add_argument(
        "--sample", action="store_true",
        help="Fetch only a small subset (10 tracks, 1 playlist, 1 album) for testing"
    )
    args = parser.parse_args()

    liked_limit = 10 if args.sample else None
    plist_limit = 1  if args.sample else None
    album_limit = 1  if args.sample else None

    # Authenticate
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SCOPE))
    call_spotify_api(sp.current_user)  # sanity-check

    # Containers
    tracks_map = OrderedDict()
    id_map     = {}  # spotify_id -> uuid
    album_ids_for_genre_lookup = set()

    # --- 1) Liked Songs ---------------------------------------------------
    liked = fetch_paginated_items(
        sp, sp.current_user_saved_tracks,
        limit=liked_limit,
        tqdm_desc="Liked songs (sample)" if args.sample else "Liked songs"
    )
    liked = maybe_slice(liked, liked_limit)
    for item in tqdm(liked, desc="Processing liked songs", unit="trk"):
        tu = extract_track_data(item, tracks_map, id_map)
        if tu:
            album_ids_for_genre_lookup.add(tracks_map[tu]["album_spotify_id"])

    # --- 2) Playlists -----------------------------------------------------
    playlists_rows = []
    pls = fetch_paginated_items(
        sp, sp.current_user_playlists,
        limit=plist_limit,
        tqdm_desc="Playlists list (sample)" if args.sample else "Playlists list"
    )
    pls = maybe_slice(pls, plist_limit)
    for pl in tqdm(pls, desc="Playlists", unit="pl"):
        # fetch items in this playlist
        items = fetch_paginated_items(
            sp, sp.playlist_items, pl["id"],
            fields=(
                "items(added_at,is_local,track("
                "id,name,artists,album(id,name,artists,images,release_date,"
                "external_urls),track_number,disc_number,duration_ms,"
                "explicit,popularity,external_ids,external_urls"
                ")),next,total,limit"
            ),
            tqdm_desc=f"Playlist «{pl['name'][:25]}»"
        )
        track_uuids = []
        for it in tqdm(items, desc="Proc playlist tracks", unit="trk", leave=False):
            tu = extract_track_data(it, tracks_map, id_map)
            if tu:
                track_uuids.append(tu)
                album_ids_for_genre_lookup.add(tracks_map[tu]["album_spotify_id"])

        playlists_rows.append({
            "uuid":         str(uuid.uuid4()),
            "spotify_id":   pl["id"],
            "name":         pl.get("name"),
            "description":  pl.get("description"),
            "owner_name":   pl["owner"]["display_name"],
            "owner_id":     pl["owner"]["id"],
            "public":       pl.get("public"),
            "collaborative":pl.get("collaborative"),
            "snapshot_id":  pl.get("snapshot_id"),
            "spotify_url":  pl["external_urls"]["spotify"],
            "track_uuids":  track_uuids,
            "track_count":  len(track_uuids),
        })

    # --- 3) Saved Albums --------------------------------------------------
    albums_rows = []
    albs = fetch_paginated_items(
        sp, sp.current_user_saved_albums,
        limit=album_limit,
        tqdm_desc="Albums list (sample)" if args.sample else "Albums list"
    )
    albs = maybe_slice(albs, album_limit)
    for entry in tqdm(albs, desc="Albums", unit="alb"):
        alb = entry.get("album") or {}
        alb_id = alb.get("id")
        alb_uuid = str(uuid.uuid4())

        # fetch full track objects for this album
        partial = call_spotify_api(sp.album_tracks, alb_id).get("items", [])
        ids = [t["id"] for t in partial if t.get("id")]
        full = []
        for i in range(0, len(ids), 50):
            batch = ids[i:i+50]
            full.extend(call_spotify_api(sp.tracks, batch).get("tracks", []))

        track_uuids = []
        for t in tqdm(full, desc="Proc album tracks", unit="trk", leave=False):
            tu = extract_track_data({"track": t, "added_at": entry.get("added_at")}, tracks_map, id_map)
            if tu:
                track_uuids.append(tu)
                album_ids_for_genre_lookup.add(alb_id)

        albums_rows.append({
            "uuid":         alb_uuid,
            "spotify_id":   alb_id,
            "name":         alb.get("name"),
            "artists":      [a["name"] for a in alb.get("artists", []) if a.get("name")],
            "album_type":   alb.get("album_type"),
            "release_date": alb.get("release_date"),
            "release_date_precision": alb.get("release_date_precision"),
            "total_tracks": alb.get("total_tracks"),
            "label":        alb.get("label"),
            "popularity":   alb.get("popularity"),
            "genres":       [],  # filled next
            "spotify_url":  alb.get("external_urls", {}).get("spotify"),
            "spotify_cover_url": (alb.get("images") or [{}])[0].get("url"),
            "track_uuids":  track_uuids,
            "added_at":     entry.get("added_at"),
        })

    # --- 4) Fetch Album Genres -------------------------------------------
    logging.info("Fetching genres for %s albums...", len(album_ids_for_genre_lookup))
    album_genres_map = {}
    batch_size = 20
    album_ids = [aid for aid in album_ids_for_genre_lookup if aid]
    for i in tqdm(range(0, len(album_ids), batch_size), desc="Album genres", unit="batch"):
        batch = album_ids[i : i + batch_size]
        try:
            info = call_spotify_api(sp.albums, batch)
            for a in info.get("albums", []):
                album_genres_map[a.get("id")] = a.get("genres", [])
        except Exception:
            logging.exception("Failed to fetch genres for batch %s", batch[:3])

    # inject genres into albums_rows
    for row in albums_rows:
        row["genres"] = album_genres_map.get(row["spotify_id"], [])

    # inject genres into track_map
    for t in tracks_map.values():
        t["album_genres"] = album_genres_map.get(t["album_spotify_id"], [])

    # --- 5) Build DataFrames --------------------------------------------
    songs_df     = pd.DataFrame(tracks_map.values())
    playlists_df = pd.DataFrame(playlists_rows)
    albums_df    = pd.DataFrame(albums_rows)

    # reorder / whitelist song columns
    songs_df = songs_df[[
        "uuid","spotify_id","title","artists","album_name","album_artists",
        "track_number","disc_number","duration_ms","explicit","popularity",
        "spotify_url","spotify_cover_url","isrc","release_date","year",
        "added_at","album_spotify_id","album_spotify_url","album_genres"
    ]]

    # ensure list-columns are real lists
    songs_df     = ensure_lists(songs_df,     ["artists","album_artists","album_genres"])
    playlists_df = ensure_lists(playlists_df, ["track_uuids"])
    albums_df    = ensure_lists(albums_df,    ["artists","genres","track_uuids"])

    # --- 6) Export to Feather v2 -----------------------------------------
    logging.info("Writing Feather v2 files...")
    songs_df.to_feather(SONGS_FILE,     version=2)
    playlists_df.to_feather(PLAYLISTS_FILE, version=2)
    albums_df.to_feather(ALBUMS_FILE,      version=2)
    logging.info("Done! ✅")

if __name__ == "__main__":
    main()
