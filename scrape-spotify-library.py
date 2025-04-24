"""
spotify_library_to_feather.py

Extract every song, playlist, and album in your Spotify library into three pandas
DataFrames and persist them in Apache Feather format.

Requirements:
    pip install spotipy pandas pyarrow tqdm

Environment variables (create a Spotify developer application first):
    SPOTIPY_CLIENT_ID
    SPOTIPY_CLIENT_SECRET
    SPOTIPY_REDIRECT_URI

Usage:
    python spotify_library_to_feather.py
"""
import os
import uuid
from typing import Dict, List, Any

import pandas as pd
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm

# Spotify authorisation scope – adjust if you need additional access
SCOPE = (
    "user-library-read "
    "playlist-read-private "
    "playlist-read-collaborative "
)

def get_spotify_client() -> Spotify:
    """Return an authenticated spotipy.Spotify client."""
    return Spotify(auth_manager=SpotifyOAuth(scope=SCOPE))

def paginate(sp: Spotify, method, *args, limit: int = 50, **kwargs):
    """Yield every item returned by a Spotify endpoint that supports limit/offset."""
    offset = 0
    while True:
        page = method(*args, limit=limit, offset=offset, **kwargs)
        for item in page["items"]:
            yield item
        if page["next"] is None:
            break
        offset += limit

def build_track_dict(track: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a track object into a dict suitable for a row in the songs table."""
    artists = "; ".join(a["name"] for a in track["artists"])
    album = track["album"]
    return {
        # identifiers
        "track_id": track["id"],
        "uuid": str(uuid.uuid4()),
        # core fields
        "name": track["name"],
        "artists": artists,
        "album_id": album["id"],
        "album_name": album["name"],
        "album_release_date": album["release_date"],
        "album_type": album["album_type"],
        "duration_ms": track["duration_ms"],
        "explicit": track["explicit"],
        "popularity": track.get("popularity"),
        "track_number": track["track_number"],
        "disc_number": track["disc_number"],
        "is_local": track["is_local"],
        "isrc": track.get("external_ids", {}).get("isrc"),
        "uri": track["uri"],
        "href": track["href"],
        "preview_url": track.get("preview_url"),
        "spotify_url": track["external_urls"]["spotify"],
    }

def main() -> None:
    sp = get_spotify_client()

    # ------------------- Songs (Liked + referenced elsewhere) -------------------
    song_rows: List[Dict[str, Any]] = []
    print("Fetching saved (liked) tracks …")
    for item in tqdm(paginate(sp, sp.current_user_saved_tracks), desc="Saved Tracks"):
        song_rows.append(build_track_dict(item["track"]))

    # helper mapping for playlist/album building
    track_id_to_uuid = {row["track_id"]: row["uuid"] for row in song_rows}

    # ------------------------------- Playlists ----------------------------------
    playlist_rows: List[Dict[str, Any]] = []
    print("Fetching playlists …")
    for playlist in tqdm(paginate(sp, sp.current_user_playlists), desc="Playlists"):
        playlist_uuid = str(uuid.uuid4())
        track_uuids: List[str] = []
        for pl_item in paginate(sp, sp.playlist_items, playlist["id"]):
            trk = pl_item["track"]
            if trk is None or trk["id"] is None:
                continue  # skip unavailable tracks
            if trk["id"] not in track_id_to_uuid:
                # unseen track encountered while traversing playlists
                row = build_track_dict(trk)
                song_rows.append(row)
                track_id_to_uuid[trk["id"]] = row["uuid"]
            track_uuids.append(track_id_to_uuid[trk["id"]])

        playlist_rows.append({
            "playlist_id": playlist["id"],
            "uuid": playlist_uuid,
            "name": playlist["name"],
            "description": playlist.get("description"),
            "owner_id": playlist["owner"]["id"],
            "public": playlist["public"],
            "snapshot_id": playlist["snapshot_id"],
            "spotify_url": playlist["external_urls"]["spotify"],
            "song_uuids": track_uuids,
        })

    # -------------------------------- Albums ------------------------------------
    album_rows: List[Dict[str, Any]] = []
    print("Fetching saved albums …")
    for album_item in tqdm(paginate(sp, sp.current_user_saved_albums), desc="Albums"):
        album = album_item["album"]
        album_uuid = str(uuid.uuid4())
        track_uuids: List[str] = []
        for trk in paginate(sp, sp.album_tracks, album["id"]):
            if trk["id"] not in track_id_to_uuid:
                row = build_track_dict(trk)
                song_rows.append(row)
                track_id_to_uuid[trk["id"]] = row["uuid"]
            track_uuids.append(track_id_to_uuid[trk["id"]])

        album_rows.append({
            "album_id": album["id"],
            "uuid": album_uuid,
            "name": album["name"],
            "artist": "; ".join(a["name"] for a in album["artists"]),
            "release_date": album["release_date"],
            "album_type": album["album_type"],
            "total_tracks": album["total_tracks"],
            "spotify_url": album["external_urls"]["spotify"],
            "song_uuids": track_uuids,
        })

    # ----------------------------- Data persistence -----------------------------
    print("Writing DataFrames to Feather …")
    pd.DataFrame(song_rows).to_feather("songs.feather")
    pd.DataFrame(playlist_rows).to_feather("playlists.feather")
    pd.DataFrame(album_rows).to_feather("albums.feather")

    print("Done ✅  → songs.feather, playlists.feather, albums.feather")


if __name__ == "__main__":
    main()
