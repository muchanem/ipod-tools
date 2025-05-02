#!/usr/bin/env python3
"""
preview_feather.py – Load three Spotify-export Feather v2 files
and print the first 5 rows of each for a quick sanity check.
"""

import pandas as pd
# Show every column
pd.set_option('display.max_columns', None)      # no limit on number of columns :contentReference[oaicite:0]{index=0}
# Don’t wrap to a fixed width
pd.set_option('display.width', None)            # disable line wrapping :contentReference[oaicite:1]{index=1}
# Don’t truncate contents of each cell
pd.set_option('display.max_colwidth', None)     # allow arbitrarily wide columns :contentReference[oaicite:2]{index=2}

def preview_feather(path: str, n: int = 5):
    """
    Read a Feather file into a DataFrame and print the first n rows.

    Parameters:
    -----------
    path : str
        Path to the .feather file.
    n : int
        Number of rows to display (default: 5).
    """
    df = pd.read_feather(path)
    print(f"\n--- Preview of {path} (first {n} rows) ---")
    json_str = df.head(5).to_json(orient="records", indent=2)
    print(json_str)


def main():
    files = [
        "spotify_songs.feather",
        "spotify_playlists.feather",
        "spotify_albums.feather",
        "spotify_songs_with_tidal_sample.feather"
    ]
    for file in files:
        try:
            preview_feather(file)
        except Exception as e:
            print(f"Error loading {file}: {e}")


if __name__ == "__main__":
    main()
