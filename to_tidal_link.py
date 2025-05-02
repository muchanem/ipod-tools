import llm
import sys
import pandas as pd
import pyarrow
import httpx
import asyncio
import time
import os
import logging
import argparse
from tqdm.asyncio import tqdm_asyncio
from urllib.parse import quote
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import random
from httpx import TimeoutException, ConnectError, ReadTimeout, HTTPStatusError, RequestError

# --- Configuration ---
load_dotenv()
FEATHER_INPUT_PATH = "spotify_songs.feather"
FEATHER_OUTPUT_PATH = "songs.feather"
TIDAL_CLIENT_ID = os.getenv("TIDAL_CLIENT_ID")
TIDAL_CLIENT_SECRET = os.getenv("TIDAL_CLIENT_SECRET")
TIDAL_API_BASE_URL = "https://openapi.tidal.com/v2"
TIDAL_AUTH_URL = "https://auth.tidal.com/v1/oauth2/token"
COUNTRY_CODE = "US"
MAX_TIDAL_RESULTS_PER_SONG = 5
REQUEST_TIMEOUT = 30
RETRY_ATTEMPTS = 3
RETRY_WAIT_MULTIPLIER = 1
RETRY_WAIT_MAX = 10
SAMPLE_SIZE = 5
SEM_MAX_CONCURRENT = 2      # throttle concurrent Tidal calls
INTERMEDIATE_CHUNK = 1000   # save every 1 000 tracks

RETRY_ERRORS = (
    TimeoutException,
    ConnectError,
    ReadTimeout,
    HTTPStatusError,
    RequestError,
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LLM Image Comparison ---
async def compare_images_llm(image1_url: str, image2_url: str) -> Optional[bool]:
    if not image1_url or not image2_url:
        logger.warning("Missing one or both image URLs for comparison.")
        return None
    try:
        model = llm.get_async_model("gemini-2.5-flash-preview-04-17")
        prompt = (
            "Do these two images look the same? If so, output only the word 'Yes' "
            "if not output only the word 'No'"
        )
        shuffled = random.sample([image1_url, image2_url], 2)
        a1, a2 = llm.Attachment(url=shuffled[0]), llm.Attachment(url=shuffled[1])
        response = await model.prompt(prompt, attachments=[a1, a2], thinking_budget=0)
        text = await response.text()
        logger.debug(f"LLM comparison: '{text}'")
        return 'yes' in text.strip().lower()
    except Exception as e:
        logger.error("LLM image comparison failed", exc_info=True)
        return None

# --- Tidal Authentication Manager ---
class TidalAuthManager:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id, self.client_secret = client_id, client_secret
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[float] = None
        self._lock = asyncio.Lock()

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, max=RETRY_WAIT_MAX),
        retry=retry_if_exception_type(RETRY_ERRORS),
        reraise=True
    )
    async def _fetch_token(self, client: httpx.AsyncClient):
        logger.info("Fetching new Tidal access token...")
        if client.is_closed:
            raise RuntimeError("HTTP client closed.")
        response = await client.post(
            TIDAL_AUTH_URL,
            auth=(self.client_id, self.client_secret),
            data={"grant_type": "client_credentials"},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        self._access_token = data.get("access_token")
        self._token_expiry = time.time() + data.get("expires_in", 0) - 60
        logger.info("Obtained new Tidal access token.")

    async def get_token(self, client: httpx.AsyncClient) -> str:
        async with self._lock:
            if client.is_closed:
                raise RuntimeError("HTTP client closed.")
            if not self._access_token or (self._token_expiry and time.time() >= self._token_expiry):
                await self._fetch_token(client)
            if not self._access_token:
                raise RuntimeError("Failed to obtain Tidal token.")
            return self._access_token

# --- Tidal API Client ---
class TidalApiClient:
    def __init__(self, auth_manager: TidalAuthManager):
        self.auth_manager = auth_manager
        self.client = httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
        self._sem = asyncio.Semaphore(SEM_MAX_CONCURRENT)

    async def close(self):
        if not self.client.is_closed:
            logger.info("Closing Tidal client...")
            await self.client.aclose()

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, max=RETRY_WAIT_MAX),
        retry=retry_if_exception_type(RETRY_ERRORS),
        reraise=True
    )
    async def _request(self, method: str, endpoint: str, params=None, data=None) -> Dict[str, Any]:
        async with self._sem:
            if self.client.is_closed:
                raise RuntimeError("HTTP client closed.")
            token = await self.auth_manager.get_token(self.client)
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.api+json"}
            if data:
                headers["Content-Type"] = "application/vnd.api+json"
            url = f"{TIDAL_API_BASE_URL}{endpoint}"
            resp = await self.client.request(method, url, headers=headers, params=params, json=data)
            if resp.status_code == 429:
                ra = int(resp.headers.get("Retry-After", "5"))
                logger.warning(f"429 on {url}, sleeping {ra}s…")
                await asyncio.sleep(ra)
                resp.raise_for_status()
            resp.raise_for_status()
            if resp.status_code == 204:
                return {}
            try:
                return resp.json()
            except Exception as e:
                logger.error(f"JSON decode error for {url}: {e}")
                return {}

    async def search(self, query: str, limit=MAX_TIDAL_RESULTS_PER_SONG):
        q = quote(query[:240])
        endpoint = f"/searchResults/{q}"
        params = {"countryCode": COUNTRY_CODE, "include": "tracks,albums,artists,artworks", "limit": limit}
        return await self._request("GET", endpoint, params=params)

    async def get_track(self, track_id: str):
        endpoint = f"/tracks/{track_id}"
        params = {"countryCode": COUNTRY_CODE, "include": "albums,artists"}
        return await self._request("GET", endpoint, params=params)

    async def get_album(self, album_id: str):
        endpoint = f"/albums/{album_id}"
        params = {"countryCode": COUNTRY_CODE, "include": "coverArt,artists"}
        return await self._request("GET", endpoint, params=params)

    async def get_artwork(self, artwork_id: str):
        endpoint = f"/artworks/{artwork_id}"
        params = {"countryCode": COUNTRY_CODE}
        return await self._request("GET", endpoint, params=params)

    def _find_included_resource(self, data: Dict[str, Any], rtype: str, rid: str):
        inc = data.get("included")
        if not isinstance(inc, list):
            return None
        for item in inc:
            if isinstance(item, dict) and item.get("type") == rtype and item.get("id") == rid:
                return item
        return None

    def _get_relationship_id(self, resource: Dict[str, Any], rel: str) -> Optional[str]:
        try:
            linkage = resource["relationships"][rel]["data"]
            if isinstance(linkage, list):
                return linkage[0]["id"]
            return linkage.get("id")
        except Exception:
            return None

    async def get_tidal_track_details(self, resource: Dict[str, Any], search_resp: Dict[str, Any]):
        track_id = resource.get("id")
        attrs = resource.get("attributes", {})
        if not track_id or not attrs:
            return None

        details = {
            "tidal_track_id": track_id,
            "title": attrs.get("title"),
            "tidal_url": None,
            "artists": None,
            "album_name": None,
            "year": None,
            "tidal_cover_url": None,
        }

        # URL
        for link in attrs.get("externalLinks", []):
            if link.get("meta", {}).get("type") == "TIDAL_SHARING":
                details["tidal_url"] = link.get("href")
                break
        if not details["tidal_url"]:
            details["tidal_url"] = f"https://tidal.com/browse/track/{track_id}"

        # Artists
        names = []
        for art in resource.get("relationships", {}).get("artists", {}).get("data", []):
            aid = art.get("id")
            inc_res = self._find_included_resource(search_resp, "artists", aid)
            if inc_res:
                names.append(inc_res.get("attributes", {}).get("name"))
        details["artists"] = names or None

        # Album & cover
        alb_id = self._get_relationship_id(resource, "albums")
        if alb_id:
            alb_res = self._find_included_resource(search_resp, "albums", alb_id)
            if not alb_res:
                alb_res = (await self.get_album(alb_id)).get("data")
            if alb_res:
                aattrs = alb_res.get("attributes", {})
                details["album_name"] = aattrs.get("title")
                rd = aattrs.get("releaseDate")
                if rd and len(str(rd)) >= 4:
                    details["year"] = str(rd)[:4]
                art_id = self._get_relationship_id(alb_res, "coverArt")
                if art_id:
                    art_res = self._find_included_resource(search_resp, "artworks", art_id)
                    if not art_res:
                        art_res = (await self.get_artwork(art_id)).get("data")
                    if art_res:
                        files = art_res.get("attributes", {}).get("files", [])
                        if files:
                            sorted_files = sorted(files, key=lambda f: f.get("meta", {}).get("width", 0), reverse=True)
                            best = next((f for f in sorted_files if f.get("meta", {}).get("width",0) >= 300), sorted_files[0])
                            details["tidal_cover_url"] = best.get("href")

        if not (details["title"] and details["artists"] and details["tidal_url"]):
            return None
        return details

# --- Metadata Scoring ---
def calculate_metadata_score(s: pd.Series, t: Dict[str, Any]) -> float:
    score = 0.0; max_score = 0.0
    # Title
    max_score += 2
    if s.get("title") and t.get("title") and s["title"].lower().strip() == t["title"].lower().strip():
        score += 2
    # Artists
    max_score += 2
    sp = {a.lower() for a in s.get("artists", [])}
    td = {a.lower() for a in t.get("artists", [])}
    if sp and td:
        if sp == td:
            score += 2
        elif sp & td:
            j = len(sp & td) / len(sp | td)
            score += 1 + j
    # Album
    max_score += 1
    if s.get("album_name") and t.get("album_name") and s["album_name"].lower().strip() == t["album_name"].lower().strip():
        score += 1
    # Year
    max_score += 0.5
    try:
        y1, y2 = int(s.get("year",0)), int(t.get("year",0))
        if y1 == y2: score += 0.5
        elif abs(y1-y2) == 1: score += 0.25
    except:
        pass
    return (score / max_score) if max_score else 0.0

# --- Matching Logic ---
async def find_best_tidal_match(row: pd.Series, client: TidalApiClient) -> Optional[str]:
    title, artists, alb, year, cover, isrc = (
        row.get("title",""),
        row.get("artists",[]),
        row.get("album_name",""),
        row.get("year",""),
        row.get("spotify_cover_url",""),
        row.get("isrc",""),
    )

    # 1) ISRC
    results = None
    if isrc:
        try:
            p = {"countryCode": COUNTRY_CODE, "filter[isrc]": isrc, "include": "albums,artists,albums.coverArt"}
            d = await client._request("GET", "/tracks", params=p)
            if d.get("data"): results = d
        except:
            pass

    # 2) Metadata search
    if not results:
        q = " ".join(filter(None, [title, " ".join(artists), alb])).strip()
        if not q:
            return None
        try:
            results = await client.search(q, limit=MAX_TIDAL_RESULTS_PER_SONG)
        except:
            return None

    if not (results.get("data") or results.get("included")):
        return None

    # collect track resources from data + included
    rd = results.get("data")
    if not isinstance(rd, list): rd = []
    tracks: List[Dict[str, Any]] = []
    for item in rd:
        if isinstance(item, dict) and item.get("type") == "tracks":
            tracks.append(item)

    ri = results.get("included")
    if not isinstance(ri, list): ri = []
    for item in ri:
        if isinstance(item, dict) and item.get("type") == "tracks" and item not in tracks:
            tracks.append(item)

    tracks = tracks[:MAX_TIDAL_RESULTS_PER_SONG]
    if not tracks:
        return None

    # fetch details
    dtasks = [client.get_tidal_track_details(t, results) for t in tracks]
    raw = await asyncio.gather(*dtasks, return_exceptions=True)

    candidates = []
    llm_tasks, llm_map = [], {}
    for r in raw:
        if isinstance(r, dict):
            mscore = calculate_metadata_score(row, r)
            candidates.append((mscore, r))
            if cover and r.get("tidal_cover_url"):
                task = asyncio.create_task(compare_images_llm(cover, r["tidal_cover_url"]))
                llm_tasks.append(task); llm_map[task] = r

    llm_res = {}
    if llm_tasks:
        done, _ = await asyncio.wait(llm_tasks)
        for t in done:
            rec = llm_map[t]
            try: llm_res[rec["tidal_track_id"]] = t.result()
            except: llm_res[rec["tidal_track_id"]] = None

    scored = []
    for mscore, r in candidates:
        lid = r["tidal_track_id"]; verdict = llm_res.get(lid)
        cscore = 0
        if verdict is True:   cscore = 5
        elif verdict is False: cscore = -5
        combined = mscore + cscore
        scored.append((combined, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored and scored[0][0] > 0:
        return scored[0][1]["tidal_url"]
    return None

# --- Main ---
async def main():
    parser = argparse.ArgumentParser(description="Find Tidal URLs for Spotify tracks.")
    parser.add_argument("--sample", action="store_true", help=f"Process only first {SAMPLE_SIZE} tracks.")
    args = parser.parse_args()

    if not TIDAL_CLIENT_ID or not TIDAL_CLIENT_SECRET:
        logger.error("TIDAL credentials missing."); sys.exit(1)

    try:
        df_full = pd.read_feather(FEATHER_INPUT_PATH)
    except Exception as e:
        logger.error(f"Load failure: {e}"); sys.exit(1)

    # ensure tidal_url column
    if 'tidal_url' not in df_full.columns:
        df_full['tidal_url'] = pd.NA

    df_proc = df_full.head(SAMPLE_SIZE).copy() if args.sample else df_full.copy()
    if 'tidal_url' not in df_proc.columns:
        df_proc['tidal_url'] = pd.NA

    total = len(df_proc)
    logger.info(f"Processing {total} tracks{' (sample)' if args.sample else ''}…")

    auth_mgr = TidalAuthManager(TIDAL_CLIENT_ID, TIDAL_CLIENT_SECRET)
    client = TidalApiClient(auth_mgr)

    results_map: Dict[int, Any] = {}
    base, ext = os.path.splitext(FEATHER_OUTPUT_PATH)

    # chunked processing and intermediate saves
    for start in range(0, total, INTERMEDIATE_CHUNK):
        end = min(start + INTERMEDIATE_CHUNK, total)
        chunk = df_proc.iloc[start:end]
        logger.info(f"Chunk {start+1}–{end}")

        # schedule tasks for this chunk
        tasks = [
            asyncio.create_task(find_best_tidal_match(row, client), name=str(idx))
            for idx, row in chunk.iterrows()
        ]

        # run & progress-bar
        results = await tqdm_asyncio.gather(
            *tasks,
            desc=f"Tracks {start+1}-{end}",
            unit="track",
            ascii=True,
            return_exceptions=True
        )

        # map results
        for t, r in zip(tasks, results):
            idx = int(t.get_name())
            results_map[idx] = r if not isinstance(r, Exception) else pd.NA

        # update DataFrame and save intermediate
        if args.sample:
            df_full.loc[results_map.keys(), 'tidal_url'] = pd.Series(results_map)
            to_save = df_full
            out_file = f"{base}_sample_part{start+1}-{end}{ext}"
        else:
            df_proc.loc[results_map.keys(), 'tidal_url'] = pd.Series(results_map)
            to_save = df_proc
            out_file = f"{base}_part{start+1}-{end}{ext}"

        to_save.to_feather(out_file)
        logger.info(f"Saved intermediate results to {out_file}")

    # final save (same pattern)
    final_file = FEATHER_OUTPUT_PATH if not args.sample else f"{base}_sample{ext}"
    df_to_final = df_proc if not args.sample else df_full
    df_to_final.to_feather(final_file)
    logger.info(f"Saved final results to {final_file}")

    await client.close()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
