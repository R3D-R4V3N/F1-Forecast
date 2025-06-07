import requests
import pandas as pd
import time
from typing import Dict, List, Iterator, Optional

# Base URLs
OPENF1_BASE = "https://api.openf1.org/v1"
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"

# Rate limiting parameters
RATE_LIMIT_SLEEP = 0.3  # seconds between requests (~3 req/sec burst)
# Jolpica API limits: 500 requests/hour => track with counter
HOUR_LIMIT = 500
CALL_COUNT = 0
WINDOW_START = time.time()

# Minimum season to fetch
MIN_SEASON = 2022


def fetch_json(url: str, params: Optional[Dict] = None, retries: int = 3, backoff: float = 1.0) -> Optional[Dict]:
    """Fetch JSON with retry logic: retry on 5xx/429 with backoff, skip on 404."""
    global CALL_COUNT, WINDOW_START
    params = params or {}
    for attempt in range(1, retries + 1):
        # Enforce hourly limit
        elapsed = time.time() - WINDOW_START
        if elapsed >= 3600:
            WINDOW_START = time.time()
            CALL_COUNT = 0
        if CALL_COUNT >= HOUR_LIMIT - 10:
            sleep_time = 3600 - elapsed + 5
            print(f"Approaching hourly limit. Sleeping for {int(sleep_time)}s...")
            time.sleep(sleep_time)
            WINDOW_START = time.time()
            CALL_COUNT = 0

        try:
            resp = requests.get(url, params=params, timeout=10)
            CALL_COUNT += 1
            if resp.status_code == 404:
                print(f"404 Not Found for {url}, skipping.")
                return None
            if resp.status_code == 429:
                wait = 60 * attempt
                print(f"429 Rate limited. Sleeping for {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                print(f"{resp.status_code} error on {url}, attempt {attempt}/{retries}")
                time.sleep(backoff * (2 ** (attempt - 1)))
                continue
            resp.raise_for_status()
            data = resp.json()
            time.sleep(RATE_LIMIT_SLEEP)
            return data
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}, attempt {attempt}/{retries}")
            time.sleep(backoff * (2 ** (attempt - 1)))
    print(f"Failed to fetch {url} after {retries} attempts.")
    return None


def fetch_paginated(url: str) -> Iterator[Dict]:
    """Yield all pages of an Ergast endpoint with throttling."""
    offset = 0
    limit = 30
    while True:
        page = fetch_json(url, params={"offset": offset, "limit": limit})
        if not page:
            break
        yield page
        total = int(page.get("MRData", {}).get("total", 0))
        offset += limit
        if offset >= total:
            break


def fetch_openf1_data():
    """Fetch and save OpenF1 weather and sessions."""
    sess = fetch_json(f"{OPENF1_BASE}/sessions", {"meeting_key": "latest", "session_key": "latest"})
    if not sess:
        print("No OpenF1 session.")
        return
    mk, sk = sess[0]["meeting_key"], sess[0]["session_key"]

    weather = fetch_json(f"{OPENF1_BASE}/weather", {"meeting_key": mk, "session_key": sk})
    if weather:
        pd.json_normalize(weather).to_csv("openf1_weather.csv", index=False)
        print("Wrote openf1_weather.csv")

    sessions = fetch_json(f"{OPENF1_BASE}/sessions", {"meeting_key": mk})
    if sessions:
        pd.json_normalize(sessions).to_csv("openf1_sessions.csv", index=False)
        print("Wrote openf1_sessions.csv")


def fetch_jolpica_data():
    """Fetch and save Jolpica endpoints for seasons >= MIN_SEASON."""
    # Get seasons
    seasons_all: List[str] = []
    for page in fetch_paginated(f"{JOLPICA_BASE}/seasons/"):
        for s in page.get("MRData", {}).get("SeasonTable", {}).get("Seasons", []):
            season = s.get("season")
            if season and season.isdigit():
                seasons_all.append(season)
    seasons = [s for s in seasons_all if int(s) >= MIN_SEASON]
    if not seasons:
        print(f"No seasons >= {MIN_SEASON} found.")
        return

    # Define endpoints
    simple_eps = {
        "circuits": ("CircuitTable", "Circuits"),
        "races": ("RaceTable", "Races"),
        "driverstandings": ("StandingsTable", "StandingsLists"),
        "constructorstandings": ("StandingsTable", "StandingsLists"),
        "status": ("StatusTable", "Statuses"),
    }
    nested_eps = {
        "results": "Results",
        "sprint": "SprintResults",
        "qualifying": "QualifyingResults",
    }

    # Process simple endpoints
    for ep, (table, record) in simple_eps.items():
        frames: List[pd.DataFrame] = []
        print(f"Fetching {ep} for seasons {seasons[0]} to {seasons[-1]}...")
        for season in seasons:
            url = f"{JOLPICA_BASE}/{season}/{ep}/"
            for page in fetch_paginated(url):
                data = page.get("MRData", {}).get(table, {})
                recs = data.get(record, [])
                if not recs:
                    continue
                df = pd.json_normalize(recs)
                df['season'] = season
                if record in ['Races', 'DriverStandings', 'ConstructorStandings', 'Statuses'] and 'round' in df.columns:
                    df['round'] = df['round']
                df['source'] = ep
                frames.append(df)
        if frames:
            out_name = f"jolpica_{ep}.csv"
            pd.concat(frames, ignore_index=True).to_csv(out_name, index=False)
            print(f"Wrote {out_name}")

    # Process nested endpoints
    for ep, key in nested_eps.items():
        frames: List[pd.DataFrame] = []
        print(f"Fetching nested {ep} for seasons {seasons[0]} to {seasons[-1]}...")
        for season in seasons:
            url = f"{JOLPICA_BASE}/{season}/{ep}/"
            for page in fetch_paginated(url):
                races = page.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                for r in races:
                    recs = r.get(key, [])
                    if not recs:
                        continue
                    df = pd.json_normalize(recs)
                    df['season'] = r.get('season', season)
                    df['round'] = r.get('round')
                    df['raceName'] = r.get('raceName')
                    df['source'] = ep
                    frames.append(df)
        if frames:
            out_name = f"jolpica_{ep}.csv"
            pd.concat(frames, ignore_index=True).to_csv(out_name, index=False)
            print(f"Wrote {out_name}")


def main():
    fetch_openf1_data()
    fetch_jolpica_data()


if __name__ == '__main__':
    main()
