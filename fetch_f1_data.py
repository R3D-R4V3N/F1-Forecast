import requests
import pandas as pd
import time
from typing import Dict, List, Optional

# Base URLs for the two APIs
OPENF1_BASE = "https://api.openf1.org/v1"
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"


def fetch_json(url: str, params: Optional[Dict] = None, retries: int = 3, backoff: float = 1.0) -> Optional[Dict]:
    """Fetch JSON with simple retry logic for 5xx errors and graceful skip on 404."""
    if params is None:
        params = {}
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 404:
                print(f"Skipping {url}: 404 Not Found")
                return None
            if resp.status_code >= 500:
                if attempt < retries - 1:
                    time.sleep(backoff * (2 ** attempt))
                    continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
                continue
            print(f"Failed to fetch {url}: {exc}")
            return None
    return None


def fetch_openf1_data() -> None:
    """Fetch latest OpenF1 session and weather data and write to CSVs."""
    sess = fetch_json(f"{OPENF1_BASE}/sessions", {"meeting_key": "latest", "session_key": "latest"})
    if not sess:
        print("No latest session found from OpenF1")
        return

    meeting_key = sess[0]["meeting_key"]
    session_key = sess[0]["session_key"]

    weather = fetch_json(f"{OPENF1_BASE}/weather", {"meeting_key": meeting_key, "session_key": session_key})
    if weather is not None:
        df_weather = pd.json_normalize(weather)
        df_weather["source"] = "openf1_weather"
        df_weather.to_csv("openf1_weather.csv", index=False)
        print(f"Wrote {len(df_weather)} rows to openf1_weather.csv")
    else:
        print("No weather data returned")

    sessions = fetch_json(f"{OPENF1_BASE}/sessions", {"meeting_key": meeting_key})
    if sessions is not None:
        df_sessions = pd.json_normalize(sessions)
        df_sessions["source"] = "openf1_sessions"
        df_sessions.to_csv("openf1_sessions.csv", index=False)
        print(f"Wrote {len(df_sessions)} rows to openf1_sessions.csv")
    else:
        print("No session list returned")


def fetch_paginated(url: str, params: Optional[Dict] = None) -> List[Dict]:
    """Retrieve all pages for a Jolpica endpoint."""
    if params is None:
        params = {}
    pages = []
    offset = 0
    while True:
        params["offset"] = offset
        data = fetch_json(url, params)
        if data is None:
            break
        pages.append(data)
        mr = data.get("MRData", {})
        limit = int(mr.get("limit", 0))
        total = int(mr.get("total", 0))
        offset += limit
        if offset >= total or limit == 0:
            break
    return pages


# Mapping of Jolpica endpoints to record_path and meta for pandas.json_normalize
JOLPICA_ENDPOINTS = {
    "circuits": ("{season}/circuits/", ["MRData", "CircuitTable", "Circuits"], ["season"]),
    "races": ("{season}/races/", ["MRData", "RaceTable", "Races"], ["season"]),
    "results": ("{season}/results/", ["MRData", "RaceTable", "Races", "Results"], ["season", "round", "raceName"]),
    "sprint": ("{season}/sprint/", ["MRData", "RaceTable", "Races", "SprintResults"], ["season", "round", "raceName"]),
    "qualifying": ("{season}/qualifying/", ["MRData", "RaceTable", "Races", "QualifyingResults"], ["season", "round", "raceName"]),
    "driverstandings": ("{season}/driverstandings/", ["MRData", "StandingsTable", "StandingsLists", "DriverStandings"], ["season"]),
    "constructorstandings": ("{season}/constructorstandings/", ["MRData", "StandingsTable", "StandingsLists", "ConstructorStandings"], ["season"]),
    "status": ("{season}/status/", ["MRData", "StatusTable", "Statuses"], ["season"]),
}


def fetch_jolpica_data() -> None:
    """Fetch data from Jolpica API for seasons 2022 onward and write each endpoint to separate CSV."""
    seasons_resp = fetch_json(f"{JOLPICA_BASE}/seasons/")
    if not seasons_resp:
        print("Could not retrieve seasons from Jolpica")
        return

    seasons_list = [s["season"] for s in seasons_resp.get("MRData", {}).get("SeasonTable", {}).get("Seasons", [])]
    seasons = [s for s in seasons_list if int(s) >= 2022]
    if not seasons:
        print("No seasons >= 2022 found")
        return

    # Accumulate DataFrames for each endpoint
    endpoint_dfs: Dict[str, List[pd.DataFrame]] = {ep: [] for ep in JOLPICA_ENDPOINTS}

    for season in seasons:
        for ep, (path, record_path, meta) in JOLPICA_ENDPOINTS.items():
            url = f"{JOLPICA_BASE}/" + path.format(season=season)
            for page in fetch_paginated(url):
                # Ensure season metadata in case it's not present in page structure
                page.setdefault("season", season)
                df = pd.json_normalize(page, record_path=record_path, meta=meta, errors="ignore")
                df["source"] = f"jolpica_{ep}"
                endpoint_dfs[ep].append(df)

    # Write DataFrames to CSV
    for ep, dfs in endpoint_dfs.items():
        if dfs:
            out_df = pd.concat(dfs, ignore_index=True, sort=False)
            filename = f"jolpica_{ep}.csv"
            out_df.to_csv(filename, index=False)
            print(f"Wrote {len(out_df)} rows to {filename}")
        else:
            print(f"No data for {ep}")


def main() -> None:
    fetch_openf1_data()
    fetch_jolpica_data()


if __name__ == "__main__":
    main()
