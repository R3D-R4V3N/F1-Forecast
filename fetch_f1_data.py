import requests
import pandas as pd
import time
from typing import List, Dict, Optional

OPENF1_BASE = "https://api.openf1.org/v1"
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"


def get_json_with_retry(
    url: str, params: Optional[Dict] = None, retries: int = 3, backoff: float = 1.0
) -> Optional[Dict]:
    if params is None:
        params = {}
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                wait = resp.headers.get("Retry-After")
                delay = float(wait) if wait and wait.isdigit() else backoff
                time.sleep(delay)
                if attempt < retries - 1:
                    continue
                return None
            if resp.status_code >= 500:
                if attempt < retries - 1:
                    time.sleep(backoff * (2 ** attempt))
                    continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))
    return None


def normalize_and_tag(data: Dict, record_path: List[str], source: str) -> pd.DataFrame:
    df = pd.json_normalize(data, record_path=record_path, errors="ignore")
    df["source"] = source
    return df


RECORD_PATHS = {
    "seasons": ["MRData", "SeasonTable", "Seasons"],
    "circuits": ["MRData", "CircuitTable", "Circuits"],
    "races": ["MRData", "RaceTable", "Races"],
    "constructors": ["MRData", "ConstructorTable", "Constructors"],
    "drivers": ["MRData", "DriverTable", "Drivers"],
    "results": ["MRData", "RaceTable", "Races"],
    "sprint": ["MRData", "RaceTable", "Races"],
    "qualifying": ["MRData", "RaceTable", "Races"],
    "driverstandings": ["MRData", "StandingsTable", "StandingsLists"],
    "constructorstandings": ["MRData", "StandingsTable", "StandingsLists"],
    "status": ["MRData", "StatusTable", "Status"],
}


def fetch_paginated(url: str, params: Optional[Dict] = None) -> List[Dict]:
    if params is None:
        params = {}
    pages = []
    offset = 0
    while True:
        params["offset"] = offset
        data = get_json_with_retry(url, params=params)
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


def fetch_openf1_data() -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    sess = get_json_with_retry(
        f"{OPENF1_BASE}/sessions", params={"meeting_key": "latest", "session_key": "latest"}
    )
    if not sess:
        return dfs
    meeting_key = sess[0]["meeting_key"]
    session_key = sess[0]["session_key"]
    weather = get_json_with_retry(
        f"{OPENF1_BASE}/weather",
        params={"meeting_key": meeting_key, "session_key": session_key},
    )
    if weather is not None:
        dfs.append(pd.json_normalize(weather).assign(source="openf1_weather"))
    sessions = get_json_with_retry(
        f"{OPENF1_BASE}/sessions", params={"meeting_key": meeting_key}
    )
    if sessions is not None:
        dfs.append(pd.json_normalize(sessions).assign(source="openf1_sessions"))
    return dfs


def fetch_jolpica_data() -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    season_pages = fetch_paginated(f"{JOLPICA_BASE}/seasons/")
    seasons: List[str] = []
    for page in season_pages:
        df = normalize_and_tag(page, RECORD_PATHS["seasons"], "jolpica_seasons")
        dfs.append(df)
        seasons.extend(
            [s["season"] for s in page.get("MRData", {}).get("SeasonTable", {}).get("Seasons", [])]
        )
    seasons = [s for s in seasons if int(s) >= 2022]
    endpoints = [
        "circuits",
        "races",
        "constructors",
        "drivers",
        "results",
        "sprint",
        "qualifying",
        "driverstandings",
        "constructorstandings",
        "status",
    ]
    for season in seasons:
        for ep in endpoints:
            url = f"{JOLPICA_BASE}/{season}/{ep}/"
            for page in fetch_paginated(url):
                df = normalize_and_tag(page, RECORD_PATHS[ep], f"jolpica_{ep}")
                dfs.append(df)
    return dfs


def main() -> None:
    dfs = fetch_openf1_data() + fetch_jolpica_data()
    if not dfs:
        print("No data fetched.")
        return
    master_df = pd.concat(dfs, ignore_index=True, sort=False)
    master_df.to_csv("f1_data.csv", index=False)
    print("Wrote", len(master_df), "rows to f1_data.csv")


if __name__ == "__main__":
    main()
