"""Initial EDA script for F1 datasets.

This script loads the Jolpica and OpenF1 CSV exports, merges them into a
single DataFrame and prints some basic statistics.  It is intentionally light on
visuals so it can run in constrained environments.
"""

import os
import pandas as pd


def main():
    """Perform initial EDA on the fetched F1 datasets."""
    # Map of DataFrame names to their CSV paths
    paths = {
        'circuits':      'jolpica_circuits.csv',
        'races':         'jolpica_races.csv',
        'results':       'jolpica_results.csv',
        'sprint':        'jolpica_sprint.csv',
        'qualifying':    'jolpica_qualifying.csv',
        'driverstd':     'jolpica_driverstandings.csv',
        'constructorstd':'jolpica_constructorstandings.csv',
        'status':        'jolpica_status.csv',
        'weather':       'openf1_weather.csv',
        'sessions':      'openf1_sessions.csv',
    }

    # Load each CSV into a dictionary of DataFrames, skipping any that are
    # missing.  This keeps the script runnable even if some datasets are absent.
    dfs = {}
    for name, fp in paths.items():
        if os.path.exists(fp):
            dfs[name] = pd.read_csv(fp)
        else:
            print(f"Warning: {fp} not found - skipping {name}")

    # Merge qualifying results with race metadata
    if 'qualifying' not in dfs or 'races' not in dfs:
        print("Missing qualifying or races CSVs; cannot build master DataFrame.")
        return

    df_master = dfs['qualifying'].merge(
        dfs['races'][['season', 'round', 'raceName', 'date']],
        on=['season', 'round', 'raceName'], how='left'
    )

    # Standardise qualifying column names to lower case
    df_master.rename(columns={'Q1': 'q1', 'Q2': 'q2', 'Q3': 'q3'}, inplace=True)

    # Merge in weather and session info.  First combine those two datasets on
    # 'meeting_key', then merge with the master DataFrame if possible.
    if 'weather' in dfs and 'sessions' in dfs:
        weather_sessions = dfs['weather'].merge(
            dfs['sessions'],
            on='meeting_key',
            how='left'
        )
        if 'meeting_key' in df_master.columns:
            df_master = df_master.merge(
                weather_sessions[
                    [
                        'meeting_key',
                        'air_temperature',
                        'track_temperature',
                        'session_key',
                        'circuit_short_name',
                    ]
                ],
                on='meeting_key',
                how='left'
            )
        else:
            print(
                "meeting_key not present in qualifying data; skipping weather/"
                "session merge"
            )

    # Display basic info and missing values
    df_master.info()
    print(df_master.isnull().sum())

    # Convert date columns to datetime
    if 'date' in df_master.columns:
        df_master['date'] = pd.to_datetime(df_master['date'], errors='coerce')
    if 'date_start' in df_master.columns:
        df_master['date_start'] = pd.to_datetime(df_master['date_start'], errors='coerce')

    # Convert potential numeric columns
    numeric_cols = ['q1', 'q2', 'q3', 'air_temperature', 'track_temperature']
    for col in numeric_cols:
        if col in df_master.columns:
            df_master[col] = pd.to_numeric(df_master[col], errors='coerce')

    # Basic statistics for numeric columns
    print(df_master.describe())

    # Correlation matrix for numeric fields
    corr = df_master.select_dtypes('number').corr()
    print(corr)

    # Placeholder for histograms and heatmap
    # import matplotlib.pyplot as plt
    # df_master['q3'].hist()
    # plt.show()
    # plt.matshow(corr)
    # plt.colorbar()
    # plt.show()


if __name__ == "__main__":
    main()
