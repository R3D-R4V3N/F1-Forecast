# eda_f1.py
import os
import pandas as pd

def main():
    """Perform initial EDA on the fetched F1 datasets."""
    # Paths to CSVs
    paths = {
        'circuits':       'jolpica_circuits.csv',
        'races':          'jolpica_races.csv',
        'results':        'jolpica_results.csv',
        'sprint':         'jolpica_sprint.csv',
        'qualifying':     'jolpica_qualifying.csv',
        'driverstd':      'jolpica_driverstandings.csv',
        'constructorstd': 'jolpica_constructorstandings.csv',
        'weather':        'openf1_weather.csv',
        'sessions':       'openf1_sessions.csv',
    }

    # Load dataframes
    dfs = {}
    for name, file in paths.items():
        if os.path.exists(file):
            dfs[name] = pd.read_csv(file)
        else:
            print(f"Warning: {file} not found, skipping '{name}'")

    # 1) Merge qualifying with race metadata
    df_master = dfs['qualifying'].merge(
        dfs['races'][['season','round','raceName','date']],
        on=['season','round','raceName'], how='left'
    )
    print("Merged qualifying and races:", df_master.shape)

    # 2) Merge weather if meeting_key exists
    if 'meeting_key' in df_master.columns and 'meeting_key' in dfs.get('weather', pd.DataFrame()).columns:
        df_master = df_master.merge(
            dfs['weather'][['meeting_key','air_temperature','track_temperature']],
            on='meeting_key', how='left'
        )
        print("Merged weather data:", df_master.shape)

    # 3) Merge sessions if season & round exist
    sess_df = dfs.get('sessions', pd.DataFrame())
    if 'season' in sess_df.columns and 'round' in sess_df.columns:
        df_master = df_master.merge(
            sess_df[['season','round','session_key','circuit_short_name']],
            on=['season','round'], how='left'
        )
        print("Merged session data:", df_master.shape)

    # 4) Display structure and missing values
    print(df_master.info())
    print("Missing values by column:\n", df_master.isnull().sum())

    # 5) Convert date columns
    for col in ['date','date_start']:
        if col in df_master.columns:
            df_master[col] = pd.to_datetime(df_master[col], errors='coerce')

    # 6) Convert numeric columns
    for col in ['q1','q2','q3','air_temperature','track_temperature']:
        if col in df_master.columns:
            df_master[col] = pd.to_numeric(df_master[col], errors='coerce')

    # 7) Basic statistics and correlation
    print("\nDescriptive statistics:\n", df_master.describe())
    corr = df_master.select_dtypes(include='number').corr()
    print("\nCorrelation matrix:\n", corr)

    # 8) Placeholder for visualizations
    # import matplotlib.pyplot as plt
    # df_master[['q3','air_temperature']].hist(bins=20)
    # plt.show()
    # plt.matshow(corr)
    # plt.colorbar()
    # plt.show()

if __name__ == '__main__':
    main()
