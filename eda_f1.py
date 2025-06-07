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

    # Load each CSV into a dictionary of DataFrames
    dfs = {name: pd.read_csv(fp) for name, fp in paths.items()}

    # Merge qualifying results with race metadata
    df_master = dfs['qualifying'].merge(
        dfs['races'][['season', 'round', 'raceName', 'date']],
        on=['season', 'round', 'raceName'], how='left'
    )

    # Merge in weather and session info using meeting_key
    df_master = df_master.merge(
        dfs['weather'][['meeting_key', 'air_temperature', 'track_temperature']],
        on='meeting_key', how='left'
    ).merge(
        dfs['sessions'][['meeting_key', 'session_key', 'circuit_short_name']],
        on='meeting_key', how='left'
    )

    # Display basic info and missing values
    print(df_master.info())
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
