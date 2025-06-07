# prepare_data.py

import pandas as pd

def main():
    # 1. Bestandspaden
    files = {
        'qual':     'jolpica_qualifying.csv',
        'races':    'jolpica_races.csv',
        'results':  'jolpica_results.csv',
        'circuits': 'jolpica_circuits.csv'
    }

    # 2. Inladen CSV's
    df_qual    = pd.read_csv(files['qual'])
    df_races   = pd.read_csv(files['races'])
    df_results = pd.read_csv(files['results'])
    df_circ    = pd.read_csv(files['circuits'])

    # 3. Hernoemen conflicterende kolommen
    df_qual    = df_qual.rename(columns={'position': 'grid_position'})
    df_results = df_results.rename(columns={'position': 'finish_position'})

    # 4. Merge kwalificatie + races metadata
    df = df_qual.merge(
        df_races[['season','round','raceName','date','Circuit.circuitId']],
        on=['season','round','raceName'],
        how='left'
    )

    # 5. Merge race-resultaten
    df = df.merge(
        df_results[['season','round','raceName','Driver.driverId','finish_position']],
        on=['season','round','raceName','Driver.driverId'],
        how='left'
    )

    # 6. Target aanmaken
    df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')
    df['top3'] = df['finish_position'] <= 3

    # 7. Q-tijden omzetten naar seconden
    def to_seconds(t):
        m, s = t.split(':')
        return int(m) * 60 + float(s)

    for col in ['Q1','Q2','Q3']:
        df[f'{col}_sec'] = pd.to_numeric(
            df[col].dropna().apply(to_seconds),
            errors='coerce'
        )

    # 8. Datum verwerken
    df['date']    = pd.to_datetime(df['date'])
    df['month']   = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday

    # 9. Impute per circuit
    for sec in ['Q1_sec','Q2_sec','Q3_sec']:
        med = df.groupby('Circuit.circuitId')[sec].transform('median')
        df[sec] = df[sec].fillna(med).fillna(df[sec].median())

    # 10. Circuit-features mergen
    df = df.merge(
        df_circ[['circuitId','circuitName',
                 'Location.lat','Location.long',
                 'Location.locality','Location.country']],
        left_on='Circuit.circuitId',
        right_on='circuitId',
        how='left'
    ).rename(columns={
        'Location.lat':'circuit_lat',
        'Location.long':'circuit_long',
        'Location.locality':'circuit_city',
        'Location.country':'circuit_country'
    })

    # 11. Rolling averages per driver
    df = df.sort_values(['Driver.driverId','date'])
    df['avg_finish_pos'] = df.groupby('Driver.driverId')['finish_position'] \
                               .transform(lambda x: x.shift().expanding().mean())
    df['avg_grid_pos']   = df.groupby('Driver.driverId')['grid_position']   \
                               .transform(lambda x: x.shift().expanding().mean())

    df['avg_finish_pos'] = df['avg_finish_pos'].fillna(df['avg_finish_pos'].median())
    df['avg_grid_pos']   = df['avg_grid_pos'].fillna(df['avg_grid_pos'].median())

    # 12. Wegschrijven
    df.to_csv('processed_data.csv', index=False)
    print(f"Verwerkte data opgeslagen in processed_data.csv — records: {len(df)}, top3: {df['top3'].sum()}")

if __name__ == '__main__':
    main()
