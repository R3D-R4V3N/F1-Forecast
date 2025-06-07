# infer.py

import pandas as pd
import joblib

# Inference script: toont top-3 voorspellingen voor de nieuwste race in het huidige seizoen

def inference():
    # 1. Laad de laatst verwerkte data
    df = pd.read_csv('processed_data.csv', parse_dates=['date'])

    # 2. Bepaal huidig seizoen en nieuwste race datum
    current_season = df['season'].max()
    latest_date = df[df['season'] == current_season]['date'].max()

    # 3. Filter op huidige seizoen en laatste race
    df_race = df[(df['season'] == current_season) & (df['date'] == latest_date)]

    # 4. Laad de getrainde pipeline
    pipeline = joblib.load('f1_top3_pipeline.joblib')

    # 5. Selecteer exacte featurekolommen
    feature_cols = [
        'grid_position','Q1_sec','Q2_sec','Q3_sec',
        'month','weekday','avg_finish_pos','avg_grid_pos','avg_const_finish',
        'air_temperature','track_temperature','grid_diff','Q3_diff','grid_temp_int',
        'circuit_country','circuit_city'
    ]
    X = df_race[feature_cols]

    # 6. Voorspel probabilities en pas threshold toe
    proba = pipeline.predict_proba(X)[:, 1]
    df_race['top3_proba'] = proba
    df_race['top3_pred']  = proba >= 0.41  # optimale threshold

    # 7. Sorteer en pak de drie unieke coureurs met hoogste kans
    top3 = (
        df_race[['Driver.driverId','raceName','top3_proba']]
        .sort_values('top3_proba', ascending=False)
        .drop_duplicates(subset=['Driver.driverId'])
        .head(3)
    )

    print(f"Top-3 voorspellingen voor seizoen {current_season}, race op {latest_date.date()}:\n")
    print(top3.to_string(index=False))

if __name__ == '__main__':
    inference()
