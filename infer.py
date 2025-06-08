# infer.py

import pandas as pd
import joblib

# Inference script: use only data up to the chosen race date to predict top-3 finishers

def inference_for_date(cutoff_date):
    # 1. Load all processed data
    df_full = pd.read_csv('processed_data.csv', parse_dates=['date'])
    # 2. Keep only data up to and including cutoff_date
    df = df_full[df_full['date'] <= cutoff_date].copy()
    # 3. Sort by date to ensure rolling features use past data
    df = df.sort_values('date')

    # 4. Recompute rolling and interaction features on this subset
    df['avg_finish_pos'] = df.groupby('Driver.driverId')['finish_position'] \
                             .transform(lambda x: x.shift().expanding().mean())
    df['avg_grid_pos']   = df.groupby('Driver.driverId')['grid_position']   \
                             .transform(lambda x: x.shift().expanding().mean())
    df['grid_diff']      = df['avg_grid_pos'] - df['grid_position']
    df['driver_avg_Q3']  = df.groupby('Driver.driverId')['Q3_sec']        \
                             .transform(lambda x: x.shift().expanding().mean())
    df['Q3_diff']        = df['driver_avg_Q3'] - df['Q3_sec']
    df['grid_temp_int']  = df['grid_position'] * df['track_temperature']

    # 5. Select only the rows of the cutoff_date for testing
    df_test = df[df['date'] == cutoff_date].copy()

    # 6. Load the serialized pipeline
    pipeline = joblib.load('f1_top3_pipeline.joblib')

    # 7. Feature columns exactly as trained
    feature_cols = [
        'grid_position','Q1_sec','Q2_sec','Q3_sec',
        'month','weekday','avg_finish_pos','avg_grid_pos','avg_const_finish',
        'air_temperature','track_temperature','grid_diff','Q3_diff','grid_temp_int',
        'circuit_country','circuit_city'
    ]
    X_test = df_test[feature_cols]

    # 8. Predict probabilities and apply threshold
    proba = pipeline.predict_proba(X_test)[:, 1]
    df_test['top3_proba'] = proba
    df_test['top3_pred']  = proba >= 0.41

    # 9. Show top-3 unique drivers
    top3 = (
        df_test[['Driver.driverId','raceName','top3_proba']]
        .sort_values('top3_proba', ascending=False)
        .drop_duplicates('Driver.driverId')
        .head(3)
    )

    print(f"Top-3 voorspellingen voor race op {cutoff_date.date()}:\n")
    print(top3.to_string(index=False))

if __name__ == '__main__':
    # Replace with desired race date for prediction/backtest
    inference_for_date(pd.Timestamp('2025-06-01'))