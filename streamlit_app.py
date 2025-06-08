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


# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

st.title("F1 Top-3 Finish Predictie Dashboard")

# Feature preparation function
def prepare_features(df_sub):
    df_sub = df_sub.sort_values('date')
    df_sub['avg_finish_pos'] = df_sub.groupby('Driver.driverId')['finish_position'] \
                             .transform(lambda x: x.shift().expanding().mean())
    df_sub['avg_grid_pos']   = df_sub.groupby('Driver.driverId')['grid_position']   \
                             .transform(lambda x: x.shift().expanding().mean())
    df_sub['grid_diff']      = df_sub['avg_grid_pos'] - df_sub['grid_position']
    df_sub['driver_avg_Q3']  = df_sub.groupby('Driver.driverId')['Q3_sec']        \
                             .transform(lambda x: x.shift().expanding().mean())
    df_sub['Q3_diff']        = df_sub['driver_avg_Q3'] - df_sub['Q3_sec']
    df_sub['grid_temp_int']  = df_sub['grid_position'] * df_sub['track_temperature']
    return df_sub

# Load data and pipeline with caching
def load_data():
    return pd.read_csv('processed_data.csv', parse_dates=['date'])

def load_pipeline():
    return joblib.load('f1_top3_pipeline.joblib')

df = load_data()
pipeline = load_pipeline()

# Sidebar for season and race selection
seasons = sorted(df['season'].unique())
selected_season = st.sidebar.selectbox('Selecteer seizoen', seasons, index=len(seasons)-1)
races = df[df['season']==selected_season]['raceName'].unique()
selected_race  = st.sidebar.selectbox('Selecteer race', races)
selected_date  = df[(df['season']==selected_season)&(df['raceName']==selected_race)]['date'].max()

# Split data for time-series evaluation
df_train = df[df['date'] < selected_date]
df_test  = df[df['date'] == selected_date]

# Prepare features
df_train = prepare_features(df_train)
feature_cols = [
    'grid_position','Q1_sec','Q2_sec','Q3_sec',
    'month','weekday','avg_finish_pos','avg_grid_pos','avg_const_finish',
    'air_temperature','track_temperature','grid_diff','Q3_diff','grid_temp_int',
    'circuit_country','circuit_city'
]

# Retrain on training data
X_train = df_train[feature_cols]
y_train = df_train['top3']
pipeline.fit(X_train, y_train)

# Prepare test features and predict
df_test = prepare_features(pd.concat([df_train, df_test]))
df_test = df_test[df_test['date']==selected_date]
X_test = df_test[feature_cols]
y_test = df_test['top3']
proba = pipeline.predict_proba(X_test)[:,1]

# Top-3 voorspellingen
# Voeg top3_proba en top3_pred toe aan df_test
# (zonder onbedoelde indentaties)
df_test['top3_proba'] = proba
df_test['top3_pred']  = proba >= 0.41
top3 = (
     df_test[['Driver.driverId','top3_proba']]
     .sort_values('top3_proba', ascending=False)
     .drop_duplicates('Driver.driverId')
     .head(3)
 )
st.subheader(f"Top-3 voorspellingen voor {selected_race} {selected_season}")
st.table(top3.rename(columns={'Driver.driverId':'Coureur','top3_proba':'Kans'}))

# Performance metrics
# Zorg dat y_pred gedefinieerd is
y_pred = pipeline.predict(X_test)
st.subheader("Model Performance voor geselecteerde race")
st.write(f"ROC AUC: {roc_auc_score(y_test, proba):.3f}")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).T)
y_pred = pipeline.predict(X_test)
st.subheader("Model Performance voor geselecteerde race")
st.write(f"ROC AUC: {roc_auc_score(y_test, proba):.3f}")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).T)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, proba)
fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC curve')
st.pyplot(fig)

# Precision-Recall curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, proba)
fig2, ax2 = plt.subplots()
ax2.plot(recall_vals, precision_vals)
ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.set_title('PR curve')
st.pyplot(fig2)
