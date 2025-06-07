# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# Load model pipeline and processed data
@st.cache(allow_output_mutation=True)
def load_pipeline():
    return joblib.load('f1_top3_pipeline.joblib')

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('processed_data.csv', parse_dates=['date'])
    return df

pipeline = load_pipeline()
df = load_data()

st.title("F1 Top-3 Finish Predictie Dashboard")

# Sidebar: seizoen en race selecteren
st.season = int(df['season'].max())
seasons = sorted(df['season'].unique())
selected_season = st.sidebar.selectbox('Selecteer seizoen', seasons, index=len(seasons)-1)

races = df[df['season']==selected_season]['raceName'].unique()
selected_race = st.sidebar.selectbox('Selecteer race', races)

# Filter data voor selectie
df_race = df[(df['season']==selected_season) & (df['raceName']==selected_race)]

# Prepare features
feature_cols = [
    'grid_position','Q1_sec','Q2_sec','Q3_sec',
    'month','weekday','avg_finish_pos','avg_grid_pos','avg_const_finish',
    'air_temperature','track_temperature','grid_diff','Q3_diff','grid_temp_int',
    'circuit_country','circuit_city'
]
X = df_race[feature_cols]

# Predict
proba = pipeline.predict_proba(X)[:,1]
df_race['top3_proba'] = proba

# Display top 3 predictions
top3 = (
    df_race[['Driver.driverId','top3_proba']]
    .sort_values('top3_proba', ascending=False)
    .drop_duplicates('Driver.driverId')
    .head(3)
)
st.subheader(f"Top-3 voorspellingen voor {selected_race} {selected_season}")
st.table(top3.rename(columns={'Driver.driverId':'Coureur','top3_proba':'Kans'}))

# Show performance metrics for train/test
st.subheader("Model Performance (Train/Test)")
# Load performance report if saved, else omit
try:
    report = pd.read_csv('model_performance.csv', index_col=0)
    st.dataframe(report)
except FileNotFoundError:
    st.write("Modelperformancerapport niet gevonden. Run train_model.py en exporteer naar model_performance.csv.")

# ROC Curve placeholder
st.subheader("ROC Curve")
fig, ax = plt.subplots()
# Dummy curve as example
fpr = [0, 0.1, 0.5, 1]
tpr = [0, 0.7, 0.9, 1]
ax.plot(fpr, tpr)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
st.pyplot(fig)

# --------------------------------------------
# Extra visualisaties en feature importance
# --------------------------------------------

# Bar chart van de top-3 kansen
fig, ax = plt.subplots()
ax.bar(top3['Driver.driverId'], top3['top3_proba'], color='skyblue')
ax.set_ylabel('Kans op top 3')
ax.set_xlabel('Coureur')
ax.set_title('Top-3 voorspellingen')
st.pyplot(fig)

# Histogram van alle voorspelde kansen voor de geselecteerde race
st.subheader('Verdeling voorspelde kansen')
fig, ax = plt.subplots()
ax.hist(df_race['top3_proba'], bins=20, color='gray', edgecolor='black')
ax.set_xlabel('Voorspelde kans')
ax.set_ylabel('Aantal rijders')
st.pyplot(fig)

# Scatterplot grid positie vs voorspelde kans
st.subheader('Grid positie vs voorspelde kans')
fig, ax = plt.subplots()
ax.scatter(df_race['grid_position'], df_race['top3_proba'], alpha=0.7)
ax.set_xlabel('Startpositie')
ax.set_ylabel('Kans op top 3')
st.pyplot(fig)

# Correlatiematrix van numerieke features
st.subheader('Correlatiematrix (numerieke features)')
numeric_feats = [
    'grid_position','Q1_sec','Q2_sec','Q3_sec',
    'month','weekday','avg_finish_pos','avg_grid_pos','avg_const_finish',
    'air_temperature','track_temperature','grid_diff','Q3_diff','grid_temp_int'
]
corr = df_race[numeric_feats].corr()
fig, ax = plt.subplots(figsize=(8,6))
cax = ax.matshow(corr, cmap='coolwarm')
fig.colorbar(cax)
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)
st.pyplot(fig)

# Feature importance grafiek indien beschikbaar
st.subheader('Feature importance')
try:
    feat_names = pipeline.named_steps['pre'].get_feature_names_out()
    importances = pipeline.named_steps['clf'].feature_importances_
    fi_df = pd.DataFrame({
        'feature': feat_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(15)
    fig, ax = plt.subplots()
    ax.barh(fi_df['feature'][::-1], fi_df['importance'][::-1])
    ax.set_xlabel('Belang')
    ax.set_ylabel('Feature')
    st.pyplot(fig)
except Exception:
    st.write('Feature importance niet beschikbaar voor dit model.')
