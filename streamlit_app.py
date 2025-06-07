# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import altair as alt
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

# Configure the Streamlit page
st.set_page_config(
    page_title="F1 Top-3 Predictie Dashboard",
    page_icon="🏎️",
    layout="wide",
)

# Load model pipeline and processed data
@st.cache_data
def load_pipeline():
    return joblib.load('f1_top3_pipeline.joblib')

@st.cache_data
def load_data():
    df = pd.read_csv('processed_data.csv', parse_dates=['date'])
    return df

pipeline = load_pipeline()
df = load_data()

# Exacte featurekolommen voor het model
FEATURE_COLS = [
    'grid_position','Q1_sec','Q2_sec','Q3_sec',
    'month','weekday','avg_finish_pos','avg_grid_pos','avg_const_finish',
    'air_temperature','track_temperature','grid_diff','Q3_diff','grid_temp_int',
    'circuit_country','circuit_city'
]

@st.cache_data
def compute_accuracy(model, data):
    X_full = data[FEATURE_COLS]
    y_true = data['top3']
    y_pred = model.predict(X_full)
    return (y_pred == y_true).mean()

accuracy = compute_accuracy(pipeline, df)
st.sidebar.metric("Model accuracy", f"{accuracy:.2%}")

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
X = df_race[FEATURE_COLS]

# Predict
proba = pipeline.predict_proba(X)[:,1]
df_race['top3_proba'] = proba

# Display predictions
preds = (
    df_race[['Driver.driverId', 'top3_proba']]
    .sort_values('top3_proba', ascending=False)
    .drop_duplicates('Driver.driverId')
)
top_preds = preds.head(3)
display_df = top_preds.rename(
    columns={"Driver.driverId": "Coureur", "top3_proba": "Kans"}
)
st.subheader(
    f"Top-3 voorspellingen voor {selected_race} {selected_season}"
)
st.table(display_df)

# Bar chart of probabilities
chart = (
    alt.Chart(display_df)
    .mark_bar()
    .encode(
        x=alt.X("Coureur:N", sort="-y"),
        y=alt.Y("Kans:Q", title="Kans op top-3"),
        tooltip=["Coureur", alt.Tooltip("Kans", format=".2f")],
    )
)
st.altair_chart(chart, use_container_width=True)

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
