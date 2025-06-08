import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

# 1) Data laden
df = pd.read_csv('processed_data.csv', parse_dates=['date'])
df = df.sort_values('date')

# 2) Feature‐lijst
numeric_feats = [
    'grid_position','Q1_sec','Q2_sec','Q3_sec',
    'month','weekday','avg_finish_pos','avg_grid_pos','avg_const_finish',
    'air_temperature','track_temperature','grid_diff','Q3_diff','grid_temp_int'
]
categorical_feats = ['circuit_country','circuit_city']
feature_cols = numeric_feats + categorical_feats

# 3) Preprocessing pipeline
numeric_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale',  StandardScaler())
])
cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
pre = ColumnTransformer([
    ('num', numeric_pipe, numeric_feats),
    ('cat', cat_pipe, categorical_feats)
])

# 4) Walk‐forward backtest
dates = df['date'].unique()
all_preds = []
all_trues = []

for current_date in dates[1:]:  # skip first date, want train data
    train = df[df['date'] < current_date]
    test  = df[df['date'] == current_date]
    if len(test) == 0:
        continue

    X_train = pre.fit_transform(train[feature_cols])
    y_train = train['top3'].astype(int)
    X_test  = pre.transform(test[feature_cols])
    y_test  = test['top3'].astype(int)

    # 5) Train een simpel model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6) Predict proba
    proba = model.predict_proba(X_test)[:,1]
    all_preds.append(proba)
    all_trues.append(y_test.values)

# 7) Metrics over alle datapunten
y_pred_all = np.concatenate(all_preds)
y_true_all = np.concatenate(all_trues)
print("Time‐series backtest ROC AUC:", roc_auc_score(y_true_all, y_pred_all))
