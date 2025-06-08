import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    make_scorer,
    roc_auc_score,
)

def main(export_csv=True, csv_path="model_performance.csv"):
    """Voert nested cross-validation uit en exporteert optioneel de resultaten."""

    # 1. Data laden
    df = pd.read_csv('processed_data.csv')
    numeric_feats = [
        'grid_position','Q1_sec','Q2_sec','Q3_sec',
        'month','weekday','avg_finish_pos','avg_grid_pos','avg_const_finish',
        'air_temperature','track_temperature'
    ]
    categorical_feats = ['circuit_country','circuit_city']
    X = df[numeric_feats + categorical_feats]
    y = df['top3']

    # 2. Preprocessing pipelines
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore'))
    ])
    pre = ColumnTransformer([
        ('num', num_pipe, numeric_feats),
        ('cat', cat_pipe, categorical_feats)
    ])

    # 3. Full pipeline + param grid
    pipe = Pipeline([
        ('pre', pre),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 5],
        'clf__min_samples_split': [2, 5]
    }

    # 4. Inner CV for tuning
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    grid = GridSearchCV(pipe, param_grid,
                        scoring=make_scorer(roc_auc_score),
                        cv=inner_cv, n_jobs=-1)

    # 5. Outer CV for evaluation
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    scores = cross_val_score(grid, X, y,
                             scoring=make_scorer(roc_auc_score),
                             cv=outer_cv, n_jobs=-1)
    print("Nested CV ROC AUC scores:", scores)
    print("Mean & std:", scores.mean(), scores.std())

    if export_csv:
        perf_df = pd.DataFrame({
            'Metric': ['Nested CV ROC AUC Mean', 'Nested CV ROC AUC Std'],
            'Value': [scores.mean(), scores.std()]
        }).set_index('Metric')
        perf_df.to_csv(csv_path)
        print(f"Model performance saved to {csv_path}")

if __name__ == '__main__':
    main()
