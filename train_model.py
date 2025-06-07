# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, make_scorer

def build_and_train_pipeline():
    # 1. Laad data
    df = pd.read_csv('processed_data.csv')
    numeric_feats = [
        'grid_position','Q1_sec','Q2_sec','Q3_sec',
        'month','weekday','avg_finish_pos','avg_grid_pos','avg_const_finish',
        'air_temperature','track_temperature','grid_diff','Q3_diff','grid_temp_int'
    ]
    categorical_feats = ['circuit_country','circuit_city']
    X = df[numeric_feats + categorical_feats]
    y = df['top3']

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3. Preprocessing
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

    # 4. Pipeline + grid-search
    pipe = Pipeline([
        ('pre', pre),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth':    [None, 5],
        'clf__min_samples_split': [2, 5]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid,
                        scoring=make_scorer(roc_auc_score),
                        cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    # Print resultaten
    print("Best params:", grid.best_params_)
    print(f"CV ROC AUC: {grid.best_score_:.3f}")
    y_pred  = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:,1]
    print("\nTest performance:")
    print(classification_report(y_test, y_pred))
    print(f"Test ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

    # Return the full pipeline (incl. preprocessing & classifier)
    return grid.best_estimator_

if __name__ == '__main__':
    # bij direct run: train en print, maar ook pipeline terug
    pipeline = build_and_train_pipeline()
