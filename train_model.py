# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    make_scorer, confusion_matrix,
    precision_recall_curve, auc
)


def main():
    # 1. Laad de verwerkte data
    df = pd.read_csv('processed_data.csv')

    # 2. Definieer features en target
    numeric_feats = [
        'grid_position','Q1_sec','Q2_sec','Q3_sec',
        'month','weekday','avg_finish_pos','avg_grid_pos','avg_const_finish',
        'air_temperature','track_temperature'
    ]
    categorical_feats = ['circuit_country','circuit_city']

    X = df[numeric_feats + categorical_feats]
    y = df['top3']

    # 3. Split in train/test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Preprocessing met imputatie + schalen en encoding
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)
    ])

    # 5. Pipeline met preprocessing en classifier
    pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # 6. Hyperparameter grid
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth':    [None, 5, 10],
        'clf__min_samples_split': [2, 5]
    }

    # 7. GridSearchCV (5-voudige stratified CV)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=make_scorer(roc_auc_score),
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    # 8. Beste parameters & CV-score
    print("Best parameters:", grid.best_params_)
    print(f"Best CV ROC AUC: {grid.best_score_:.3f}")

    # 9. Evaluatie op testset
    y_pred  = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]
    print("\nTestset performance:")
    print(classification_report(y_test, y_pred))
    print(f"Test ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

    # 10. Error-analyse
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    print(f"Precision-Recall AUC: {pr_auc:.3f}")

    # 11. Gedetailleerde foutanalyse: misclassificaties inspecteren
    df_test = df.loc[X_test.index].copy()
    df_test['pred']  = y_pred
    df_test['proba'] = y_proba
    misclassified = df_test[df_test['pred'] != df_test['top3']]
    print("\nVoorbeeld misclassificaties:")
    print(
        misclassified[[
            'Driver.driverId','raceName','finish_position','pred','proba'
        ]].head(10)
    )

if __name__ == '__main__':
    main()
