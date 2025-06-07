# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    make_scorer,
    confusion_matrix,
    precision_recall_curve,
    auc,
    mean_absolute_error
)

def build_and_train_pipeline(export_csv=True, csv_path="model_performance.csv"):
    """Bouwt de pipeline, traint hem en retourneert het beste model.

    Parameters
    ----------
    export_csv : bool, optional
        Of de evaluatiemetrics naar ``csv_path`` weggeschreven moeten worden.
    csv_path : str, optional
        Pad waar het CSV-bestand met prestaties wordt opgeslagen.
    """

    # 1. Laad de verwerkte data
    df = pd.read_csv('processed_data.csv')

    # 2. Definieer features & target
    numeric_feats = [
        'grid_position','Q1_sec','Q2_sec','Q3_sec',
        'month','weekday','avg_finish_pos','avg_grid_pos','avg_const_finish',
        'air_temperature','track_temperature','grid_diff','Q3_diff','grid_temp_int'
    ]
    categorical_feats = ['circuit_country','circuit_city']

    X = df[numeric_feats + categorical_feats]
    y = df['top3']

    # 3. Train/test-split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Preprocessing pipelines
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_feats),
        ('cat', cat_pipe, categorical_feats)
    ])

    # 5. Full pipeline
    pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # 6. Hyperparameter grid
    param_grid = {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth':    [None, 5, 10],
        'clf__min_samples_split': [2, 5]
    }

    # 7. GridSearchCV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipe, param_grid,
        scoring=make_scorer(roc_auc_score),
        cv=cv, n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    # 8. Print performances
    print("Best parameters:", grid.best_params_)
    print(f"Best CV ROC AUC: {grid.best_score_:.3f}")

    y_pred  = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]
    print("\nTestset performance:")
    print(classification_report(y_test, y_pred))
    print(f"Test ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

    # **MAE erbij**
    mae = mean_absolute_error(y_test, y_proba)
    print(f"Mean Absolute Error (proba vs true): {mae:.3f}")

    # Confusion matrix, PR-AUC, misclassificaties …
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    print(f"Precision-Recall AUC: {pr_auc:.3f}")

    df_test = df.loc[X_test.index].copy()
    df_test['pred']  = y_pred
    df_test['proba'] = y_proba
    miscl = df_test[df_test['pred'] != df_test['top3']]
    print("\nVoorbeeld misclassificaties:")
    print(miscl[['Driver.driverId','raceName','finish_position','pred','proba']].head(5))

    if export_csv:
        perf_df = pd.DataFrame({
            'Metric': ['CV ROC AUC', 'Test ROC AUC', 'Mean Abs Error', 'PR AUC'],
            'Value': [grid.best_score_, roc_auc_score(y_test, y_proba), mae, pr_auc]
        }).set_index('Metric')
        perf_df.to_csv(csv_path)
        print(f"Model performance saved to {csv_path}")

    # Return de uiteindelijke pipeline
    return grid.best_estimator_

def main():
    build_and_train_pipeline()

if __name__ == '__main__':
    main()
