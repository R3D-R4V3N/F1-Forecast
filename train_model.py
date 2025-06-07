# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def main():
    # 1. Laad data
    df = pd.read_csv('processed_data.csv')

    # 2. Feature / target
    feature_cols = [
        'grid_position','Q1_sec','Q2_sec','Q3_sec',
        'month','weekday',
        'avg_finish_pos','avg_grid_pos'
    ]
    X = df[feature_cols]
    y = df['top3']

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Train baseline
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluatie
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

if __name__ == '__main__':
    main()
