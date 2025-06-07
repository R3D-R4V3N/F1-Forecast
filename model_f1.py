# model_f1.py
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix


def load_and_merge():
    """Load and merge raw CSVs into a master DataFrame for modeling."""
    df_q = pd.read_csv('jolpica_qualifying.csv')
    df_r = pd.read_csv(
        'jolpica_races.csv',
        usecols=['season', 'round', 'raceName', 'date']
    )
    df_w = pd.read_csv(
        'openf1_weather.csv',
        usecols=['air_temperature', 'track_temperature']
    )
    df_s = pd.read_csv(
        'openf1_sessions.csv',
        usecols=['session_key', 'circuit_short_name']
    )

    # Merge and add constant weather/session features
    df = df_q.merge(df_r, on=['season', 'round', 'raceName'], how='left')
    df['air_temperature'] = df_w['air_temperature'].iloc[0]
    df['track_temperature'] = df_w['track_temperature'].iloc[0]
    df['session_key'] = df_s['session_key'].iloc[0]
    df['circuit_short_name'] = df_s['circuit_short_name'].iloc[0]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df


def prepare_features(df):
    """Create target, handle missing lap times per race, and engineer features."""
    df = df.copy()
    df['top3'] = (df['position'] <= 3).astype(int)

    for col in ['Q1', 'Q2', 'Q3']:
        df[f'{col}_is_missing'] = df[col].isna().astype(int)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        race_med = df.groupby(['season', 'round'])[col].transform('median')
        global_med = df[col].median(skipna=True)
        df[col] = df[col].fillna(race_med).fillna(global_med)

    df['q2_diff'] = df['Q1'] - df['Q2']
    df['q3_diff'] = df['Q2'] - df['Q3']
    df['temp_diff'] = df['track_temperature'] - df['air_temperature']
    return df


def build_pipeline(model_type='random_forest'):
    """Build preprocessing + model pipeline: choose 'random_forest' or 'hist_gradient'."""
    numeric_features = [
        'Q1', 'Q2', 'Q3', 'q2_diff', 'q3_diff',
        'air_temperature', 'track_temperature', 'temp_diff',
        'Q1_is_missing', 'Q2_is_missing', 'Q3_is_missing'
    ]
    categorical_features = ['Driver.driverId', 'Constructor.constructorId']

    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        [('num', numeric_transformer, numeric_features),
         ('cat', categorical_transformer, categorical_features)]
    )

    if model_type == 'hist_gradient':
        clf = HistGradientBoostingClassifier(random_state=42, learning_rate=0.1)
    else:
        clf = RandomForestClassifier(random_state=42, class_weight='balanced')

    pipeline = Pipeline([('pre', preprocessor), ('clf', clf)])
    return pipeline


def train_and_evaluate(df):
    """Train/test split, hyperparameter tuning, and evaluation."""
    feature_cols = [
        'Q1', 'Q2', 'Q3', 'q2_diff', 'q3_diff',
        'air_temperature', 'track_temperature', 'temp_diff',
        'Q1_is_missing', 'Q2_is_missing', 'Q3_is_missing',
        'Driver.driverId', 'Constructor.constructorId'
    ]
    X = df[feature_cols]
    y = df['top3']

    # Final NaN guard
    num_feats = X.select_dtypes(include=['number']).columns
    cat_feats = X.select_dtypes(include=['object']).columns
    X.loc[:, num_feats] = X[num_feats].fillna(X[num_feats].median())
    X.loc[:, cat_feats] = X[cat_feats].fillna('missing')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Evaluate both models
    for model_key in ['random_forest', 'hist_gradient']:
        print(f"\nTraining model: {model_key}")
        pipeline = build_pipeline(model_type=model_key)
        if model_key == 'random_forest':
            param_grid = {'clf__n_estimators': [100, 200], 'clf__max_depth': [None, 10]}
        else:
            param_grid = {'clf__max_iter': [100, 200], 'clf__max_depth': [None, 10]}
        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', error_score='raise')
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        print(f"Best params ({model_key}):", grid.best_params_)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    df = load_and_merge()
    df = prepare_features(df)
    train_and_evaluate(df)
