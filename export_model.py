# export_model.py

import joblib
from train_model import build_and_train_pipeline

def save_pipeline():
    # Bouw en train de pipeline
    pipeline = build_and_train_pipeline()
    # Sla op
    joblib.dump(pipeline, 'f1_top3_pipeline.joblib')
    print("Pipeline opgeslagen als f1_top3_pipeline.joblib")

if __name__ == '__main__':
    save_pipeline()
