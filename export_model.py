# export_model.py

import joblib
from train_model import build_and_train_pipeline

def save_pipeline():
    pipe = build_and_train_pipeline()
    # Serialiseer naar schijf
    joblib.dump(pipe, 'f1_top3_pipeline.joblib')
    print("Pipeline opgeslagen als f1_top3_pipeline.joblib")

if __name__ == '__main__':
    save_pipeline()
