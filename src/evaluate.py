# src/evaluate.py

import joblib
from sklearn.metrics import classification_report
from data_loader import load_data
from preprocess import preprocess
from config import MODEL_PATH, VECTORIZER_PATH


def evaluate():

    df = preprocess(load_data())

    model = joblib.load(MODEL_PATH)
    vec = joblib.load(VECTORIZER_PATH)

    X = vec.transform(df['text'])
    y = df['label']

    preds = model.predict(X)

    print(classification_report(y, preds))