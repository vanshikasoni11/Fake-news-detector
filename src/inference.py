import joblib
from config import MODEL_PATH, VECTORIZER_PATH

model = joblib.load(MODEL_PATH)
vec = joblib.load(VECTORIZER_PATH)


def predict(text: str):

    X = vec.transform([text])
    prob = model.predict_proba(X)[0][1]

    label = "REAL" if prob > 0.5 else "FAKE"

    return {
        "label": label,
        "confidence": float(prob)
    }