import joblib
import re
from pathlib import Path

# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ---------- LOAD MODEL ----------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "outputs" / "model.joblib"
VEC_PATH = BASE_DIR / "outputs" / "vectorizer.joblib"

clf = joblib.load(MODEL_PATH)
vec = joblib.load(VEC_PATH)

# ---------- MAIN FUNCTION ----------
def hybrid_predict(text):

    # Clean input
    clean = clean_text(text)

    # Transform
    X = vec.transform([clean])

    # Predict probability
    real_prob = float(clf.predict_proba(X)[0, 1])  # 1 = REAL
    fake_prob = 1 - real_prob

    # Label logic (balanced)
    if real_prob >= 0.6:
        final_label = "REAL"
    elif real_prob <= 0.4:
        final_label = "FAKE"
    else:
        final_label = "UNCERTAIN"

    # ML label (simple)
    ml_label = "REAL" if real_prob > 0.5 else "FAKE"

    # Dummy fact-check (for UI)
    facts = []
    keywords = ["breaking", "shocking", "viral", "exclusive"]

    for word in keywords:
        if word in text.lower():
            facts.append(f"Suspicious keyword detected: {word}")

    # RETURN (VERY IMPORTANT STRUCTURE)
    return {
        "final_label": final_label,
        "ml_label": ml_label,
        "real_prob": round(real_prob, 3),
        "fake_prob": round(fake_prob, 3),
        "facts": facts
    }