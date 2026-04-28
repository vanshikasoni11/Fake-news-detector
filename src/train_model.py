import pandas as pd
import re
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ---------- PATH SETUP ----------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- LOAD DATA ----------
fake_df = pd.read_csv(DATA_DIR / "Fake.csv")
true_df = pd.read_csv(DATA_DIR / "True.csv")

# ---------- ADD LABELS ----------
fake_df["label"] = 0   # FAKE
true_df["label"] = 1   # REAL

# ---------- COMBINE ----------
df = pd.concat([fake_df, true_df], ignore_index=True)

# ---------- SHUFFLE ----------
df = df.sample(frac=1, random_state=42)

# ---------- CHECK COLUMNS ----------
print("Columns:", df.columns)

# ---------- USE TEXT ----------
# If dataset has 'title' and 'text', combine them
if "title" in df.columns:
    df["text"] = df["title"] + " " + df["text"]

# ---------- CLEAN ----------
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].apply(clean_text)

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# ---------- VECTORIZER ----------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------- MODEL ----------
model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# ---------- EVALUATE ----------
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------- SAVE ----------
joblib.dump(model, OUTPUT_DIR / "model.joblib")
joblib.dump(vectorizer, OUTPUT_DIR / "vectorizer.joblib")

print("\n✅ Model trained and saved in outputs/")

# ---------- TEST ----------
sample = ["Breaking: aliens landed on earth"]
pred = model.predict_proba(vectorizer.transform(sample))

print("\nTest Prediction (REAL prob, FAKE prob):", pred)