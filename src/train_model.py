import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ---------- LOAD DATA ----------
df = pd.read_csv("fake_or_real_news.csv")

# 🔥 IMPORTANT: check column names
print(df.columns)

# assume:
# text column = 'text'
# label column = 'label' (REAL/FAKE)

# ---------- FIX LABEL ----------
df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})

# ---------- CLEAN ----------
df['text'] = df['text'].apply(clean_text)

# ---------- DROP NULL ----------
df = df.dropna()

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# ---------- VECTORIZER ----------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------- MODEL ----------
model = LogisticRegression(max_iter=2000)
model.fit(X_train_vec, y_train)

# ---------- EVALUATE ----------
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------- SAVE ----------
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

print("✅ Model trained and saved")