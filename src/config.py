from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "fake_or_real_news.csv"
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"