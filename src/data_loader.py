import pandas as pd
from config import DATA_PATH

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[['text', 'label']].dropna()
    df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})
    return df