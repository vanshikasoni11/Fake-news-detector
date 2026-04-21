import streamlit as st
import argparse
from pathlib import Path
import re
import joblib
import time
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


st.set_page_config(page_title="TruthGuard AI v2", layout="centered")


st.markdown("""
<style>
body { background-color: #f8f5f2; }

.block-container {
    max-width: 720px;
    padding-top: 2rem;
}

.big-title {
    font-size: 38px;
    font-weight: 800;
    text-align: center;
    color: #3b2f2f;
}

.sub-text {
    text-align: center;
    color: #7a5c58;
    margin-bottom: 25px;
}

.stButton>button {
    width: 100%;
    background-color: #5a3e36;
    color: white;
    border-radius: 8px;
    padding: 10px;
}

hr { display: none; }
</style>
""", unsafe_allow_html=True)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def default_paths():
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs"
    return {
        "pipeline": out / "pipeline.joblib",
        "model": out / "model.joblib",
        "vectorizer": out / "vectorizer.joblib",
        "bert_model": out / "bert_model",
    }


def load_sklearn(model_path, vectorizer_path):
    if model_path.exists() and vectorizer_path.exists():
        return joblib.load(model_path), joblib.load(vectorizer_path)
    return None, None


@st.cache_resource
def load_bert(model_path):
    if not model_path.exists():
        return None, None, None

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_path))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    return tokenizer, model, "bert"


def predict_bert(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
    return float(probs[0])  # REAL probability

def predict_sklearn(text, clf, vec):
    X = vec.transform([text])
    return float(clf.predict_proba(X)[0, 1])




def main():

    dp = default_paths()

    clf, vec = load_sklearn(dp["model"], dp["vectorizer"])
    tokenizer, bert_model, mode = load_bert(dp["bert_model"])

    st.markdown('<p class="big-title">TruthGuard AI v2</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Hybrid Fake News Detection (ML + BERT)</p>', unsafe_allow_html=True)

    txt = st.text_area("Paste news text", height=160)
    threshold = st.slider("Sensitivity", 0.2, 0.8, 0.5)

    analyze = st.button("Analyze")

    if analyze and txt.strip():

        with st.spinner("Analyzing with AI..."):
            time.sleep(1)

        clean = clean_text(txt)

        
        if tokenizer and bert_model:
            real_prob = predict_bert(clean, tokenizer, bert_model)
            used_model = "DistilBERT"
        elif clf and vec:
            real_prob = predict_sklearn(clean, clf, vec)
            used_model = "Sklearn (TF-IDF)"
        else:
            st.error("No model found. Train first.")
            return

        fake_prob = 1 - real_prob
        st.write("DEBUG real_prob:", real_prob)
        st.write("DEBUG fake_prob:", fake_prob)

        
        if real_prob >= threshold:
            label = "REAL"
        elif fake_prob >= threshold:
            label = "FAKE"
        else:
            label = "UNCERTAIN"

    
        st.markdown("### Result")

        if label == "REAL":
            st.success(f"REAL NEWS ({real_prob*100:.1f}%)")
        elif label == "FAKE":
            st.error(f"FAKE NEWS ({fake_prob*100:.1f}%)")
        else:
            st.warning("UNCERTAIN RESULT")

        st.progress(int(real_prob * 100))

        st.caption(f"Model used: {used_model}")

        
        st.markdown("### Explanation")

        reasons = []

        if real_prob > 0.75:
            reasons.append("Strong confidence in real-world language patterns")
        elif fake_prob > 0.75:
            reasons.append("High similarity to misinformation patterns")
        else:
            reasons.append("Mixed linguistic signals detected")

        words = ["breaking", "shocking", "urgent", "viral", "exclusive"]
        found = [w for w in words if w in txt.lower()]

        if found:
            reasons.append(f"Clickbait words detected: {', '.join(found)}")

        if len(txt.split()) < 6:
            reasons.append("Very short text (low context)")

        if "!" in txt:
            reasons.append("Emotional punctuation detected")

        for r in reasons:
            st.write("•", r)

if __name__ == "__main__":
    main()