import streamlit as st
from pathlib import Path
import re
import joblib
import time
import requests
import os

from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")



from fact_check import API_KEY, fact_check


def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={API_KEY}"

    response = requests.get(url)
    data = response.json()

    articles = []

    if "articles" in data:
        for a in data["articles"][:5]:
            title = a.get("title", "")
            desc = a.get("description", "")
            articles.append(title + " " + desc)

    return articles

def fetch_related_news(query):

    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"

    r = requests.get(url)
    data = r.json()

    if "articles" in data and data["articles"]:
        a = data["articles"][0]
        return a.get("title","") + " " + a.get("description","")

    return ""


st.set_page_config(
    page_title="TruthGuard AI - Fake News Detector",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>

/* Hide Streamlit menu */
#MainMenu {visibility: hidden;}

/* Hide footer */
footer {visibility: hidden;}

/* Hide header */
header {visibility: hidden;}

/* Remove top padding */
.block-container {
    padding-top: 1.5rem;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

.main {
    background: #0e1117;
}

.block-container{
    padding-top: 5rem;
    max-width: 1100px;
}

.big-title{
    font-size: 72px;
    font-weight: 800;
    text-align:center;
    color:white;
    margin-bottom:0px;
}

.sub-text{
    text-align:center;
    color:#9ca3af;
    font-size:18px;
    margin-bottom:30px;
}

.stTextArea textarea{
    border-radius:16px !important;
    padding:18px !important;
    font-size:18px !important;
    border:1px solid #374151 !important;
}

.stButton>button{
    width:100%;
    border-radius:14px;
    height:52px;
    font-size:18px;
    font-weight:700;
    background: linear-gradient(90deg,#2563eb,#06b6d4);
    color:white;
    border:none;
}

.stButton>button:hover{
    transform:scale(1.01);
}

h3{
    color:white !important;
    margin-top:25px;
}

</style>
""", unsafe_allow_html=True)



# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def is_short_text(text):
    return len(text.split()) < 12


@st.cache_resource
def load_model():
    BASE_DIR = Path(__file__).resolve().parents[1]
    model_path = BASE_DIR / "outputs" / "model.joblib"
    vec_path = BASE_DIR / "outputs" / "vectorizer.joblib"

    if not model_path.exists() or not vec_path.exists():
        return None, None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)

    return model, vectorizer

def predict(text, model, vectorizer):

    clean = clean_text(text)
    X = vectorizer.transform([clean])

    real_prob = float(model.predict_proba(X)[0, 1])
    fake_prob = 1 - real_prob

    words = len(text.split())
    short = words < 10
    text_lower = text.lower()

    institutional_terms = [
        "government", "minister", "court", "supreme court",
        "rbi", "parliament", "police", "isro", "nasa"
    ]

    clickbait_terms = [
        "shocking", "viral", "miracle", "secret",
        "guaranteed", "instantly", "100%", "aliens",
        "free money", "exclusive"
    ]

    if short:
        
        extra_text = fetch_related_news(text)
        if extra_text:
            text = text + " " + extra_text

        # Small credibility boost only
        if any(t in text_lower for t in institutional_terms):
            real_prob += 0.08

        # Stronger penalty for sensational claims
        if any(t in text_lower for t in clickbait_terms):
            real_prob -= 0.25

        # suspicious numbers/promises
        if "₹" in text or "free" in text_lower:
            real_prob -= 0.15

        real_prob = max(0, min(real_prob, 1))
        fake_prob = 1 - real_prob

        if real_prob >= 0.58:
            label = "REAL"
        elif real_prob <= 0.40:
            label = "FAKE"
        else:
            label = "UNCERTAIN"

    else:
        if real_prob >= 0.55:
            label = "REAL"
        elif real_prob <= 0.45:
            label = "FAKE"
        else:
            label = "UNCERTAIN"

    return label, real_prob, fake_prob, short

# ---------- MAIN ----------
def main():

    model, vectorizer = load_model()

    st.markdown('<p class="big-title">TruthGuard AI v2</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Fake News Detection using Machine Learning</p>', unsafe_allow_html=True)

    text = st.text_area("Paste News Text", height=150)

    if st.button("Analyze"):

        if not text.strip():
            st.warning("Please enter some text")
            return

        if model is None:
            st.error("Model not found. Please train model first.")
            return

        with st.spinner("Analyzing..."):
            label, real_prob, fake_prob, short = predict(text, model, vectorizer)
        # ---------- RESULT ----------
        st.markdown("### Result")
        if short:
            st.info("Short headline detected — using adaptive prediction")

        if label == "REAL":
            st.success(f"REAL NEWS ({real_prob*100:.1f}%)")
        elif label == "FAKE":
            st.error(f"FAKE NEWS ({fake_prob*100:.1f}%)")
        else:
            st.warning("UNCERTAIN RESULT")

        st.progress(int(real_prob * 100))

        st.markdown("### Fact Check Results")

        results = fact_check(text)

        if results:
            for r in results:
                st.write("Claim:", r["text"])
                st.write("Rating:", r["rating"])
                st.write("Publisher:", r["publisher"])
                st.write("---")
        else:
            st.write("No verified fact-check results found.")
        

        # ---------- DETAILS ----------
        st.markdown("### Details")
        st.write("Real Probability:", round(real_prob, 3))
        st.write("Fake Probability:", round(fake_prob, 3))

        # ---------- EXPLANATION ----------
        st.markdown("### Explanation")

        reasons = []

        if real_prob > 0.75:
            reasons.append("Strong real-world language pattern detected")
        elif fake_prob > 0.75:
            reasons.append("Strong misinformation pattern detected")
        else:
            reasons.append("Mixed signals in text")

        keywords = ["breaking", "shocking", "viral", "exclusive"]
        found = [w for w in keywords if w in text.lower()]

        if found:
            reasons.append(f"Clickbait words: {', '.join(found)}")

        if len(text.split()) < 6:
            reasons.append("Very short text (low context)")

        for r in reasons:
            st.write("•", r)

    st.markdown("### 🧠 Real-Time News Detection")

    if st.button("Fetch Latest News"):

        news_list = fetch_news()

        for i, news in enumerate(news_list):

            st.write(f"**{i+1}. {news}**")

            if st.button(f"Analyze News {i+1}", key=f"news_{i}"):

                label, real_prob, fake_prob, short = predict(news, model, vectorizer)

                if label == "REAL":
                    st.success(f"REAL ({real_prob*100:.1f}%)")
                elif label == "FAKE":
                    st.error(f"FAKE ({fake_prob*100:.1f}%)")
                else:
                    st.warning("UNCERTAIN")
                
                
if __name__ == "__main__":
    main()