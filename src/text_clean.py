import streamlit as st
import requests
from hybrid import hybrid_predict

st.set_page_config(page_title="TruthGuard AI", layout="centered")

st.title("TruthGuard AI 🔍")
st.write("BERT-powered Fake News Detection")

text = st.text_area("Enter News")

if st.button("Analyze"):

    if text.strip():

        result = hybrid_predict(text)

        st.subheader("Final Result")

        if result["final_label"] == "REAL":
            st.success("REAL NEWS")
        elif result["final_label"] == "FAKE":
            st.error("FAKE NEWS")
        else:
            st.warning("UNCERTAIN")

        st.write("ML Prediction:", result["ml_label"])
        st.write("Real Probability:", result["real_prob"])
        st.write("Fake Probability:", result["fake_prob"])

        st.subheader("Fact Check Results")

        if result["facts"]:
            for f in result["facts"]:
                st.write("•", f)
        else:
            st.write("No verified fact-check results found")

    else:
        st.warning("Enter text first")