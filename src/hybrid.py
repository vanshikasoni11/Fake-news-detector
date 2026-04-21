from predict_bert import predict
from fact_check import fact_check

def hybrid_predict(text):

    label, real, fake = predict(text)
    facts = fact_check(text)

    final_label = label

    # ONLY override if strong evidence
    if facts:
        for f in facts:
            rating = str(f.get("rating", "")).lower()

            if "false" in rating and fake > 0.7:
                final_label = "FAKE"
            elif "true" in rating and real > 0.7:
                final_label = "REAL"

    # safety fallback
    if max(real, fake) < 0.6:
        final_label = "UNCERTAIN"

    return {
        "final_label": final_label,
        "ml_label": label,
        "real_prob": real,
        "fake_prob": fake,
        "facts": facts
    }