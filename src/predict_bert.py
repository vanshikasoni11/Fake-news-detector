import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizerFast.from_pretrained("bert_model")
model = DistilBertForSequenceClassification.from_pretrained("bert_model")
model.eval()

def predict(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    fake_prob = float(probs[0])   # index 0 = FAKE
    real_prob = float(probs[1])   # index 1 = REAL

    # DEBUG (important)
    print("REAL:", real_prob, "FAKE:", fake_prob)

    if real_prob >= 0.6:
        label = "REAL"
    elif real_prob <= 0.4:
        label = "FAKE"
    else:
        label = "UNCERTAIN"

    confidence = max(real_prob, fake_prob)

    return label, real_prob, fake_prob, confidence
