import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizerFast.from_pretrained("model/bert_model")
model = DistilBertForSequenceClassification.from_pretrained("model/bert_model")
model.eval()

def predict_news(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    fake = float(probs[0])
    real = float(probs[1])

    if real > 0.7:
        label = "REAL"
    elif fake > 0.7:
        label = "FAKE"
    else:
        label = "UNCERTAIN"

    return label, real, fake