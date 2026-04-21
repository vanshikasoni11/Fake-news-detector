import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
from sklearn.metrics import accuracy_score

# ---------------- LOAD DATA ----------------
df = pd.read_csv("fake_or_real_news.csv")
df = df[['text', 'label']].dropna()

df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# ---------------- SPLIT ----------------
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42
)

# ---------------- TOKENIZER ----------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
test_enc = tokenizer(test_texts, truncation=True, padding=True, max_length=256)

# ---------------- DATASET ----------------
class NewsDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_enc, train_labels)
test_dataset = NewsDataset(test_enc, test_labels)

# ---------------- MODEL ----------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# ---------------- METRICS ----------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# ---------------- TRAINING ----------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# ---------------- SAVE MODEL ----------------
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")

print("Model training complete and saved!")