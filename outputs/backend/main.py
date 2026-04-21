from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_news

app = FastAPI(title="TruthGuard AI API")

class NewsRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(req: NewsRequest):

    label, real, fake = predict_news(req.text)

    return {
        "label": label,
        "real_probability": real,
        "fake_probability": fake
    }