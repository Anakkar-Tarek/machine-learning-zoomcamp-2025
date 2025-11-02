from fastapi import FastAPI
import pickle

app = FastAPI()

with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Lead scoring API is running!"}

@app.post("/predict")
def predict(client: dict):
    X = [client]
    y_pred = model.predict_proba(X)[0, 1]
    return {"probability": float(y_pred)}
