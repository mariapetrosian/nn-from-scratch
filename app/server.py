from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np
from typing import Dict
from app.NeuralNetwork import NeuralNetwork

model: NeuralNetwork = joblib.load('app/model.joblib')

class_names: np.ndarray = np.array(['1', '0'])

class PredictionRequest(BaseModel):
    features: list[float]

app: FastAPI = FastAPI()

@app.get('/')
def read_root() -> Dict[str, str]:
    return {"version": "Heart-disease detector 0.1", "status": "OK"}

@app.post('/predict')
def predict(data: PredictionRequest) -> Dict[str, float]:
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    raw_output = model.forward(features)
    predicted_class = int(prediction[0][0])
    predicted_probability = raw_output[0][predicted_class]
    return {'predicted_class': predicted_class, 'predicted_probability': predicted_probability}
