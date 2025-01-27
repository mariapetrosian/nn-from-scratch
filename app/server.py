from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np
from typing import Dict
from app.NeuralNetwork import NeuralNetwork

model: joblib.Memory = joblib.load('app/model.joblib')
class_names: np.ndarray = np.array(['1', '0'])

class PredictionRequest(BaseModel):
    features: list[float]

app: FastAPI = FastAPI()

@app.get('/')
def read_root() -> Dict[str, str]:
    return {'message': 'Heart disease risk model API'}

@app.post('/predict')
def predict(data: PredictionRequest):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features) 
    class_name = int(prediction[0][0])
    return {'predicted_class': class_name}

