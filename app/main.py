import pandas as pd
import numpy as np
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .NeuralNetwork import NeuralNetwork
from pathlib import Path

file_path: Path = Path("C:/Users/USER/OneDrive/Desktop/project/data/Dataset Heart Disease.csv")
df: pd.DataFrame = pd.read_csv(file_path)

#Feature and target variables
X: pd.DataFrame = df.drop('target', axis=1)
y: pd.Series = df['target']

#Splitting the data into training and testing sets
X_train: pd.DataFrame
X_test: pd.DataFrame
y_train: np.ndarray
y_test: np.ndarray

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.values.reshape(-1, 1)

#Normalizing the data (standardizing)
scaler: StandardScaler = StandardScaler()
X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
X_test_scaled: np.ndarray = scaler.transform(X_test)

model = NeuralNetwork(input_size=12, hidden_sizes=[64, 32], output_size=1)

print("Training the model...")
model.train(X_train_scaled, y_train, epochs=50, learning_rate=0.01)

print("\nTesting the model...")
y_pred: np.ndarray = model.predict(X_test_scaled)
print(y_pred.T)



