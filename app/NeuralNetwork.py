import numpy as np
from typing import Any, List, Dict
import joblib

# Activation Functions
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    sigmoid_output: np.ndarray = sigmoid(x)  # Ensures x is passed through sigmoid first
    return sigmoid_output * (1 - sigmoid_output)

def ReLU(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)  # Sets negative values to 0, passes positive values

def ReLU_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)  # Returns 1 for positive inputs, 0 for negative ones

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int) -> None:
        self.layers: List[Dict] = []
        previous_size: int = input_size
        for hidden_size in hidden_sizes:
            self.layers.append({
                "weights": 0.01 * np.random.randn(previous_size, hidden_size),
                "biases": np.zeros((1, hidden_size))
            })
            previous_size = hidden_size

        #output layer
        self.output_layer: Dict = {
            "weights": 0.01 * np.random.randn(previous_size, output_size),
            "biases": np.zeros((1, output_size))
        }

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.layer_outputs: List[np.ndarray] = []
        self.layer_inputs: List[np.ndarray] = []

        current_input: np.ndarray = inputs
        for layer in self.layers:
            layer_input: np.ndarray = np.dot(current_input, layer["weights"]) + layer["biases"]
            self.layer_inputs.append(layer_input)
            layer_output: np.ndarray = ReLU(layer_input)
            self.layer_outputs.append(layer_output)
            current_input = layer_output

        #output layer
        output_layer_input: np.ndarray = np.dot(current_input, self.output_layer["weights"]) + self.output_layer["biases"]
        self.layer_inputs.append(output_layer_input)
        output: np.ndarray = sigmoid(output_layer_input)
        return output

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        outputs: np.ndarray = self.forward(inputs)
        predictions: np.ndarray = (outputs > 0.5).astype(np.int32)
        return predictions

    def backward(self, inputs: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        #forward pass to get predictions
        predictions = self.forward(inputs)
        loss = y - predictions

        #gradient for the output layer
        output_delta: np.ndarray = loss * sigmoid_derivative(self.layer_inputs[-1])
        self.output_layer["weights"] += self.layer_outputs[-1].T.dot(output_delta) * learning_rate
        self.output_layer["biases"] += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        #backpropagate through hidden layers
        output_delta = output_delta.dot(self.output_layer["weights"].T)  #update delta for next layer
        for i in range(len(self.layers) - 1, -1, -1):
            if i == 0:
                previous_output = inputs
            else:
                previous_output = self.layer_outputs[i - 1]  #previous hidden layer output

            #gradient for the current hidden layer
            hidden_delta: np.ndarray = output_delta * ReLU_derivative(self.layer_inputs[i])
            self.layers[i]["weights"] += previous_output.T.dot(hidden_delta) * learning_rate
            self.layers[i]["biases"] += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

            #update delta for the next layer
            output_delta = hidden_delta.dot(self.layers[i]["weights"].T)

    def bce_loss(self, y: np.ndarray, predictions: np.ndarray) -> float:
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7) #to prevent log(0) errors
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return loss

    def compute_accuracy(self, y: np.ndarray, predictions: np.ndarray) -> float:
        predicted_classes = (predictions > 0.5).astype(np.int32)
        return np.mean(predicted_classes == y)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float) -> None:
        for epoch in range(epochs):
            predictions: np.ndarray = self.forward(X)
            self.backward(X, y, learning_rate)

            loss: float = self.bce_loss(y, predictions)
            accuracy: float = self.compute_accuracy(y, predictions)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")


#save the model
model = NeuralNetwork(input_size=12, hidden_sizes=[64, 32], output_size=1)
joblib.dump(model, 'app/model.joblib')
