import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms

# Define the model class (this should be the same as when the model was trained)
class NeuralNet(nn.Module):
    def __init__(self, num_layers, num_neurons, dropout_rate, activation_fn):
        super(NeuralNet, self).__init__()
        layers = []
        input_size = 3

        for i in range(num_layers):
            layers.append(nn.Linear(input_size, num_neurons))
            layers.append(nn.BatchNorm1d(num_neurons))
            if activation_fn == 'ReLU':
                layers.append(nn.ReLU())
            elif activation_fn == 'ELU':
                layers.append(nn.ELU())
            elif activation_fn == 'LeakyReLU':
                layers.append(nn.LeakyReLU())
            elif activation_fn == 'Tanh':
                layers.append(nn.Tanh())
            elif activation_fn == 'Sigmoid':
                layers.append(nn.Sigmoid())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            input_size = num_neurons

        layers.append(nn.Linear(num_neurons, 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
num_layers = 5  # Optuna's best number of layers
num_neurons = 455  # Optuna's best number of neurons
dropout_rate = 0.12074189901206918  # Optuna's best dropout rate
activation_fn = 'LeakyReLU'  # Optuna's best activation function

model = NeuralNet(num_layers, num_neurons, dropout_rate, activation_fn)
model.load_state_dict(torch.load('/Users/jackhilton-jones/Deep-Learning-Learning/Deep-Learning-Learning/best_model2.pth'))
model.eval()


# Prepare your input data
input_data = np.array([[0.506781, 14584.272722, 4.692308]])
input_tensor = torch.from_numpy(input_data).float()
input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

# Make a prediction
with torch.no_grad():
    logits = model(input_tensor)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1)
    confidence = torch.max(probabilities, dim=1)

# Print prediction and confidence
print(f"Predicted class: {prediction.item()}, Confidence: {confidence.values.item():.2f}")