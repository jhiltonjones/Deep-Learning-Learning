import torch
import torch.nn as nn
import numpy as np

# Ensure this class definition matches the one used during training
class NeuralNet(nn.Module):
    def __init__(self, num_layers, num_neurons, dropout_rate, activation_fn):
        super(NeuralNet, self).__init__()
        layers = []
        input_size = 3  # Assuming input size is 3 as in your training script

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

        layers.append(nn.Linear(num_neurons, 2))  # Output layer
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Use the same parameters that were found to be best during training
num_layers = 5
num_neurons = 455
dropout_rate = 0.12074189901206918
activation_fn = 'LeakyReLU'

# Initialize the model and load the saved state dictionary
model = NeuralNet(num_layers, num_neurons, dropout_rate, activation_fn)
model.load_state_dict(torch.load('/lyceum/jhj1g23/Deep-Learning-Learning/uav_data/WildFires/best_model2.pth'))
model.eval()  # Set the model to evaluation mode

# Prepare an example input for prediction
# Ensure this preprocessing matches what was done during training
example_input = np.array([[0.454228,	14929.571429,	4.904762]])  # Example input
input_tensor = torch.tensor(example_input, dtype=torch.float32)

import torch.nn.functional as F

# Make a prediction
with torch.no_grad():  # No need to compute gradients
    logits = model(input_tensor)
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

# Mapping for class labels
class_mapping = {0: 'no_fire', 1: 'fire'}

# Print the predicted class and corresponding probability
predicted_label = class_mapping[predicted_class.item()]
predicted_probability = probabilities[0, predicted_class.item()].item()

print(f"Predicted Class: {predicted_label}, Probability: {predicted_probability:.4f}")


