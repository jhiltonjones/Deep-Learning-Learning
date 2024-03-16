import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import torchbearer
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchbearer import Trial
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# fix random seed for reproducibility
seed = 7
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(seed)

# flatten 28*28 images to a 784 vector for each image
transform = transforms.Compose([
    transforms.ToTensor(),  # convert to tensor
    transforms.Lambda(lambda x: x.view(-1))  # flatten into vector
])

# load data
trainset = MNIST(".", train=True, download=True, transform=transform)
testset = MNIST(".", train=False, download=True, transform=transform)

# create data loaders
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=True)
# define baseline model
class BaselineModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BaselineModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out

hidden_layers = [50, 100, 500, 1000, 2000, 10000, 20000, 40000, 80000, 100000, 125000, 175000, 200000, 225000, 250000, 300000, 350000, 375000, 400000, 450000]
test_acc = []
train_acc = []
train_loss = []
test_loss = []
test_loss_list = []
val_loss_list = []  

for layer_size in hidden_layers:
    print(f"Training with {layer_size} hidden units")
    model = BaselineModel(784, layer_size, 10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    trial = Trial(model, optimizer, loss_fn, metrics=['accuracy', 'loss']).to(device)
    trial.with_generators(trainloader, val_generator=testloader)
    history = trial.run(epochs=100, verbose=0)

    # Extract validation loss from the history object
    val_loss = history[-1]['val_loss']

    # Manually compute accuracy and loss on test set
    model.eval()
    total = 0
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += loss_fn(outputs, target).item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_accuracy = correct / total
    test_loss /= len(testloader)

    # Append the results to your lists
    train_acc.append(history[-1]['acc'])
    train_loss.append(history[-1]['loss'])
    test_acc.append(test_accuracy)
    test_loss_list.append(test_loss)
    val_loss_list.append(val_loss)  # Append validation loss


# Plotting section
results_dir = 'lab4/results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Plot for loss
plt.figure()
plt.plot(hidden_layers, train_loss, label='Train Loss')
plt.plot(hidden_layers, val_loss_list, label='Validation Loss')  # Added validation loss
plt.plot(hidden_layers, test_loss_list, label='Test Loss')  # use test_loss_list here
plt.xlabel("Number of Hidden Units")
plt.ylabel("Loss")
plt.title("Comparison of Hidden Layers to Train, Validation, and Test Loss")
plt.legend()
plt.savefig(os.path.join(results_dir, 'figure1.png'))
plt.show()

# Plot for accuracy
plt.figure()
plt.plot(hidden_layers, train_acc, label='Train Accuracy')
plt.plot(hidden_layers, test_acc, label='Test Accuracy')
plt.xlabel("Number of Hidden Units")
plt.ylabel("Accuracy")
plt.title("Comparison of Hidden Layers to Train and Test Accuracy")
plt.legend()
plt.savefig(os.path.join(results_dir, 'figure2.png'))
plt.show()
