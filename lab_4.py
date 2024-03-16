import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import torchbearer
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchbearer.metrics import TopKCategoricalAccuracy
from torchbearer import Trial
import numpy
from torchbearer import Callback
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
overfitted = []
train_acc5 =[]
test_acc5 = []

for layer_size in hidden_layers:
  model = BaselineModel(784, layer_size, 10).to(device)
  model.to(device)
  loss = nn.CrossEntropyLoss()
  optimiser = optim.SGD(model.parameters(), lr=0.01)
  metrics = ['accuracy', 'loss', TopKCategoricalAccuracy(k=5)]
  trial = Trial(model, optimiser, loss, metrics=metrics).to(device)
  trial.with_generators(trainloader, test_generator=testloader)

  history = trial.run(epochs=100, verbose=0)
   
  
  
  final_train_acc = history[-1]['acc']
  train_acc.append(final_train_acc)

  final_train_loss = history[-1]['loss']
  train_loss.append(final_train_loss)

  final_train_acc5 = history[-1]['top_5_acc']
  train_acc5.append(final_train_acc5)

  test_results = trial.evaluate(data_key=torchbearer.TEST_DATA)
  test_accuracy = test_results['test_acc']
  test_acc.append(test_accuracy)

  test_accuracy3 = test_results['test_loss']
  test_loss.append(test_accuracy3)

  test_accuracy2 = test_results['test_top_5_acc']
  test_acc5.append(test_accuracy2)


results_dir = 'lab4/results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Saving Figure 1
plt.figure()
plt.plot(hidden_layers, train_acc5, label='Train Top 5 Accuracy')
plt.plot(hidden_layers, test_acc5, label='Test Top 5 Accuracy')
plt.xlabel("Number of Hidden Units")
plt.ylabel("Top 5 Loss")
plt.title("Comparison between the Train Top 5 Loss and the Test Top 5 Loss")
plt.legend()
plt.savefig(os.path.join(results_dir, 'figure1.png'))
plt.show()

# Saving Figure 2
plt.figure()
plt.plot(hidden_layers, train_loss, label='Train Loss')
plt.plot(hidden_layers, test_loss, label='Test Loss')
plt.xlabel("Number of Hidden Units")
plt.ylabel("Loss")
plt.title("Comparison between the Quantity of Hidden Layers to Train Loss and Test Loss")
plt.legend()
plt.savefig(os.path.join(results_dir, 'figure2.png'))
plt.show()

# Saving Figure 3
plt.figure()
plt.plot(hidden_layers, train_acc, label='Train Accuracy')
plt.plot(hidden_layers, test_acc, label='Test Accuracy')
plt.xlabel("Number of Hidden Units")
plt.ylabel("Accuracy")
plt.title("Comparison between the Quantity of Hidden Layers to Train Accuracy and Test Accuracy")
plt.legend()
plt.savefig(os.path.join(results_dir, 'figure3.png'))
plt.show()
