import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import torchbearer
from torch.utils.data import DataLoader, TensorDataset
import mnist
from torchbearer.metrics import TopKCategoricalAccuracy
from torchbearer import Trial
import numpy as np

# Set seed for reproducibility
seed = 1
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

# Data loading and transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
x_train, t_train, x_test, t_test = mnist.load()

# Convert to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float)
t_train = torch.tensor(t_train, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float)
t_test = torch.tensor(t_test, dtype=torch.long)

# Datasets and DataLoaders
train_dataset = TensorDataset(x_train, t_train)
test_dataset = TensorDataset(x_test, t_test)
trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Model definition
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

# EarlyStopping Callback
class EarlyStopping(Callback):
    def __init__(self, patience=5, delta=0, layer_size=None):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.layer_size = layer_size
        self.stopped_layers = []

    def on_end_epoch(self, state):
        val_loss = state['val_loss'] if 'val_loss' in state else None
        if val_loss is None:
            return
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                state[torchbearer.STOP_TRAINING] = True
                self.stopped_layers.append(self.layer_size)

hidden_layers = [50, 500, 1000, 10000, 50000, 100000, 150000, 200000, 250000, 300000, 350000]
epoch_losses = {size: [] for size in hidden_layers}
epoch_accuracies = {size: [] for size in hidden_layers}

device = "cuda" if torch.cuda.is_available() else "cpu"

for layer_size in hidden_layers:
    print(f'Layer size: {layer_size}')
    model = BaselineModel(784, layer_size, 10).to(device)
    optimiser = optim.SGD(model.parameters(), lr=0.01)
    metrics = ['accuracy', 'loss', TopKCategoricalAccuracy(k=5)]
    early_stopping = EarlyStopping(patience=5, delta=0.01, layer_size=layer_size)
    trial = Trial(model, optimiser, nn.CrossEntropyLoss(), metrics=metrics, callbacks=[early_stopping]).to(device)
    trial.with_generators(trainloader, test_generator=testloader)
    history = trial.run(epochs=100, verbose=0)
    
    epoch_losses[layer_size] = [h['loss'] for h in history]
    epoch_accuracies[layer_size] = [h['acc'] for h in history]

    if early_stopping.early_stop:
        stopped_layers.extend(early_stopping.stopped_layers)

# Plotting
for layer_size in hidden_layers:
    plt.figure()
    plt.plot(epoch_losses[layer_size], label=f'Loss for {layer_size} units')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Epoch vs Loss for {layer_size} units')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epoch_accuracies[layer_size], label=f'Accuracy for {layer_size} units')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Epoch vs Accuracy for {layer_size} units')
    plt.legend()
    plt.show()

print("Training stopped early for layers:", stopped_layers)
