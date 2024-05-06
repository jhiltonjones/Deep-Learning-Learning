import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import torchbearer
from torch.utils.data import DataLoader, TensorDataset
import mnist
from torchbearer import Trial

# Set seed for reproducibility
seed = 1
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

# Data loading and transformation
x_train, t_train, x_test, t_test = mnist.load()

# Convert to PyTorch tensors and apply transformations
x_train = torch.tensor(x_train.reshape(-1, 784) / 255., dtype=torch.float)  # Reshape and normalize
t_train = torch.tensor(t_train, dtype=torch.long)
x_test = torch.tensor(x_test.reshape(-1, 784) / 255., dtype=torch.float)  # Reshape and normalize
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

hidden_layers = [50, 500, 1000, 10000, 50000, 100000, 150000, 200000, 250000, 300000, 350000]
epoch_losses = {size: [] for size in hidden_layers}
epoch_accuracies = {size: [] for size in hidden_layers}

device = "cuda" if torch.cuda.is_available() else "cpu"

for layer_size in hidden_layers:
    model = BaselineModel(784, layer_size, 10).to(device)
    optimiser = optim.SGD(model.parameters(), lr=0.01)
    trial = Trial(model, optimiser, nn.CrossEntropyLoss(), metrics=['acc'], verbose=1).to(device)
    trial.with_generators(trainloader, test_generator=testloader)
    history = trial.run(epochs=100)

    epoch_losses[layer_size] = [(h['training_loss'], h['val_loss']) for h in history]
    epoch_accuracies[layer_size] = [(h['training_acc'], h['val_acc']) for h in history]

# Directory to save the figures
save_dir = "/path/to/save/results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Plotting and saving figures
for metric_name, data_dict in [('loss', epoch_losses), ('accuracy', epoch_accuracies)]:
    plt.figure(figsize=(12, 8))
    for layer_size, values in data_dict.items():
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, [v[0] for v in values], label=f'{layer_size} units (Train {metric_name})')
        plt.plot(epochs, [v[1] for v in values], '--', label=f'{layer_size} units (Val {metric_name})')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'Training and Validation {metric_name.capitalize()} vs. Epochs for Various Layer Sizes')
    plt.legend()
    fig_path = os.path.join(save_dir, f'{metric_name}_vs_epochs.png')
    plt.savefig(fig_path)
    plt.close()

# Note: Adjust "/path/to/save/results" to your actual directory path where you want to save results.
