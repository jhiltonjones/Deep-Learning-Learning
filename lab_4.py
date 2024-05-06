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
import numpy
from torchbearer import Callback
seed = 1
# fix random seed for reproducibility
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
x_train, t_train, x_test, t_test = mnist.load()

# Convert the numpy arrays to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float)
t_train = torch.tensor(t_train, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float)
t_test = torch.tensor(t_test, dtype=torch.long)

# Create TensorDataset objects
train_dataset = TensorDataset(x_train, t_train)
test_dataset = TensorDataset(x_test, t_test)

# Create DataLoader objects
trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

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
        # Use the validation loss as the metric to monitor
        val_loss = state['val_loss'] if 'val_loss' in state else None

        if val_loss is None:
            return  # Skip if validation loss is not available

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
test_acc = []
train_acc = []
train_loss = []
test_loss = []
overfitted = []
train_acc5 =[]
test_acc5 = []
stopped_layers = []

device = "cuda" if torch.cuda.is_available() else "cpu"
for layer_size in hidden_layers:
  print(f'Layer size: {layer_size}')
  model = BaselineModel(784, layer_size, 10).to(device)
  model.to(device)
  loss = nn.CrossEntropyLoss()
  optimiser = optim.SGD(model.parameters(), lr=0.01)
  metrics = ['accuracy', 'loss', TopKCategoricalAccuracy(k=5)]
  early_stopping = EarlyStopping(patience=5, delta=0.01, layer_size=layer_size)
  trial = Trial(model, optimiser, loss, metrics=metrics, callbacks=[early_stopping]).to(device)
  trial.with_generators(trainloader, test_generator=testloader)

  history = trial.run(epochs=100, verbose=0)
  if early_stopping.early_stop:
        stopped_layers.extend(early_stopping.stopped_layers)
   
  
  
  final_train_acc = history[-1]['acc']
  train_acc.append(final_train_acc)

  final_train_loss = history[-1]['loss']
  train_loss.append(final_train_loss)

  """final_train_acc5 = history[-1]['top_5_acc']
  train_acc5.append(final_train_acc5)"""

  test_results = trial.evaluate(data_key=torchbearer.TEST_DATA)
  test_accuracy = test_results['test_acc']
  test_acc.append(test_accuracy)

  test_accuracy3 = test_results['test_loss']
  test_loss.append(test_accuracy3)

"""test_accuracy2 = test_results['test_top_5_acc']
  test_acc5.append(test_accuracy2)"""



"""plt.figure()
plt.plot(hidden_layers, train_acc5, label = 'Train Top 5 Accuracy')
plt.plot(hidden_layers, test_acc5, label = 'Test Top 5 Accuracy')
plt.xlabel("NUmber of Hidden Units")
plt.ylabel("Top 5 Loss")
plt.title("Comparison between the Train Top 5 Loss and the Test Top 5 Loss")
plt.legend()
plt.savefig('/Users/jackhilton-jones/Deep_Learning/21')
plt.show()"""

plt.figure()
plt.plot(hidden_layers, train_loss, label = 'Train Loss')
plt.plot(hidden_layers, test_loss, label ='Test Loss')
plt.xlabel("Number of Hidden Units")
plt.ylabel("Loss")
plt.title("Comparison between the Quantity of Hidden Layers to Train Loss and Test Loss")
plt.legend()
plt.savefig('/lyceum/jhj1g23/Deep-Learning-Learning/lab4/results/train_loss_and_test_loss')
plt.show()

plt.figure()
plt.plot(hidden_layers, train_acc, label = 'Train Accuracy')
plt.plot(hidden_layers, test_acc, label ='Test Accuracy')
plt.xlabel("Number of Hidden Units")
plt.ylabel("Accuracy")
plt.title("Comparison between the Quantity of Hidden Layers to Train Accuracy and Test Accuracy")
plt.legend()
plt.savefig('/lyceum/jhj1g23/Deep-Learning-Learning/lab4/results/train_accuracy_and_test_acccuracy')
plt.show()
print("Training stopped early for layers:", stopped_layers)
