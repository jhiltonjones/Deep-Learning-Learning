import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import optuna
from torch.optim import Adam, RMSprop, SGD

data = pd.read_csv('/lyceum/jhj1g23/Deep-Learning-Learning/uav_data/WildFires/WildFires_DataSet.csv')
label_encoder = LabelEncoder()
data['CLASS'] = label_encoder.fit_transform(data['CLASS'])
X = data[['NDVI', 'LST', 'BURNED_AREA']].values
y = data['CLASS'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

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

def objective(trial):
    num_layers = trial.suggest_int('num_layers', 1, 5)
    num_neurons = trial.suggest_int('num_neurons', 32, 512)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr_decay = trial.suggest_float('lr_decay', 0.95, 1.0)
    activation_fn = trial.suggest_categorical('activation', ['ReLU', 'ELU', 'LeakyReLU', 'Tanh', 'Sigmoid'])

    model = NeuralNet(num_layers, num_neurons, dropout_rate, activation_fn).to(device)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    for epoch in range(500):
        model.train()
        for batch_x, batch_y in DataLoader(train_dataset, batch_size=64, shuffle=True):
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in DataLoader(test_dataset, batch_size=64, shuffle=False):
                output = model(batch_x)
                _, predicted = torch.max(output.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        accuracy = correct / total

        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

best_params = study.best_params
best_model = NeuralNet(best_params['num_layers'], best_params['num_neurons'], best_params['dropout_rate'], best_params['activation']).to(device)
optimizer = getattr(torch.optim, best_params['optimizer'])(best_model.parameters(), lr=best_params['lr'])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=best_params['lr_decay'])
criterion = nn.CrossEntropyLoss()
full_train_loader = DataLoader(dataset=TensorDataset(torch.tensor(X_scaled, dtype=torch.float32).to(device), 
                                                     torch.tensor(y, dtype=torch.long).to(device)), 
                               batch_size=64, shuffle=True)
for epoch in range(1000):
    best_model.train()
    for batch_x, batch_y in full_train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = best_model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

torch.save(best_model.state_dict(), '/lyceum/jhj1g23/Deep-Learning-Learning/uav_data/WildFires/best_model2.pth')

print("Model saved to '/lyceum/jhj1g23/Deep-Learning-Learning/uav_data/WildFires/best_model2.pth'")

