import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import optuna
from torch.optim import Adam

# Load and preprocess dataset
data = pd.read_csv('/lyceum/jhj1g23/Deep-Learning-Learning/uav_data/WildFires/WildFires_DataSet.csv')
label_encoder = LabelEncoder()
data['CLASS'] = label_encoder.fit_transform(data['CLASS'])
X = data[['NDVI', 'LST', 'BURNED_AREA']].values
y = data['CLASS'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

class NeuralNet(nn.Module):
    def __init__(self, num_layers, num_neurons, dropout_rate):
        super(NeuralNet, self).__init__()
        layers = []
        input_size = 3  # Number of features

        for i in range(num_layers):
            layers.append(nn.Linear(input_size, num_neurons))
            layers.append(nn.BatchNorm1d(num_neurons))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            input_size = num_neurons

        layers.append(nn.Linear(num_neurons, 2))  # 2 output classes
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def objective(trial):
    num_layers = trial.suggest_int('num_layers', 1, 3)
    num_neurons = trial.suggest_int('num_neurons', 64, 256)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    model = NeuralNet(num_layers, num_neurons, dropout_rate).to('cpu')
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    
    for epoch in range(500):  
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
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
study.optimize(objective, n_trials=50)  


print("Best parameters: ", study.best_params)
print("Best accuracy: ", study.best_value)

best_params = study.best_params
best_model = NeuralNet(best_params['num_layers'], best_params['num_neurons'], best_params['dropout_rate']).to('cpu')
optimizer = Adam(best_model.parameters(), lr=best_params['lr'])
criterion = nn.CrossEntropyLoss()


full_train_loader = DataLoader(dataset=TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.long)), batch_size=64, shuffle=True)


for epoch in range(1000):  
    best_model.train()
    for batch_x, batch_y in full_train_loader:
        optimizer.zero_grad()
        output = best_model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# Save the model
torch.save(best_model.state_dict(), '/lyceum/jhj1g23/Deep-Learning-Learning/uav_data/WildFires/best_model.pth')

print("Model saved to '/lyceum/jhj1g23/Deep-Learning-Learning/uav_data/WildFires/best_model.pth'")
