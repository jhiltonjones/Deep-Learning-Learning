from torchvision.models import resnet50, ResNet50_Weights
from urllib.request import urlopen
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchbearer
from torchbearer import Trial
from sklearn import metrics

batch_size=128
# the size of the images that we'll learn on - we'll shrink them from the original size for speed
image_size=(30, 100)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()  # convert to tensor
])
imagenet_labels = urlopen("https://raw.githubusercontent.com/Lasagne/Recipes/master/examples/resnet50/imagenet_classes.txt").read().decode('utf-8').split("\n")
train_dataset = ImageFolder("/Users/jackhilton-jones/Deep_Learning/Lab6/data/train", transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = ImageFolder("/Users/jackhilton-jones/Deep_Learning/Lab6/data/valid", transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = ImageFolder("/Users/jackhilton-jones/Deep_Learning/Lab6/data/test", transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Load the model state
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.avgpool = nn.AdaptiveAvgPool2d((1,1))
model.fc = nn.Linear(2048, len(train_dataset.classes))
model.train()
"print(model)"
model_file = '/Users/jackhilton-jones/Deep_Learning/Lab6/resnet_model.pth'
torch.save(model.state_dict(), model_file)

# Freeze layers by not tracking gradients
for param in model.parameters():
    param.requires_grad = False
model.fc.weight.requires_grad = True #unfreeze last layer weights
model.fc.bias.requires_grad = True #unfreeze last layer biases

# Create the optimizer with different learning rates for the two parameter groups
optimiser = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4) #only optimse non-frozen layers
device = "mps" if torch.backends.mps.is_available() else "cpu"
loss_function = nn.CrossEntropyLoss()
trial = Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_loader, val_generator=val_loader, test_generator=test_loader)
trial.run(epochs=100)
results = trial.evaluate(data_key=torchbearer.VALIDATION_DATA)
print()
print(results)
predictions = trial.predict()
predicted_classes = predictions.argmax(1).cpu()
true_classes = list(x for (_,x) in test_dataset.samples)


report = metrics.classification_report(true_classes, predicted_classes, target_names=train_dataset.classes, zero_division=0)
# Save the report to a file
file_path = "/Users/jackhilton-jones/Deep_Learning/Lab6/classification_report_default.txt"
with open(file_path, 'w') as file:
    file.write(report)

print("Classification report saved to:", file_path)

model_file = '/Users/jackhilton-jones/Deep_Learning/Lab6/resnet_model_default.pth'
torch.save(model.state_dict(), model_file)
