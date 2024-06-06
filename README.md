# LeNet-5-CNN
This code implements a deep learning pipeline using PyTorch to train and evaluate the LeNet-5 convolutional neural network model on the MNIST dataset. Here’s a step-by-step breakdown of how it works:

1. Imports and Device Setup

import torch

import matplotlib.pyplot as plt

import torch.optim as optim

import torch.nn as nn

from torchvision import datasets, transforms

from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Import necessary libraries.

Check if a GPU is available and set the device accordingly.

2. Define Image Transformations
   
def trans():

    trans = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    return trans

trans = trans()

Define a function to transform the MNIST images by resizing them to 32x32 pixels and converting them to PyTorch tensors.

3. Load the MNIST Dataset

train_data = datasets.MNIST(root='./data', train=True, transform=trans, download=True)

test_data = datasets.MNIST(root='./data', train=False, transform=trans)

train_load = DataLoader(dataset=train_data, batch_size=32, shuffle=True)

test_load = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

Download and load the MNIST training and test datasets.

Create DataLoader objects for batching and shuffling the datasets.

4. Define the LeNet-5 Model

class LNet5(nn.Module):

    def __init__(self, num_classes=10):
        super(LNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = LNet5().to(device)

Define the LeNet-5 model architecture with three convolutional layers followed by fully connected layers.

Instantiate the model and move it to the selected device (CPU or GPU).

5. Define the Training and Evaluation Functions

def train_model(model, train_load, criteria, optimize, epochs):
    train_loss = []
    test_accuracy = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for img, labels in train_load:
            img, labels = img.to(device), labels.to(device)

            optimize.zero_grad()
            output = model(img)
            loss = criteria(output, labels)
            loss.backward()
            optimize.step()

            epoch_loss += loss.item()

        avg_e_loss = epoch_loss / len(train_load)
        train_loss.append(avg_e_loss)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for img, labels in test_load:
                img, labels = img.to(device), labels.to(device)
                output = model(img)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            test_accuracy.append(accuracy)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_e_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%')

    return train_loss, test_accuracy
    
Define a function to train the model, compute the training loss, and evaluate the test accuracy at each epoch.

Track the training loss and test accuracy for each epoch.

6. Training Loop

criteria = nn.CrossEntropyLoss()

optimize = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

train_losses, test_accuracies = train_model(model, train_load, criteria, optimize, epochs)

Set the loss function and optimizer.

Define the number of training epochs.

Train the model using the train_model function and store the training losses and test accuracies.

7. Save the Model

torch.save(model.state_dict(), 'lenet5_model.pth')

Save the trained model's parameters to a file.

8. Plotting Results
   
plt.plot(train_losses, label='Training Loss')

plt.title('Training Loss over Epochs')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.show()

plt.plot(test_accuracies, label='Test Accuracy')

plt.title('Test Accuracy over Epochs')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.show()

Plot the training loss over epochs.

Plot the test accuracy over epochs.
