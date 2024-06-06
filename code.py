import torch  # Import PyTorch library
import matplotlib.pyplot as plt  # Import Matplotlib library for plotting
import torch.optim as optim  # Optimization algorithms from PyTorch
import torch.nn as nn  # Neural network module from PyTorch
from torchvision import datasets, transforms  # Vision-related functionalities
from torch.utils.data import DataLoader  # DataLoader for handling datasets

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU

# Define transforms
def trans():
    """Define image transformations."""
    trans = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize images to 32x32 pixels
        transforms.ToTensor(),  # Convert images to PyTorch tensors
    ])
    return trans

trans = trans()  # Instantiate the defined transformations

# Download and create datasets
train_data = datasets.MNIST(root='./data', train=True, transform=trans, download=True)  # Download and create training dataset
test_data = datasets.MNIST(root='./data', train=False, transform=trans)  # Download and create test dataset

# Define data loaders
train_load = DataLoader(dataset=train_data, batch_size=32, shuffle=True)  # Create DataLoader for training dataset
test_load = DataLoader(dataset=test_data, batch_size=32, shuffle=False)  # Create DataLoader for test dataset

# LeNet-5 Model
class LNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),  # First convolutional layer with 6 filters
            nn.Tanh(),  # Tanh activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling layer
            nn.Conv2d(6, 16, kernel_size=5),  # Second convolutional layer with 16 filters
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),  # Third convolutional layer with 120 filters
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),  # First fully connected layer with 84 neurons
            nn.Tanh(),
            nn.Linear(84, num_classes)  # Output layer with num_classes neurons
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training and evaluation functions
def train_model(model, train_load, criteria, optimize, epochs):
    """Train the model and evaluate on the test set."""
    train_loss = []  # to store training losses over epochs
    test_accuracy = []  # to store test accuracies over epochs

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

        # Evaluate the test accuracy
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

# Instantiate the model
model = LNet5().to(device)

# Loss function and optimize
criteria = nn.CrossEntropyLoss()
optimize = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
train_losses, test_accuracies = train_model(model, train_load, criteria, optimize, epochs)

# Save the trained model
torch.save(model.state_dict(), 'lenet5_model.pth')

# Plotting the training losses
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the test accuracies
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Test Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
