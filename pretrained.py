import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset paths
data_dir = "./BrainTumor_1"  # Adjust this path if necessary
train_dir = os.path.join(data_dir, "Train")
test_dir = os.path.join(data_dir, "Test")

# Data transformations (Reduced augmentation for speed)
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms["train"])
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms["test"])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Increased batch size
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)   # Increased batch size

# Class names
class_names = train_dataset.classes
print(f"Classes: {class_names}")

# Define model (Using pre-trained ResNet-18)
model = models.resnet18(weights="IMAGENET1K_V1")  # Load pre-trained weights (updated API)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))  # Adjust output layer to match number of classes
model = model.to(device)

# Freeze all layers except the final layer (for transfer learning)
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True  # Unfreeze the final layer

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Pass only the parameters of the final layer to the optimizer
optimizer = Adam(model.fc.parameters(), lr=0.001)

# Filepath to save the model
model_path = "brain_tumor_model_pretrained.pth"

# Load or train the model
if os.path.exists(model_path):
    print("Loading the saved pre-trained model...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
else:
    print("Training the model...")
    # Training function
    def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
        train_losses, test_losses = [], []

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            # Training phase
            model.train()
            train_loss, train_correct = 0.0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()

            train_loss /= len(train_loader.dataset)
            train_acc = train_correct / len(train_loader.dataset)
            train_losses.append(train_loss)

            # Testing phase
            model.eval()
            test_loss, test_correct = 0.0, 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item() * inputs.size(0)
                    test_correct += (outputs.argmax(1) == labels).sum().item()

            test_loss /= len(test_loader.dataset)
            test_acc = test_correct / len(test_loader.dataset)
            test_losses.append(test_loss)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        return train_losses, test_losses

    # Train the model
    train_losses, test_losses = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10)

    # Save the model
    torch.save(model.state_dict(), model_path)
    print("Model saved!")

# Evaluate the model
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Plot the losses
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
