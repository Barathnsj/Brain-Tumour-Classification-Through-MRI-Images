import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt

# Paths
data_dir = "./BrainTumor_1"
train_dir = os.path.join(data_dir, "Train")
val_dir = os.path.join(data_dir, "Test")
model_path = "brain_tumor_model.pth"
csv_output_path = "classification_report.csv"

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transformations
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms["train"])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms["val"])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Class names
class_names = train_dataset.classes
print(f"Classes: {class_names}")

# Define model
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Check if model exists
if os.path.exists(model_path):
    print("Loading pre-trained model...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
else:
    print("No pre-trained model found. Training is required.")

    # Training function
    def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            # Training phase
            model.train()
            train_loss, train_correct = 0.0, 0
            for inputs, labels in dataloaders["train"]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()

            train_loss /= len(dataloaders["train"].dataset)
            train_acc = train_correct / len(dataloaders["train"].dataset)
            train_losses.append(train_loss)

            # Validation phase
            model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for inputs, labels in dataloaders["val"]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()

            val_loss /= len(dataloaders["val"].dataset)
            val_acc = val_correct / len(dataloaders["val"].dataset)
            val_losses.append(val_loss)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        return train_losses, val_losses

    # Train the model
    dataloaders = {"train": train_loader, "val": val_loader}
    train_losses, val_losses = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)

    # Save the model
    torch.save(model.state_dict(), model_path)
    print("Model saved!")

    # Plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

# Evaluate the model
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Classification report
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Save classification report to CSV
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(csv_output_path, index=True)
print(f"Classification report saved to {csv_output_path}")