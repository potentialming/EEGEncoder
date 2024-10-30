import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model_imagenet import EEGEncoder  # Import the model
import numpy as np
from dataset import create_EEG_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import os
from datetime import datetime

# Hyperparameters
batch_size = 32
learning_rate = 3e-4
num_epochs = 200
patience = 15  # Early stopping patience
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
data_path = './dataset/ImageNet/EEG/eeg_5_95_std.pth'
images_path = './dataset/ImageNet/imageNet_images'
split_file = './dataset/ImageNet/block_splits_by_image_all.pth'
outputdir = './results/imageNet'  # Output directory for saving results

# Create a new directory based on the current time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join(outputdir, current_time)
os.makedirs(save_dir, exist_ok=True)

# Image transformations (if necessary)
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images if needed
    transforms.ToTensor(),          # Convert images to tensor
])

train_dataset, test_dataset = create_EEG_dataset(eeg_signals_path=data_path, splits_path=split_file, 
        image_transform=image_transforms, subject=0)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = EEGEncoder().to(device)
criterion = nn.CrossEntropyLoss()  # Assuming a classification task
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Add learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Lists for losses and accuracies
train_losses = []
val_losses = []
val_accuracies = []
best_val_loss = np.inf
patience_counter = 0
best_accuracy = 0
best_epoch = 0

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, batch in enumerate(train_loader):
        eeg_data, labels = batch["eeg"].to(device), batch["labels"].to(device)

        # Forward pass
        outputs = model(eeg_data)  # Get model predictions
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))  # Compute loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return epoch_loss, accuracy

# Validation function
def validate(model, test_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            eeg_data, labels = batch["eeg"].to(device), batch["labels"].to(device)

            # Forward pass
            outputs = model(eeg_data)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            val_loss += loss.item()

            # Statistics
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = val_loss / len(test_loader)
    accuracy = 100. * correct / total
    return epoch_loss, accuracy, np.concatenate(all_preds), np.concatenate(all_labels)

# Function to compute top-k accuracy
def top_k_accuracy(output, target, k):
    with torch.no_grad():
        max_k = torch.topk(output, k, dim=1)[1]  # Get top-k predictions
        correct = max_k.eq(target.view(-1, 1).expand_as(max_k))  # Compare to true labels
        return correct.float().sum().item() / target.size(0)

# Main training loop with early stopping
for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    
    # Training
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    
    # Validation
    val_loss, val_acc, preds, labels = validate(model, test_loader, criterion, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    # Adjust learning rate based on validation loss
    scheduler.step(val_loss)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_accuracy = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(save_dir, "best_eeg_encoder.pth"))
        best_preds = preds
        best_labels = labels
        print(f"Validation loss improved. Saved Best Model with Accuracy: {best_accuracy:.2f}%")
    else:
        patience_counter += 1
        print(f"Validation loss did not improve for {patience_counter} epoch(s).")
        
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Load the best model for evaluation
model.load_state_dict(torch.load(os.path.join(save_dir, "best_eeg_encoder.pth")))
model.eval()

# Calculate Top-1, Top-3, Top-5 accuracy and F1-score on the test set
top1_correct = 0
top3_correct = 0
top5_correct = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for eeg_data, labels in test_loader:
        eeg_data, labels = eeg_data.to(device), labels.to(device)
        outputs = model(eeg_data)
        
        top1_correct += top_k_accuracy(outputs, labels, 1)
        top3_correct += top_k_accuracy(outputs, labels, 3)
        top5_correct += top_k_accuracy(outputs, labels, 5)
        
        _, preds = outputs.max(1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Calculate F1-score
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
f1 = f1_score(all_labels, all_preds, average='weighted')

# Calculate final accuracies
top1_accuracy = top1_correct / len(test_loader)
top3_accuracy = top3_correct / len(test_loader)
top5_accuracy = top5_correct / len(test_loader)

# Save results to a text file
results_file = os.path.join(save_dir, 'results.txt')
with open(results_file, 'w') as f:
    f.write(f"Best Model (Epoch {best_epoch+1}):\n")
    f.write(f"Top-1 Accuracy: {top1_accuracy:.4f}\n")
    f.write(f"Top-3 Accuracy: {top3_accuracy:.4f}\n")
    f.write(f"Top-5 Accuracy: {top5_accuracy:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

# Plot training and validation loss and save the figure
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()
plt.savefig(os.path.join(save_dir, 'train_val_loss.png'))  # Save the plot
plt.show()

# Confusion Matrix for best validation epoch and save the figure
conf_matrix = confusion_matrix(best_labels, best_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (Test Set) (Epoch {best_epoch+1})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))  # Save the plot
plt.show()

print(f"Training Complete. Results saved to {save_dir}")
