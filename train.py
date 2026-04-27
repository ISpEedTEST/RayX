import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

print("Setting up the training environment and data...")

# 1. Enable GPU for faster training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data processing: Resize images and convert to Tensors
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load images from directory and create batches (32 images per batch)
train_dir = "./chest_xray/train"
train_data = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 4. Load the pre-trained model (ResNet18) and modify it for our project
model = models.resnet18(weights='DEFAULT')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) # Modify output to 2 classes (Normal / Pneumonia)
model = model.to(device) # Move model to GPU

# 5. Define the Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Start the Training Loop
epochs = 20 # Training for 20 epochs as a start
print(f"Training started on GPU: {torch.cuda.get_device_name(0)}")
print("-" * 40)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Iterate over data batches
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass, calculate loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass, optimize (Real learning happens here)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print updates every 10 batches
        if (i+1) % 10 == 0:
            print(f"Batch [{i+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

    # Final epoch accuracy
    epoch_acc = 100 * correct / total
    print("-" * 40)
    print(f"Epoch {epoch+1} finished! | Current Model Accuracy: {epoch_acc:.2f}%")

# 7. Save the trained model's brain
torch.save(model.state_dict(), 'medical_ai_model.pth')
print("Model saved successfully as: medical_ai_model.pth")