import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import os

# Import the dataset loader
from nih_dataset import NIHDistribution, data_frame, image_dir, classes

print("Preparing to resume Multi-Label training...")

# 1. Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load the Custom Dataset
train_dataset = NIHDistribution(dataframe=data_frame, image_dir=image_dir, classes=classes, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 4. Rebuild the Brain (ResNet18)
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))

# 5. Load your previously trained NIH brain
model_path = 'nih_multilabel_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Previous NIH model loaded successfully! Continuing education...")
else:
    print(f"Error: Could not find {model_path}. You need to run train_multilabel.py first.")
    exit()

model = model.to(device)

# 6. Loss function and Optimizer (Lower learning rate for fine-tuning)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 7. Start Training Loop
epochs = 4
print(f"Training started on GPU: {torch.cuda.get_device_name(0)}")
print("-" * 40)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Batch [{i+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

    print("-" * 40)
    print(f"Epoch {epoch+1} finished! Average Loss: {running_loss/len(train_loader):.4f}")

# 8. Overwrite the old brain with the new upgraded one
torch.save(model.state_dict(), 'nih_multilabel_model.pth')
print("Upgraded model saved successfully as: nih_multilabel_model.pth")