# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import models, transforms
# from torch.utils.data import DataLoader

# # Import the dataset loader we built earlier
# from nih_dataset import NIHDistribution, data_frame, image_dir, classes

# print("Setting up Multi-Label Training Environment...")

# # 1. Setup device (GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 2. Image Preprocessing
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # 3. Load the Custom Dataset
# train_dataset = NIHDistribution(dataframe=data_frame, image_dir=image_dir, classes=classes, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# # 4. Build the Brain (ResNet18) for Multi-Label
# model = models.resnet18(weights='DEFAULT')
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, len(classes)) # Dynamic output size based on found diseases
# model = model.to(device)

# # 5. Loss function and Optimizer
# criterion = nn.BCEWithLogitsLoss() # Magic equation for multiple diseases at once
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 6. Start Training Loop
# epochs = 1 # 3 epochs as a starting point
# print(f"Training started on GPU: {torch.cuda.get_device_name(0)}")
# print(f"Learning to detect {len(classes)} diseases simultaneously...")
# print("-" * 40)

# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
    
#     for i, (inputs, labels) in enumerate(train_loader):
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
        
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
        
#         if (i+1) % 10 == 0:
#             print(f"Epoch [{epoch+1}/{epochs}] | Batch [{i+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

#     print("-" * 40)
#     print(f"Epoch {epoch+1} finished! Average Loss: {running_loss/len(train_loader):.4f}")

# # 7. Save the Multi-Label Model
# torch.save(model.state_dict(), 'nih_multilabel_model.pth')
# print("Model saved successfully as: nih_multilabel_model.pth")



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader

from nih_dataset import NIHDistribution, data_frame, image_dir, classes

print("Setting up Clean Multi-Label Training Environment...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = NIHDistribution(dataframe=data_frame, image_dir=image_dir, classes=classes, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# طباعة عدد الصور التي سيتدرب عليها للتأكد أن المجلدات الجديدة تمت إضافتها
print("-" * 40)
print(f"Total images loaded for training: {len(train_dataset)}")
print("-" * 40)

model = models.resnet18(weights='DEFAULT')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes)) 
model = model.to(device)

# عدنا للدالة الطبيعية بدون أوزان عقابية مجنونة
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4 دورات ستكون ممتازة جداً لـ 15 أو 20 ألف صورة
epochs = 4 
print(f"Training started on GPU: {torch.cuda.get_device_name(0)}")

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

torch.save(model.state_dict(), 'nih_multilabel_model.pth')
print("Model saved successfully as: nih_multilabel_model.pth")