import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import sys

from nih_dataset import NIHDistribution, data_frame, classes

# حماية الكود الخاصة بنظام ويندوز لتعمل الـ num_workers بسلام
if __name__ == '__main__':
    print("🚀 Initializing Clean Training Engine on RTX 4060...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = NIHDistribution(dataframe=data_frame, transform=transform)
    # العمال الـ 4 سيعملون الآن بأمان وسرعة فائقة
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    print("-" * 50)
    print(f"🔥 Ready to train on {len(train_dataset)} images!")
    print("-" * 50)

    model = models.resnet18(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes)) 
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    epochs = 5
    print(f"⚙️ Training started on GPU: {torch.cuda.get_device_name(0)}")
    print("💡 Tip: You can press Ctrl+C anytime to safely stop and save the model.")
    print("-" * 50)

    try:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # -------- MAGIC MASKING LOGIC --------
                mask = (labels != -1.0).float()
                safe_labels = labels.clone()
                safe_labels[safe_labels == -1.0] = 0.0
                
                raw_loss = criterion(outputs, safe_labels)
                masked_loss = raw_loss * mask
                
                loss = masked_loss.sum() / (mask.sum() + 1e-8)
                # --------------------------------------
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if (i+1) % 20 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] | Batch [{i+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

            avg_loss = running_loss/len(train_loader)
            print("=" * 50)
            print(f"✅ Epoch {epoch+1} finished! Average Loss: {avg_loss:.4f}")
            
            torch.save(model.state_dict(), 'nih_multilabel_model_v2.pth')
            print(f"💾 Checkpoint saved: nih_multilabel_model_v2.pth")
            print("=" * 50)

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user (Ctrl+C)!")
        torch.save(model.state_dict(), 'nih_multilabel_model_v2.pth')
        print("✅ Emergency Save Complete: All learning up to this point is saved.")
        sys.exit(0)

    print("🎉 Full Training Completed Successfully!")