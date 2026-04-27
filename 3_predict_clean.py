import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

print("⏳ Loading Brain V2...")

# 1. Disease Classes
classes = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# 2. Environment Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))

# 3. Load Brain V2
model_path = 'nih_multilabel_model_v2.pth'
try:
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print("✅ Brain V2 Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit()

model = model.to(device)
model.eval() 

# 4. Image Transformer
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. Smart Prediction Function
def examine_xray(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"❌ Error: Could not find the image at this path: {image_path}")
        return
    except Exception as e:
        print(f"❌ Error opening the image: {e}")
        return

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs)[0] * 100 

    # Extract filename for the report
    file_name = os.path.basename(image_path)
    
    print(f"\n🩺 Medical Report for: {file_name}")
    print("=" * 40)
    
    results = [(classes[i], probs[i].item()) for i in range(len(classes))]
    results.sort(key=lambda x: x[1], reverse=True)

    for disease, prob in results:
        if prob > 50:
            print(f"🚨 {disease}: {prob:.2f}% (HIGH RISK)")
        elif prob > 20:
            print(f"⚠️ {disease}: {prob:.2f}% (Moderate)")
        else:
            print(f"✅ {disease}: {prob:.2f}% (Unlikely)")
    print("=" * 40)

# ==========================================
# 🧪 Interactive Clinic Loop
# ==========================================

print("\n" + "="*50)
print("🏥 Welcome to the Smart AI Clinic!")
print("💡 Tip: You can drag and drop the X-ray image directly here.")
print("🛑 Type 'q' or 'exit' to close the program.")
print("="*50)

while True:
    user_input = input("\n📁 Paste image path here: ").strip()
    
    # Exit commands
    if user_input.lower() in ['q', 'exit', 'quit']:
        print("👋 Goodbye! Clinic is now closed.")
        break
        
    # Skip if user accidentally hits Enter
    if not user_input:
        continue
        
    # Clean the path from Windows drag-and-drop quotes
    clean_path = user_input.strip('"').strip("'")
    
    # Examine the image
    examine_xray(clean_path)