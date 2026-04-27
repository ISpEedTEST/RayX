import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Import classes dynamically from your dataset file
from nih_dataset import classes

print("\nStarting the Medical AI Diagnostic System...")

# 1. Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Rebuild the Brain Architecture
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))

# 3. Load the trained Multi-Label weights
model.load_state_dict(torch.load('nih_multilabel_model.pth', map_location=device))
model = model.to(device)
model.eval() # Switch to testing mode
print("AI Brain loaded successfully!\n")

# 4. Image Preprocessing (Must match training exactly)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. Get Image Path from User
image_path = input("Please paste the full path of the X-ray image: ").strip().replace('"', '')

if not os.path.exists(image_path):
    print(f"Error: Could not find the file at {image_path}")
else:
    print("-" * 40)
    print(f"Analyzing Image: {os.path.basename(image_path)}")
    print("Calculating probabilities for multiple diseases...\n")

    # 6. Process Image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 7. Make the Diagnosis
    with torch.no_grad():
        outputs = model(input_tensor)
        # CRITICAL: Use Sigmoid for Multi-Label (not Softmax)
        probabilities = torch.sigmoid(outputs[0])
        
    # 8. Display Results neatly
    print("🩺 AI DIAGNOSTIC REPORT:")
    print("-" * 40)
    
    # Pair classes with their probabilities
    results = []
    for i, class_name in enumerate(classes):
        prob = probabilities[i].item() * 100
        results.append((class_name, prob))
        
    # Sort from highest probability to lowest
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Print the findings
    for class_name, prob in results:
        if prob > 50.0:
            print(f"🚨 {class_name}: {prob:.2f}% (HIGH RISK)")
        elif prob > 20.0:
            print(f"⚠️ {class_name}: {prob:.2f}% (Possible)")
        else:
            print(f"✅ {class_name}: {prob:.2f}% (Unlikely)")
            
    print("-" * 40)