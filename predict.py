import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

print("Loading the AI brain for diagnosis...")

# 1. Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Rebuild the model architecture
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# 3. Load your trained brain into the model
model.load_state_dict(torch.load('medical_ai_model.pth', map_location=device))
model = model.to(device)
model.eval()
print("Model loaded successfully!\n")

# 4. Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. Ask the user for a specific image path
image_path = input("Please paste the full path of the X-ray image: ")

# Clean up the path (Windows sometimes adds quotes around copied paths)
image_path = image_path.strip().replace('"', '')

if not os.path.exists(image_path):
    print(f"Error: Could not find the file at {image_path}")
else:
    print("-" * 40)
    print(f"Analyzing Image: {os.path.basename(image_path)}")

    # 6. Load and process the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 7. Make the Diagnosis!
    classes = ['NORMAL', 'PNEUMONIA']

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        diagnosis = classes[predicted_idx.item()]
        confidence_percent = confidence.item() * 100

    print("-" * 40)
    print(f"AI DIAGNOSIS: {diagnosis}")
    print(f"CONFIDENCE: {confidence_percent:.2f}%")
    print("-" * 40)