import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Import Deep Analysis Library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

print("⏳ Initializing Deep Analysis System (Grad-CAM)...")

# 1. Disease Classes
classes = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# 2. Load Your Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load('nih_multilabel_model_v2.pth', map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# 3. Define Target Layer
target_layers = [model.layer4[-1]]

# 4. Image Transformer
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def analyze_image_deeply(image_path):
    try:
        original_img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    input_tensor = transform(original_img).unsqueeze(0).to(device)

    # 1. Standard Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs)[0] * 100
        
    # Print Full Report First to see the truth
    print("\n🩺 Complete AI Scan Results:")
    results = [(classes[i], probs[i].item()) for i in range(len(classes))]
    results.sort(key=lambda x: x[1], reverse=True)
    
    for disease, prob in results[:4]: # Show top 4 possibilities
        print(f" - {disease}: {prob:.2f}%")

    # 2. Smart Target Selection Logic for Heatmap
    no_finding_idx = classes.index('No Finding')
    
    # Clone probabilities and ignore 'No Finding' to force anomaly detection
    anomaly_probs = probs.clone()
    anomaly_probs[no_finding_idx] = -1.0 
    
    target_idx = torch.argmax(anomaly_probs).item()
    target_name = classes[target_idx]
    target_prob = probs[target_idx].item()
    
    # If the highest anomaly is extremely low (< 15%) AND No Finding is high, then it's truly healthy
    if target_prob < 15.0 and probs[no_finding_idx].item() > 50.0:
        target_idx = no_finding_idx
        target_name = 'No Finding'
        target_prob = probs[no_finding_idx].item()
        print("\n✅ Conclusion: Lungs appear mostly clear. No significant anomaly to map.")
    else:
        print(f"\n🧠 Generating Heatmap for the primary anomaly: {target_name} ({target_prob:.2f}%)")

    # 3. Deep Analysis (Generate Heatmap)
    targets = [ClassifierOutputTarget(target_idx)]
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
    # 4. Overlay Heatmap
    rgb_img = np.array(original_img.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 5. Save and Display
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original X-Ray")
    plt.imshow(rgb_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Grad-CAM Target:\n{target_name} ({target_prob:.1f}%)")
    plt.imshow(visualization)
    plt.axis('off')
    
    plt.tight_layout()
    output_filename = 'deep_analysis_result.png'
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Map saved as: '{output_filename}'")
    plt.show()

# =======================
# Run
# =======================
image_to_test = input("\n📁 Paste X-ray image path here for deep analysis: ").strip().strip('"').strip("'")
if image_to_test:
    analyze_image_deeply(image_to_test)