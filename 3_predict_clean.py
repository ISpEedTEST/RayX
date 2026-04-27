# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import sys
# import os

# print("⏳ Loading Brain V2...")

# # 1. Disease Classes
# classes = [
#     'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
#     'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
#     'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
# ]

# # 2. Environment Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet18(weights=None)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, len(classes))

# # 3. Load Brain V2
# model_path = 'nih_multilabel_model_v2.pth'
# try:
#     model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
#     print("✅ Brain V2 Loaded Successfully!")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")
#     sys.exit()

# model = model.to(device)
# model.eval() 

# # 4. Image Transformer
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # 5. Smart Prediction Function
# def examine_xray(image_path):
#     try:
#         image = Image.open(image_path).convert('RGB')
#     except FileNotFoundError:
#         print(f"❌ Error: Could not find the image at this path: {image_path}")
#         return
#     except Exception as e:
#         print(f"❌ Error opening the image: {e}")
#         return

#     input_tensor = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = model(input_tensor)
#         probs = torch.sigmoid(outputs)[0] * 100 

#     # Extract filename for the report
#     file_name = os.path.basename(image_path)
    
#     print(f"\n🩺 Medical Report for: {file_name}")
#     print("=" * 40)
    
#     results = [(classes[i], probs[i].item()) for i in range(len(classes))]
#     results.sort(key=lambda x: x[1], reverse=True)

#     for disease, prob in results:
#         if prob > 50:
#             print(f"🚨 {disease}: {prob:.2f}% (HIGH RISK)")
#         elif prob > 20:
#             print(f"⚠️ {disease}: {prob:.2f}% (Moderate)")
#         else:
#             print(f"✅ {disease}: {prob:.2f}% (Unlikely)")
#     print("=" * 40)

# # ==========================================
# # 🧪 Interactive Clinic Loop
# # ==========================================

# print("\n" + "="*50)
# print("🏥 Welcome to the Smart AI Clinic!")
# print("💡 Tip: You can drag and drop the X-ray image directly here.")
# print("🛑 Type 'q' or 'exit' to close the program.")
# print("="*50)

# while True:
#     user_input = input("\n📁 Paste image path here: ").strip()
    
#     # Exit commands
#     if user_input.lower() in ['q', 'exit', 'quit']:
#         print("👋 Goodbye! Clinic is now closed.")
#         break
        
#     # Skip if user accidentally hits Enter
#     if not user_input:
#         continue
        
#     # Clean the path from Windows drag-and-drop quotes
#     clean_path = user_input.strip('"').strip("'")
    
#     # Examine the image
#     examine_xray(clean_path)


import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from PIL import Image
import sys
import os
import time

print("⏳ Loading RayX Brain V2 (TTA Enabled)...")

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

# 4. Image Transformer (Base)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. Smart Prediction Function with TTA
def examine_xray_tta(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"❌ Error: Could not find the image at this path: {image_path}")
        return
    except Exception as e:
        print(f"❌ Error opening the image: {e}")
        return

    print("\n🔍 Initiating TTA Deep Scan...")
    time.sleep(0.5) # توقف خفيف جداً لترتيب طباعة النصوص في الشاشة
    
    # 1. الصورة الأصلية
    print("⚙️ 1. Analyzing original view...")
    t0 = transform(image)
    
    # 2. زوايا الميلان
    print("📐 2. Analyzing structural variations (Angles)...")
    t1 = transform(TF.rotate(image, 3))
    t2 = transform(TF.rotate(image, -3))
    
    # 3. تباين الإضاءة
    print("💡 3. Analyzing contrast variations (Tissue Density)...")
    t3 = transform(TF.adjust_brightness(image, 1.1))
    t4 = transform(TF.adjust_brightness(image, 0.9))
    
    # 4. دمج الصور وحساب الاحتمالات
    print("🧠 4. Fusing AI predictions...")
    
    # كرت الشاشة (RTX 4060) سيستقبل مصفوفة من 5 صور دفعة واحدة
    input_batch = torch.stack([t0, t1, t2, t3, t4]).to(device)

    with torch.no_grad():
        outputs = model(input_batch)
        probs = torch.sigmoid(outputs) 
        
        # أخذ المتوسط الحسابي للـ 5 قراءات
        final_probs = probs.mean(dim=0) * 100 

    file_name = os.path.basename(image_path)
    
    print(f"\n🩺 RayX Medical Report for: {file_name}")
    print("=" * 45)
    
    results = [(classes[i], final_probs[i].item()) for i in range(len(classes))]
    results.sort(key=lambda x: x[1], reverse=True)

    for disease, prob in results:
        if prob > 50:
            print(f"🚨 {disease}: {prob:.2f}% (HIGH RISK)")
        elif prob > 20:
            print(f"⚠️ {disease}: {prob:.2f}% (Moderate)")
        else:
            print(f"✅ {disease}: {prob:.2f}% (Unlikely)")
    print("=" * 45)

# ==========================================
# 🧪 Interactive Terminal Clinic
# ==========================================

print("\n" + "="*50)
print("🏥 Welcome to the RayX AI Terminal!")
print("💡 Tip: You can drag and drop the X-ray image directly here.")
print("🛑 Type 'q' or 'exit' to close the program.")
print("="*50)

while True:
    user_input = input("\n📁 Paste image path here: ").strip()
    
    if user_input.lower() in ['q', 'exit', 'quit']:
        print("👋 Goodbye! RayX Terminal closed.")
        break
        
    if not user_input:
        continue
        
    clean_path = user_input.strip('"').strip("'")
    
    examine_xray_tta(clean_path)