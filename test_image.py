import torch
from PIL import Image
from model import ChestXRayClassifier
from config import IMG_SIZE, NUM_CLASSES
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChestXRayClassifier(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

img = Image.open(r"C:\Users\khoba\Desktop\Medical_AI_V0.02\chest_xray\test\PNEUMONIA\123.jpeg").convert('RGB')
img_np = np.array(img)
tensor = transform(image=img_np)['image'].unsqueeze(0).to(device)

with torch.no_grad():
    probs = torch.sigmoid(model(tensor)).cpu().numpy()[0]

class_names = ["Pneumonia", "Effusion", "Atelectasis", "Cardiomegaly", "Infiltration"]
for name, prob in zip(class_names, probs):
    print(f"{name}: {prob:.4f}")