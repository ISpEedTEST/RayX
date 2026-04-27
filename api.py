# api.py
import io
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
from model import ChestXRayClassifier
from gradcam import grad_cam, overlay_heatmap, encode_to_base64
from config import NUM_CLASSES, DEVICE, IMG_SIZE
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Global variables
model = None
device = None
class_names = ["Pneumonia", "Effusion", "Atelectasis", "Cardiomegaly", "Infiltration"]
transform = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device, transform
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = ChestXRayClassifier(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    print(f"Model loaded on {device}")
    yield
    print("Shutting down API...")

app = FastAPI(
    title="ChestXray AI Assistant (5 diseases)",
    version="1.0",
    description="Assistive multi-label classification",
    lifespan=lifespan
)

def preprocess_image(image: Image.Image):
    image = image.convert('RGB')
    img_np = np.array(image)
    augmented = transform(image=img_np)
    return augmented['image'].unsqueeze(0)

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_n: int = 5):
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        input_tensor = preprocess_image(pil_image).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Top-N predictions
        top_indices = np.argsort(probs)[::-1][:top_n]
        top_predictions = [
            {"disease": class_names[idx], "confidence": float(probs[idx])}
            for idx in top_indices if probs[idx] > 0.1
        ]

        # Grad-CAM for the top disease (with safety)
        cam_heatmap = None
        if top_indices.size > 0:
            try:
                top_idx = top_indices[0]
                # Need to recompute forward with gradient for Grad-CAM
                model.zero_grad()
                cam = grad_cam(model, input_tensor, top_idx, device)
                overlayed = overlay_heatmap(pil_image, cam)
                cam_heatmap = encode_to_base64(overlayed)
            except Exception as e:
                print(f"Heatmap generation failed: {e}")
                # Continue without heatmap

        return {
            "top_predictions": top_predictions,
            "all_scores": {class_names[i]: float(probs[i]) for i in range(NUM_CLASSES)},
            "heatmap": cam_heatmap,
            "disclaimer": "⚠️ Assistive tool only. Not for diagnostic use. Final diagnosis must be made by a qualified radiologist."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}