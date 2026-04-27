import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

# Grad-CAM Imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import your model class (must be in same folder)
from model import ChestXRayClassifier
from config import NUM_CLASSES, IMG_SIZE

# -------------------------------------------------------------------
# 1. Page Config
st.set_page_config(page_title="RayX AI (5 Diseases)", page_icon="🩺", layout="centered")

# 2. UI Dictionary (same style, but adapted for 5 diseases)
lang = st.sidebar.selectbox("Select Language / اختر اللغة", ["English", "العربية"])

ui = {
    "English": {
        "title": "🩺 RayX AI - Chest X-Ray Analyzer (5 Diseases)",
        "desc": "Upload a Chest X-ray image for AI analysis of 5 common thoracic diseases: Pneumonia, Effusion, Atelectasis, Cardiomegaly, Infiltration.",
        "upload_label": "📂 Drag and drop image here or click to browse",
        "btn_scan": "🔍 Start Deep Chest Analysis",
        "status_start": "Initiating Two-Pass Verification Scan...",
        "status_1": "Phase 1: Global Thoracic Scan (TTA)...",
        "status_2": "Phase 2: Zoomed-in Targeted Verification...",
        "status_3": "Cross-referencing Phase 1 & 2 results...",
        "status_4": "Generating Explainable AI Heatmap...",
        "status_done": "✅ Double-Verification Complete!",
        "success": "✅ Analysis Complete! Medical Report:",
        "risk_high": "HIGH RISK",
        "risk_med": "Moderate",
        "risk_low": "Unlikely",
        "heatmap_title": "🧠 Explainable AI (Grad-CAM Focus Map)",
        "heatmap_desc": "The AI focused on the red/yellow areas to make its diagnosis.",
        "clear_lungs": "✅ Lungs appear mostly clear. No significant anomaly to map.",
        "original_img": "Original X-Ray",
        "ai_img": "AI Heatmap",
        "footer": "Trained on NIH ChestX-ray (5 diseases). This is an assistive tool only. Please consult a radiologist for professional advice."
    },
    "العربية": {
        "title": "🩺 RayX AI (التحليل الصدري – 5 أمراض)",
        "desc": "قم برفع صورة أشعة سينية للصدر لتحليل 5 أمراض صدرية شائعة: الالتهاب الرئوي، الانصباب، الانخماص، تضخم القلب، الارتشاح.",
        "upload_label": "📂 اسحب وأفلت الصورة هنا أو اضغط للاختيار",
        "btn_scan": "🔍 ابدأ الفحص العميق",
        "status_start": "بدء نظام التحقق المزدوج...",
        "status_1": "المرحلة الأولى: المسح الشامل للقفص الصدري...",
        "status_2": "المرحلة الثانية: التركيز والتحقق العميق من الأنسجة...",
        "status_3": "مقاطعة نتائج المرحلتين لحساب الدقة النهائية...",
        "status_4": "توليد الخريطة الحرارية للأدلة البصرية...",
        "status_done": "✅ اكتمل التحقق المزدوج بنجاح!",
        "success": "✅ اكتمل الفحص! إليك التقرير الطبي:",
        "risk_high": "خطر مرتفع",
        "risk_med": "احتمال متوسط",
        "risk_low": "سليم / مستبعد",
        "heatmap_title": "🧠 التحليل البصري العميق (Grad-CAM)",
        "heatmap_desc": "المناطق الملونة بالأحمر/الأصفر هي ما اعتمد عليه الذكاء الاصطناعي لاكتشاف المرض.",
        "clear_lungs": "✅ الرئة تبدو سليمة تماماً. لا توجد مناطق مرضية لتحديدها.",
        "original_img": "الأشعة الأصلية",
        "ai_img": "خريطة التحليل الحرارية",
        "footer": "تم تدريبه على قاعدة بيانات NIH (5 أمراض). هذا نظام مساعد فقط، يرجى مراجعة أخصائي الأشعة للحصول على استشارة طبية."
    }
}

st.title(ui[lang]["title"])
st.write(ui[lang]["desc"])

# -------------------------------------------------------------------
# 3. Class names (5 diseases) and translations
classes = ['Pneumonia', 'Effusion', 'Atelectasis', 'Cardiomegaly', 'Infiltration']

disease_translation = {
    "English": {cls: cls for cls in classes},
    "العربية": {
        'Pneumonia': 'التهاب رئوي',
        'Effusion': 'انصباب سوائل',
        'Atelectasis': 'انخماص الرئة',
        'Cardiomegaly': 'تضخم القلب',
        'Infiltration': 'ارتشاح'
    }
}

# -------------------------------------------------------------------
# 4. Load your trained model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChestXRayClassifier(num_classes=NUM_CLASSES, dropout=0.2)
    # Load weights (map_location handles CPU/GPU)
    model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# -------------------------------------------------------------------
# 5. Image transforms (same as training: Resize 224, ImageNet normalization)
transform_base = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_zoomed_image(img):
    """Crop 85% of the image (central) for second-pass verification"""
    width, height = img.size
    new_width = int(width * 0.85)
    new_height = int(height * 0.85)
    return TF.center_crop(img, [new_height, new_width])

# -------------------------------------------------------------------
# 6. Main App UI
uploaded_file = st.file_uploader(ui[lang]["upload_label"], type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)

    if st.button(ui[lang]["btn_scan"]):
        with st.status("🔍 " + ui[lang]["status_start"], expanded=True) as status:
            # ----- Phase 1: Global scan with test-time augmentation (TTA) -----
            st.write("🌍 " + ui[lang]["status_1"])
            t0 = transform_base(image)
            t1 = transform_base(TF.rotate(image, 3))
            t2 = transform_base(TF.rotate(image, -3))
            t3 = transform_base(TF.adjust_brightness(image, 1.1))
            t4 = transform_base(TF.adjust_brightness(image, 0.9))
            batch_phase1 = torch.stack([t0, t1, t2, t3, t4]).to(device)

            with torch.no_grad():
                out_phase1 = model(batch_phase1)        # raw logits
                probs_phase1 = torch.sigmoid(out_phase1).mean(dim=0) * 100

            # ----- Phase 2: Zoomed verification -----
            st.write("🔬 " + ui[lang]["status_2"])
            zoomed_img = get_zoomed_image(image)
            z0 = transform_base(zoomed_img)
            z1 = transform_base(TF.rotate(zoomed_img, 3))
            z2 = transform_base(TF.rotate(zoomed_img, -3))
            batch_phase2 = torch.stack([z0, z1, z2]).to(device)

            with torch.no_grad():
                out_phase2 = model(batch_phase2)
                probs_phase2 = torch.sigmoid(out_phase2).mean(dim=0) * 100

            # ----- Phase 3: Cross-referencing (average) -----
            st.write("⚖️ " + ui[lang]["status_3"])
            final_probs = (probs_phase1 + probs_phase2) / 2.0

            results = [(classes[i], final_probs[i].item()) for i in range(len(classes))]
            results.sort(key=lambda x: x[1], reverse=True)

            # ----- Phase 4: Grad-CAM (on the most probable disease) -----
            st.write("🧠 " + ui[lang]["status_4"])

            # Determine which disease has the highest probability (ignore threshold)
            target_idx = torch.argmax(final_probs).item()
            target_prob = final_probs[target_idx].item()

            generate_heatmap = True
            # If the highest probability is below 15% (very low confidence), skip heatmap
            if target_prob < 15.0:
                generate_heatmap = False
            else:
                # Use the last convolutional layer of DenseNet for Grad-CAM
                target_layers = [model.backbone.features.norm5]   # as used in training
                targets = [ClassifierOutputTarget(target_idx)]
                input_tensor = t0.unsqueeze(0).to(device)

                with GradCAM(model=model, target_layers=target_layers) as cam:
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                    grayscale_cam = grayscale_cam[0, :]

                # Prepare original image for overlay (resized to 224x224)
                rgb_img = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            status.update(label=ui[lang]["status_done"], state="complete", expanded=False)

        # ----- Display Results -----
        st.success(ui[lang]["success"])
        for disease, prob in results:
            display_name = disease_translation[lang][disease]
            if prob > 50:
                st.error(f"🚨 {display_name}: {prob:.2f}% ({ui[lang]['risk_high']})")
            elif prob > 20:
                st.warning(f"⚠️ {display_name}: {prob:.2f}% ({ui[lang]['risk_med']})")
            else:
                st.success(f"✅ {display_name}: {prob:.2f}% ({ui[lang]['risk_low']})")

        # ----- Display Heatmap Section -----
        st.write("---")
        st.subheader(ui[lang]["heatmap_title"])

        if not generate_heatmap:
            st.info(ui[lang]["clear_lungs"])
        else:
            st.write(ui[lang]["heatmap_desc"])
            target_display = disease_translation[lang][classes[target_idx]]
            st.write(f"**🎯 Target Focus / منطقة التركيز:** {target_display} ({target_prob:.1f}%)")

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption=ui[lang]["original_img"], use_container_width=True)
            with col2:
                st.image(visualization, caption=ui[lang]["ai_img"], use_container_width=True)

        # Optional: clear GPU cache to prevent memory build-up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# -------------------------------------------------------------------
# 7. Footer
st.write("---")
st.markdown(f"<div style='text-align: center; color: gray; font-size: 0.8em;'>{ui[lang]['footer']}</div>", unsafe_allow_html=True)