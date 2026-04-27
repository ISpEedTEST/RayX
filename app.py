# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# import torchvision.transforms.functional as TF
# from PIL import Image
# import numpy as np
# import time

# # Grad-CAM Imports
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

# # 1. Page Config
# st.set_page_config(page_title="RayX AI", page_icon="🩺", layout="centered")

# # 2. UI Languages Dictionary
# lang = st.sidebar.selectbox("Select Language / اختر اللغة", ["English", "العربية"])

# ui = {
#     "English": {
#         "title": "🩺 RayX AI - Chest X-Ray Analyzer",
#         "desc": "Upload a Chest X-ray image for instant AI-powered analysis of 15 thoracic diseases.",
#         "upload_label": "📂 Drag and drop image here or click to browse",
#         "btn_scan": "🔍 Start Deep Chest Analysis",
#         "status_start": "Initiating TTA Deep Scan...",
#         "status_1": "1. Analyzing original view...",
#         "status_2": "2. Analyzing structural variations (Angles)...",
#         "status_3": "3. Analyzing contrast variations (Tissue Density)...",
#         "status_4": "4. Fusing AI predictions & Generating Heatmap...",
#         "status_done": "✅ Scan Complete!",
#         "success": "✅ Analysis Complete! Medical Report:",
#         "risk_high": "HIGH RISK",
#         "risk_med": "Moderate",
#         "risk_low": "Unlikely",
#         "heatmap_title": "🧠 Explainable AI (Grad-CAM Focus Map)",
#         "heatmap_desc": "The AI focused on the red/yellow areas to make its diagnosis.",
#         "clear_lungs": "✅ Lungs appear mostly clear. No significant anomaly to map.",
#         "original_img": "Original X-Ray",
#         "ai_img": "AI Heatmap",
#         "footer": "Trained by **Gharib Cloud**. This is an AI tool designed for chest X-rays and should not be fully relied upon. Please consult a hospital for professional medical advice."
#     },
#     "العربية": {
#         "title": "🩺 RayX AI (التحليل الصدري العميق)",
#         "desc": "قم برفع صورة أشعة سينية للصدر (Chest X-ray) ليقوم الذكاء الاصطناعي بتحليل 15 مرضاً صدرياً فوراً.",
#         "upload_label": "📂 اسحب وأفلت الصورة هنا أو اضغط للاختيار",
#         "btn_scan": "🔍 ابدأ الفحص العميق",
#         "status_start": "بدء الفحص العميق المتعدد (TTA)...",
#         "status_1": "1. تحليل الزاوية الأصلية...",
#         "status_2": "2. تحليل التغيرات الهيكلية للقفص الصدري...",
#         "status_3": "3. الفحص الإشعاعي لتباين الأنسجة السائلة...",
#         "status_4": "4. دمج القراءات وتوليد الخريطة الحرارية...",
#         "status_done": "✅ اكتمل التحليل بنجاح!",
#         "success": "✅ اكتمل الفحص! إليك التقرير الطبي:",
#         "risk_high": "خطر مرتفع",
#         "risk_med": "احتمال متوسط",
#         "risk_low": "سليم / مستبعد",
#         "heatmap_title": "🧠 التحليل البصري العميق (Grad-CAM)",
#         "heatmap_desc": "المناطق الملونة بالأحمر/الأصفر هي ما اعتمد عليه الذكاء الاصطناعي لاكتشاف المرض (مثل السوائل والالتهابات).",
#         "clear_lungs": "✅ الرئة تبدو سليمة تماماً. لا توجد مناطق مرضية لتحديدها.",
#         "original_img": "الأشعة الأصلية",
#         "ai_img": "خريطة التحليل الحرارية",
#         "footer": "تم تدريبه بواسطة **Gharib Cloud**. هذا نظام ذكاء اصطناعي مخصص لأشعة الصدر ولا يعتمد عليه بشكل تام، يرجى مراجعة المستشفيات للحصول على استشارة طبية احترافية."
#     }
# }

# st.title(ui[lang]["title"])
# st.write(ui[lang]["desc"])

# # 3. Classes & Translations
# classes = [
#     'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
#     'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
#     'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
# ]

# disease_translation = {
#     "English": {cls: cls for cls in classes},
#     "العربية": {
#         'Atelectasis': 'انخماص الرئة', 'Cardiomegaly': 'تضخم القلب', 'Consolidation': 'تصلب الرئة',
#         'Edema': 'وذمة رئوية', 'Effusion': 'انصباب سوائل', 'Emphysema': 'نفاخ رئوي',
#         'Fibrosis': 'تليف رئوي', 'Hernia': 'فتق', 'Infiltration': 'ارتشاح', 'Mass': 'كتلة',
#         'No Finding': 'سليم / لا توجد أمراض', 'Nodule': 'عُقدة', 'Pleural_Thickening': 'سماكة غشاء الرئة',
#         'Pneumonia': 'التهاب رئوي', 'Pneumothorax': 'استرواح الصدر'
#     }
# }

# # 4. Load Model
# @st.cache_resource
# def load_model():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = models.resnet18(weights=None)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, len(classes))
#     model.load_state_dict(torch.load('nih_multilabel_model_v2.pth', map_location=device, weights_only=True))
#     model = model.to(device)
#     model.eval()
#     return model, device

# model, device = load_model()

# # 5. Image Transformer
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # 6. Main App Logic
# uploaded_file = st.file_uploader(ui[lang]["upload_label"], type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert('RGB')
#     st.image(image, use_container_width=True)
    
#     if st.button(ui[lang]["btn_scan"]):
        
#         with st.status("🔍 " + ui[lang]["status_start"], expanded=True) as status:
            
#             # TTA Process
#             st.write("⚙️ " + ui[lang]["status_1"])
#             t0 = transform(image)
            
#             st.write("📐 " + ui[lang]["status_2"])
#             t1 = transform(TF.rotate(image, 3))
#             t2 = transform(TF.rotate(image, -3))
            
#             st.write("💡 " + ui[lang]["status_3"])
#             t3 = transform(TF.adjust_brightness(image, 1.1))
#             t4 = transform(TF.adjust_brightness(image, 0.9))
            
#             st.write("🧠 " + ui[lang]["status_4"])
#             input_batch = torch.stack([t0, t1, t2, t3, t4]).to(device)
            
#             with torch.no_grad():
#                 outputs = model(input_batch)
#                 probs = torch.sigmoid(outputs)
#                 final_probs = probs.mean(dim=0) * 100 
                
#             results = [(classes[i], final_probs[i].item()) for i in range(len(classes))]
#             results.sort(key=lambda x: x[1], reverse=True)
            
#             # Select target for Heatmap (Ignore No Finding)
#             no_finding_idx = classes.index('No Finding')
#             anomaly_probs = final_probs.clone()
#             anomaly_probs[no_finding_idx] = -1.0 
            
#             target_idx = torch.argmax(anomaly_probs).item()
#             target_prob = final_probs[target_idx].item()
            
#             # Generate Heatmap if disease is found
#             generate_heatmap = True
#             if target_prob < 15.0 and final_probs[no_finding_idx].item() > 50.0:
#                 generate_heatmap = False
#             else:
#                 target_layers = [model.layer4[-1]]
#                 targets = [ClassifierOutputTarget(target_idx)]
#                 single_tensor = t0.unsqueeze(0).to(device) # Use original image for map
                
#                 with GradCAM(model=model, target_layers=target_layers) as cam:
#                     grayscale_cam = cam(input_tensor=single_tensor, targets=targets)
#                     grayscale_cam = grayscale_cam[0, :]
                
#                 rgb_img = np.array(image.resize((224, 224))) / 255.0
#                 visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

#             status.update(label=ui[lang]["status_done"], state="complete", expanded=False)
            
#         # Display Results
#         st.success(ui[lang]["success"])
#         for disease, prob in results:
#             display_name = disease_translation[lang][disease]
#             status_text = ui[lang]["risk_low"]
#             if prob > 50:
#                 status_text = ui[lang]["risk_high"]
#                 st.error(f"🚨 {display_name}: {prob:.2f}% ({status_text})")
#             elif prob > 20:
#                 status_text = ui[lang]["risk_med"]
#                 st.warning(f"⚠️ {display_name}: {prob:.2f}% ({status_text})")
#             else:
#                 st.success(f"✅ {display_name}: {prob:.2f}% ({status_text})")
                
#         # Display Heatmap Section
#         st.write("---")
#         st.subheader(ui[lang]["heatmap_title"])
        
#         if not generate_heatmap:
#             st.info(ui[lang]["clear_lungs"])
#         else:
#             st.write(ui[lang]["heatmap_desc"])
#             display_target = disease_translation[lang][classes[target_idx]]
#             st.write(f"**🎯 Target Focus / منطقة التركيز:** {display_target} ({target_prob:.1f}%)")
            
#             # Split screen into 2 columns for comparison
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image, caption=ui[lang]["original_img"], use_container_width=True)
#             with col2:
#                 st.image(visualization, caption=ui[lang]["ai_img"], use_container_width=True)

# # 7. Footer
# st.write("---")
# st.markdown(f"<div style='text-align: center; color: gray; font-size: 0.8em;'>{ui[lang]['footer']}</div>", unsafe_allow_html=True)
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

# 1. Page Config
st.set_page_config(page_title="RayX AI", page_icon="🩺", layout="centered")

# 2. UI Languages Dictionary
lang = st.sidebar.selectbox("Select Language / اختر اللغة", ["English", "العربية"])

ui = {
    "English": {
        "title": "🩺 RayX AI - Chest X-Ray Analyzer",
        "desc": "Upload a Chest X-ray image for instant AI-powered analysis of 15 thoracic diseases.",
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
        "footer": "Trained by **Gharib Cloud**. This is an AI tool designed for chest X-rays and should not be fully relied upon. Please consult a hospital for professional medical advice."
    },
    "العربية": {
        "title": "🩺 RayX AI (التحليل الصدري العميق)",
        "desc": "قم برفع صورة أشعة سينية للصدر (Chest X-ray) ليقوم الذكاء الاصطناعي بتحليل 15 مرضاً صدرياً فوراً.",
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
        "footer": "تم تدريبه بواسطة **Gharib Cloud**. هذا نظام ذكاء اصطناعي مخصص لأشعة الصدر ولا يعتمد عليه بشكل تام، يرجى مراجعة المستشفيات للحصول على استشارة طبية احترافية."
    }
}

st.title(ui[lang]["title"])
st.write(ui[lang]["desc"])

# 3. Classes & Translations
classes = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

disease_translation = {
    "English": {cls: cls for cls in classes},
    "العربية": {
        'Atelectasis': 'انخماص الرئة', 'Cardiomegaly': 'تضخم القلب', 'Consolidation': 'تصلب الرئة',
        'Edema': 'وذمة رئوية', 'Effusion': 'انصباب سوائل', 'Emphysema': 'نفاخ رئوي',
        'Fibrosis': 'تليف رئوي', 'Hernia': 'فتق', 'Infiltration': 'ارتشاح', 'Mass': 'كتلة',
        'No Finding': 'سليم / لا توجد أمراض', 'Nodule': 'عُقدة', 'Pleural_Thickening': 'سماكة غشاء الرئة',
        'Pneumonia': 'التهاب رئوي', 'Pneumothorax': 'استرواح الصدر'
    }
}

# 4. Load Model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    # Ensure map_location is set to device to handle both CPU/GPU environments
    model.load_state_dict(torch.load('nih_multilabel_model_v2.pth', map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# 5. Image Transformers
transform_base = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_zoomed_image(img):
    width, height = img.size
    new_width = int(width * 0.85)
    new_height = int(height * 0.85)
    return TF.center_crop(img, [new_height, new_width])

# 6. Main App Logic
uploaded_file = st.file_uploader(ui[lang]["upload_label"], type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    if st.button(ui[lang]["btn_scan"]):
        
        with st.status("🔍 " + ui[lang]["status_start"], expanded=True) as status:
            
            # --- PHASE 1: Global Scan (TTA) ---
            st.write("🌍 " + ui[lang]["status_1"])
            t0 = transform_base(image)
            t1 = transform_base(TF.rotate(image, 3))
            t2 = transform_base(TF.rotate(image, -3))
            t3 = transform_base(TF.adjust_brightness(image, 1.1))
            t4 = transform_base(TF.adjust_brightness(image, 0.9))
            
            batch_phase1 = torch.stack([t0, t1, t2, t3, t4]).to(device)
            
            with torch.no_grad():
                out_phase1 = model(batch_phase1)
                probs_phase1 = torch.sigmoid(out_phase1).mean(dim=0) * 100
            
            # --- PHASE 2: Targeted Verification (Zoomed) ---
            st.write("🔬 " + ui[lang]["status_2"])
            zoomed_img = get_zoomed_image(image)
            z0 = transform_base(zoomed_img)
            z1 = transform_base(TF.rotate(zoomed_img, 3))
            z2 = transform_base(TF.rotate(zoomed_img, -3))
            
            batch_phase2 = torch.stack([z0, z1, z2]).to(device)
            
            with torch.no_grad():
                out_phase2 = model(batch_phase2)
                probs_phase2 = torch.sigmoid(out_phase2).mean(dim=0) * 100
                
            # --- PHASE 3: Cross-Referencing ---
            st.write("⚖️ " + ui[lang]["status_3"])
            final_probs = (probs_phase1 + probs_phase2) / 2.0
            
            results = [(classes[i], final_probs[i].item()) for i in range(len(classes))]
            results.sort(key=lambda x: x[1], reverse=True)
            
            # --- PHASE 4: Grad-CAM Heatmap ---
            st.write("🧠 " + ui[lang]["status_4"])
            no_finding_idx = classes.index('No Finding')
            anomaly_probs = final_probs.clone()
            anomaly_probs[no_finding_idx] = -1.0 
            
            target_idx = torch.argmax(anomaly_probs).item()
            target_prob = final_probs[target_idx].item()
            
            generate_heatmap = True
            if target_prob < 15.0 and final_probs[no_finding_idx].item() > 50.0:
                generate_heatmap = False
            else:
                target_layers = [model.layer4[-1]]
                targets = [ClassifierOutputTarget(target_idx)]
                single_tensor = t0.unsqueeze(0).to(device)
                
                with GradCAM(model=model, target_layers=target_layers) as cam:
                    grayscale_cam = cam(input_tensor=single_tensor, targets=targets)
                    grayscale_cam = grayscale_cam[0, :]
                
                rgb_img = np.array(image.resize((224, 224))) / 255.0
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            status.update(label=ui[lang]["status_done"], state="complete", expanded=False)
            
        # Display Results
        st.success(ui[lang]["success"])
        for disease, prob in results:
            display_name = disease_translation[lang][disease]
            status_text = ui[lang]["risk_low"]
            if prob > 50:
                status_text = ui[lang]["risk_high"]
                st.error(f"🚨 {display_name}: {prob:.2f}% ({status_text})")
            elif prob > 20:
                status_text = ui[lang]["risk_med"]
                st.warning(f"⚠️ {display_name}: {prob:.2f}% ({status_text})")
            else:
                st.success(f"✅ {display_name}: {prob:.2f}% ({status_text})")
                
        # Display Heatmap Section
        st.write("---")
        st.subheader(ui[lang]["heatmap_title"])
        
        if not generate_heatmap:
            st.info(ui[lang]["clear_lungs"])
        else:
            st.write(ui[lang]["heatmap_desc"])
            display_target = disease_translation[lang][classes[target_idx]]
            st.write(f"**🎯 Target Focus / منطقة التركيز:** {display_target} ({target_prob:.1f}%)")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption=ui[lang]["original_img"], use_container_width=True)
            with col2:
                st.image(visualization, caption=ui[lang]["ai_img"], use_container_width=True)
                
        # Memory Cleanup to prevent crashes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 7. Footer
st.write("---")
st.markdown(f"<div style='text-align: center; color: gray; font-size: 0.8em;'>{ui[lang]['footer']}</div>", unsafe_allow_html=True)