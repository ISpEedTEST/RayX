import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. إعدادات الصفحة
st.set_page_config(page_title="AI Medical Scanner", page_icon="🩺", layout="centered")

# 2. نظام اللغات (UI Strings)
# تم وضع خيار اللغة في القائمة الجانبية ليكون التطبيق احترافياً
lang = st.sidebar.selectbox("Select Language / اختر اللغة", ["English", "العربية"])


ui = {
    "English": {
        "title": "🩺 RayX AI - Chest X-Ray Analyzer",
        "desc": "Upload a Chest X-ray image for instant AI-powered analysis of 15 thoracic diseases.",
        "upload_label": "📂 Drag and drop image here or click to browse",
        "btn_scan": "🔍 Start Chest Analysis",
        "loading": "Analyzing Chest X-ray... Please wait",
        "success": "✅ Analysis Complete! Medical Report:",
        "risk_high": "HIGH RISK",
        "risk_med": "Moderate",
        "risk_low": "Unlikely",
        "footer": "Trained by **Gharib Cloud**. This is an AI tool designed for chest X-rays and should not be fully relied upon. Please consult a hospital for professional medical advice."
    },
    "العربية": {
        "title": "🩺 RayX AI (للتحليل الصدري)",
        "desc": "قم برفع صورة أشعة سينية للصدر (Chest X-ray) ليقوم الذكاء الاصطناعي بتحليل 15 مرضاً صدرياً فوراً.",
        "upload_label": "📂 اسحب وأفلت الصورة هنا أو اضغط للاختيار",
        "btn_scan": "🔍 ابدأ فحص الصدر",
        "loading": "جاري تحليل الصدر... يرجى الانتظار",
        "success": "✅ اكتمل الفحص! إليك التقرير الطبي:",
        "risk_high": "خطر مرتفع",
        "risk_med": "احتمال متوسط",
        "risk_low": "سليم / مستبعد",
        "footer": "تم تدريبه بواسطة **Gharib Cloud**. هذا نظام ذكاء اصطناعي مخصص لأشعة الصدر ولا يعتمد عليه بشكل تام، يرجى مراجعة المستشفيات للحصول على استشارة طبية احترافية."
    }
}

st.title(ui[lang]["title"])
st.write(ui[lang]["desc"])

# 3. القائمة التقنية (للنموذج)
classes = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# قاموس ترجمة الأمراض للعرض فقط
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

# 4. تحميل النموذج
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(torch.load('nih_multilabel_model_v2.pth', map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# 5. معالج الصور
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 6. واجهة الرفع والنتائج
uploaded_file = st.file_uploader(ui[lang]["upload_label"], type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    if st.button(ui[lang]["btn_scan"]):
        with st.spinner(ui[lang]["loading"]):
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.sigmoid(outputs)[0] * 100 
                
            results = [(classes[i], probs[i].item()) for i in range(len(classes))]
            results.sort(key=lambda x: x[1], reverse=True)
            
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

# 7. الفوتر (Footer) - يظهر دائماً في الأسفل
st.write("---")
st.markdown(f"<div style='text-align: center; color: gray; font-size: 0.8em;'>{ui[lang]['footer']}</div>", unsafe_allow_html=True)