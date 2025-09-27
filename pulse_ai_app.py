import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from fpdf import FPDF
import os
import gdown

# تحميل النموذج من Google Drive إذا ما كان موجود
drive_url = "https://drive.google.com/uc?id=1qPiZUGLxxCZqx1YmEId0FQQVbz4pZb7j"
model_filename = "emotion_classifier.h5"

if not os.path.exists(model_filename):
    gdown.download(drive_url, model_filename, quiet=False)

# تحميل النموذج
model = tf.keras.models.load_model(model_filename)

# تصنيف المشاعر
class_names = ['Angry', 'Fear', 'Happy', 'Sad']

# تصميم واجهة Streamlit
st.title("🎨 PulseAI - Emotion Detection from Children's Drawings")
st.write("Upload a child's drawing and let AI detect the emotional state behind it.")

uploaded_file = st.file_uploader("Choose a drawing image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Drawing', use_column_width=True)

    # معالجة الصورة
    image = image.resize((64, 64))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # التنبؤ
    predictions = model.predict(image_array)[0]

    # عرض النتائج
    st.subheader("Predicted Emotion:")
    for i, prob in enumerate(predictions):
        st.write(f"{class_names[i]}: {prob*100:.2f}%")

    # أعلى نتيجة
    predicted_label = class_names[np.argmax(predictions)]
    st.success(f"Detected Emotion: **{predicted_label}**")

    # تصدير PDF
    if st.button("Export PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="PulseAI - Emotion Detection Report", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Predicted Emotion: {predicted_label}", ln=True)
        pdf.ln(10)

        for i, prob in enumerate(predictions):
            pdf.cell(200, 10, txt=f"{class_names[i]}: {prob*100:.2f}%", ln=True)

        pdf.output("pulseai_report.pdf")
        st.success("PDF report generated successfully!")
