import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from fpdf import FPDF
import os
import uuid

# Load trained model
model = tf.keras.models.load_model("emotion_classifier.h5")

# Class names
class_names = ['Angry', 'Fear', 'Happy', 'Sad']

# Explanation for each emotion
emotion_explanations = {
    "Angry": {
        "description": "This may indicate hidden frustration or difficulty accepting reality.",
        "suggestion": "Provide a safe space for expression and encourage conversations."
    },
    "Fear": {
        "description": "Might reflect inner fears or lack of safety.",
        "suggestion": "Talk to the child about fears and provide emotional support."
    },
    "Happy": {
        "description": "Represents comfort, joy, and emotional stability.",
        "suggestion": "Continue fostering a positive environment."
    },
    "Sad": {
        "description": "Could suggest loneliness or emotional stress due to loss or pressure.",
        "suggestion": "Spend time and help the child express feelings safely."
    }
}

# Page setup
st.set_page_config(page_title="PulseAI", page_icon="🎨", layout="centered")
st.title("🎨 PulseAI – Emotional Analysis from Children's Drawings")

st.write(
    "Upload a child's drawing and let our AI analyze potential emotional indicators. "
    "You will also receive a downloadable psychological insights report."
)

# Upload
uploaded_file = st.file_uploader("📤 Upload a drawing image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Drawing", use_column_width=True)

    # Preprocess
    img = image.resize((128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)[0]

    # Show results
    st.subheader("🔍 Emotion Prediction:")
    for i in range(len(class_names)):
        st.write(f"**{class_names[i]}**: {prediction[i]:.2%}")
        st.progress(int(prediction[i] * 100))

    top_emotion = class_names[np.argmax(prediction)]
    st.success(f"🧠 Most likely emotion: **{top_emotion}**")

    # Explanation
    explanation = emotion_explanations[top_emotion]
    st.markdown("### 🧠 Psychological Insight:")
    st.markdown(f"**📌 Explanation:** {explanation['description']}")
    st.markdown(f"**💡 Suggested Support:** {explanation['suggestion']}")

    # 🔺 Smart alert if negative emotion > 60%
    negative_emotions = ['Fear', 'Sad']
    for idx, emotion in enumerate(class_names):
        if emotion in negative_emotions and prediction[idx] > 0.6:
            st.warning("⚠️ High levels of negative emotion detected. Consider consulting a specialist.")

    # 📝 Generate PDF report
    if st.button("📄 Download PDF Report"):
        report_id = str(uuid.uuid4())[:8]
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="PulseAI - Emotional Analysis Report", ln=True, align="C")
        pdf.cell(200, 10, txt=f"Report ID: {report_id}", ln=True, align="C")
        pdf.ln(10)

        pdf.cell(200, 10, txt="Emotion Probabilities:", ln=True)
        for i in range(len(class_names)):
            pdf.cell(200, 10, txt=f"{class_names[i]}: {prediction[i]*100:.2f}%", ln=True)

        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Most Likely Emotion: {top_emotion}", ln=True)
        pdf.cell(200, 10, txt=f"Explanation: {explanation['description']}", ln=True)
        pdf.multi_cell(0, 10, txt=f"Suggested Support: {explanation['suggestion']}")

        report_path = f"report_{report_id}.pdf"
        pdf.output(report_path)

        with open(report_path, "rb") as file:
            btn = st.download_button(
                label="📥 Click to download your report",
                data=file,
                file_name=report_path,
                mime="application/pdf"
            )

        # Clean up
        os.remove(report_path)
