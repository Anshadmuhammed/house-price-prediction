import streamlit as st
import numpy as np
import os
from PIL import Image

st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="centered")

CLASS_NAMES = [
    "Airplane","Automobile","Bird","Cat","Deer",
    "Dog","Frog","Horse","Ship","Truck"
]

CLASS_EMOJI = {
    "Airplane":   "✈️", "Automobile": "🚗", "Bird":   "🐦", "Cat":   "🐱",
    "Deer":       "🦌", "Dog":        "🐶", "Frog":   "🐸", "Horse": "🐴",
    "Ship":       "🚢", "Truck":      "🚚",
}

@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        if not os.path.exists("cnn_model.h5"):
            st.error("cnn_model.h5 not found. Please run  `python train.py`  first.")
            st.stop()
        return tf.keras.models.load_model("cnn_model.h5")
    except ImportError:
        st.error("TensorFlow is not installed. Run: pip install tensorflow")
        st.stop()

model = load_model()

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((32, 32))
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🖼️ Image Classification — CNN")
st.markdown("Upload an image and the CNN will classify it into one of **10 CIFAR-10 categories**.")
st.markdown("**Categories:** Airplane · Automobile · Bird · Cat · Deer · Dog · Frog · Horse · Ship · Truck")
st.markdown("---")

uploaded = st.file_uploader("Upload an image (JPG / PNG)", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("Classifying..."):
            x     = preprocess_image(img)
            preds = model.predict(x)[0]
            top3  = np.argsort(preds)[::-1][:3]

        pred_label = CLASS_NAMES[top3[0]]
        pred_conf  = preds[top3[0]] * 100

        st.markdown(f"### {CLASS_EMOJI[pred_label]}  {pred_label}")
        st.progress(int(pred_conf))
        st.markdown(f"**Confidence: {pred_conf:.1f}%**")

        st.markdown("#### Top 3 Predictions")
        for idx in top3:
            name = CLASS_NAMES[idx]
            conf = preds[idx] * 100
            st.write(f"{CLASS_EMOJI[name]}  **{name}** — {conf:.1f}%")
            st.progress(int(conf))

st.caption("Built with Python · TensorFlow/Keras · Streamlit · CIFAR-10")
