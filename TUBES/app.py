import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load model
model = YOLO("best.pt")

st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("🔍 YOLO Object Detection - Rempah")

menu = st.sidebar.selectbox(
    "Pilih Mode",
    ("Upload Gambar", "Webcam")
)

# =========================
# UPLOAD GAMBAR
# =========================
if menu == "Upload Gambar":
    uploaded_file = st.file_uploader(
        "Upload gambar", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar Asli", use_container_width=True)

        if st.button("Deteksi Objek"):
            results = model(image)
            result_img = results[0].plot()
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

# =========================
# WEBCAM
# =========================
elif menu == "Webcam":
    st.warning("Tekan STOP di terminal untuk menghentikan webcam")

    if st.button("Mulai Webcam"):
        cap = cv2.VideoCapture(1)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR")

        cap.release()