import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.title("Clothing Detection")

model = YOLO("yolov8n.pt")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    image = Image.open(file)
    img = np.array(image)

    results = model(img)

    annotated = results[0].plot()

    st.image(annotated)
