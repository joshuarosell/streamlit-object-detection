import cv2
import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np

st.set_page_config(page_title="YOLO Object Detection",
                   layout='wide',
                   page_icon='./images/object.png')

st.header('🖼️Object Detection for Image')
st.write('Upload an image and run detections.')

with st.spinner('Loading YOLO model...'):
    yolo = YOLO_Pred(onnx_model='./models/best.onnx', data_yaml='./models/data.yaml')

def upload_image():
    file = st.file_uploader('Upload Image', type=['jpg','jpeg','png'])
    if not file:
        return None
    size_mb = file.size / (1024**2)
    details = {
        "filename": file.name,
        "filetype": file.type,
        "filesize": f"{size_mb:,.2f} MB"
    }
    if file.type not in ('image/png','image/jpeg'):
        st.error('INVALID image type (use png or jpeg)')
        return None
    # Decode bytes to BGR (matches cv2.imread)
    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Failed to decode image.")
        return None
    # Resize EXACTLY to training size 640x480 (width=640, height=480)
    img_bgr = cv2.resize(img_bgr, (640, 480), interpolation=cv2.INTER_LINEAR)
    return {"bgr": img_bgr, "details": details}

def main():
    uploaded_img = upload_image()
    if not uploaded_img:
        return

    col1, col2 = st.columns(2)
    with col1:
        st.info("Resized 640x480 Preview (BGR->RGB)")
        st.image(cv2.cvtColor(uploaded_img["bgr"], cv2.COLOR_BGR2RGB),
                 caption=uploaded_img["details"]["filename"])

    with col2:
        st.subheader("File Details")
        st.json(uploaded_img["details"])
        if st.button("Run Detection"):
            with st.spinner("Running YOLO..."):
                pred_bgr = yolo.predictions(uploaded_img["bgr"])  # keep BGR inside
                st.subheader("Predicted Image")
                st.caption("YOLO detections (640x480)")
                st.image(cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    main()