import os
import cv2
import time
import tempfile
from pathlib import Path
import streamlit as st
from yolo_predictions import YOLO_Pred

# Page config
st.set_page_config(page_title="YOLO Video Detection", layout="wide", page_icon="./images/object.png")
st.header("🎥Object Detection for Video")
st.write("Upload a video and run detections.")

with st.sidebar:
    st.subheader("Model Settings")
    resize_to_640x480 = st.checkbox("Resize frames to 640x480 before inference", value=True)
    save_output = st.checkbox("Save annotated video", value=True)

# Load model
with st.spinner("Loading YOLO model..."):
    yolo = YOLO_Pred(onnx_model='./models/best.onnx', data_yaml='./models/data.yaml')

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
temp_in_path = None
cap = None

if uploaded_file:
    # Write uploaded bytes to a temp file for OpenCV
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
        tmp.write(uploaded_file.read())
        temp_in_path = tmp.name
    st.caption(f"Loaded: {uploaded_file.name}")
    st.video(temp_in_path)

    if st.button("Run Detection"):
        cap = cv2.VideoCapture(temp_in_path)

if cap is not None and cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Probe first frame to decide output size
    ret, probe = cap.read()
    if not ret or probe is None:
        st.error("Could not read frames from the uploaded video.")
        cap.release()
    else:
        if resize_to_640x480:
            out_w, out_h = 640, 480
        else:
            out_h, out_w = probe.shape[:2]

        out_path = None
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fd, out_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

        # UI placeholders
        progress = st.progress(0) if total > 0 else None
        status = st.empty()
        frame_placeholder = st.empty()
        t0 = time.time()
        processed = 0

        # Rewind to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                # Optional resize to match training (BGR)
                if resize_to_640x480:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

                # Run model prediction (expects BGR)
                annotated = yolo.predictions(frame)

                if writer is not None:
                    writer.write(annotated)

                # Show current annotated frame (convert BGR->RGB)
                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Frame {processed + 1}"
                )

                processed += 1
                if total > 0 and progress is not None:
                    progress.progress(min(processed / total, 1.0))
                status.text(
                    f"Processed: {processed} / {total if total>0 else '?'} "
                    f"| FPS: {processed / max(time.time() - t0, 1e-6):.1f}"
                )
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        st.success(f"Done. Processed {processed} frames.")
        if save_output and out_path and os.path.exists(out_path):
            st.subheader("Annotated Video")
            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button("Download annotated video", data=f, file_name="annotated.mp4", mime="video/mp4")