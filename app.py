"""BlurMe - a small Streamlit app for blurring images.

Run it with::

    streamlit run app.py

Face mode needs ``face_detection_yunet_2023mar.onnx`` next to this file.
"""

from __future__ import annotations

import streamlit as st

from blur_utils import METHODS, encode_image, process, process_faces

st.set_page_config(page_title="BlurMe", page_icon="🫥", layout="wide")

st.title("🫥 BlurMe")
st.caption("Upload an image, pick a blur, download the result - it never leaves your machine.")

# Defaults for the region controls (they are hidden in face mode).
mode = "Whole image"
center_x, center_y, radius = 0.5, 0.5, 0.35

with st.sidebar:
    st.header("Blur settings")
    method = st.radio("Method", list(METHODS), index=0, help="How the pixels are mixed.")
    strength = st.slider("Strength", 0, 100, 50, help="0 = untouched, 100 = maximum blur.")

    st.divider()
    face_blur = st.checkbox(
        "Blur faces only (YuNet ONNX detector)",
        help="Requires face_detection_yunet_2023mar.onnx next to this file.",
    )

    if face_blur:
        confidence = (
            st.slider("Detection confidence", 10, 95, 50, help="Higher = fewer, safer detections.")
            / 100
        )
        padding = st.slider(
            "Face padding", 0, 50, 20, help="Extra area around each face, in percent of its size."
        ) / 100
    else:
        confidence, padding = 0.5, 0.2
        st.header("Region")
        mode = st.radio(
            "Region",
            ["Whole image", "Blur circle only", "Blur everything but circle"],
            help="Optionally confine the blur to a circle.",
        )
        col_x, col_y = st.columns(2)
        with col_x:
            center_x = st.slider("Circle centre X", 0, 100, 50) / 100
        with col_y:
            center_y = st.slider("Circle centre Y", 0, 100, 50) / 100
        radius = st.slider("Circle radius", 5, 100, 35, help="Percent of the shorter side.") / 100

region_mode = {
    "Whole image": "whole",
    "Blur circle only": "circle",
    "Blur everything but circle": "inverse",
}[mode]

uploads = st.file_uploader(
    "Drop one or more images here",
    type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"],
    accept_multiple_files=True,
)

if not uploads:
    st.info("Upload an image to get started.")
    st.stop()

for upload in uploads:
    st.subheader(upload.name)
    left, right = st.columns(2)

    faces = None
    try:
        if face_blur:
            result, faces = process_faces(
                upload.getvalue(),
                method=method,
                strength=strength,
                confidence=confidence,
                padding=padding,
            )
        else:
            result = process(
                upload.getvalue(),
                method=method,
                strength=strength,
                mode=region_mode,
                center=(center_x, center_y),
                radius=radius,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(f"Could not process {upload.name}: {exc}")
        continue

    if faces == 0:
        st.warning(f"No faces detected in {upload.name} - try lowering the confidence threshold.")

    caption = f"{method} · {strength}"
    if faces is not None:
        caption += f" · {faces} face(s)"

    with left:
        st.image(upload.getvalue(), caption="Original", use_container_width=True)
    with right:
        st.image(result, caption=caption, use_container_width=True)

    st.download_button(
        f"Download {upload.name}",
        data=encode_image(result),
        file_name=f"blurred_{upload.name.rsplit('.', 1)[0]}.png",
        mime="image/png",
        key=f"dl_{upload.name}",
    )
