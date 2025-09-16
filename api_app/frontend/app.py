import os
from io import BytesIO
from typing import Optional

import streamlit as st
import requests
from PIL import Image
import numpy as np

DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
DISCLAIMER = "For clinical decision support only — not a diagnostic replacement. Clinical decisions must be made by qualified physicians."


def call_backend(backend_url: str, image_bytes: bytes):
    url = backend_url.rstrip("/") + "/predict"
    files = {"file": ("upload.png", image_bytes, "image/png")}
    try:
        resp = requests.post(url, files=files, timeout=30)
    except requests.RequestException as e:
        st.error(f"Backend request failed: {e}")
        return None
    if resp.status_code != 200:
        st.error(f"Backend error {resp.status_code}: {resp.text}")
        return None
    return resp.json()


def main():
    st.set_page_config(page_title="NeuroClassify API Frontend", layout="wide")
    st.title("NeuroClassify (API-based)")
    st.caption(DISCLAIMER)

    with st.sidebar:
        st.header("Backend Settings")
        backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND)
        st.write("Health Check:")
        if st.button("Ping /health"):
            try:
                h = requests.get(backend_url.rstrip("/") + "/health", timeout=10).json()
                st.success(h)
            except Exception as e:
                st.error(f"Health check failed: {e}")

    st.header("Upload MRI Image (PNG/JPG)")
    file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if file is None:
        st.info("Upload an image to get predictions from the backend API.")
        return

    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        buf = BytesIO()
        image.save(buf, format="PNG")
        data = call_backend(backend_url, buf.getvalue())
        if data is None:
            return
        classes = data.get("classes", [])
        probs = data.get("probabilities", [])
        pred = data.get("pred_class", "")

        st.subheader("Predicted Probabilities")
        if classes and probs and len(classes) == len(probs):
            st.table({"Class": classes, "Probability": [f"{p:.4f}" for p in probs]})
        else:
            st.write(data)
        st.success(f"Predicted: {pred}")

    st.markdown("---")
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
