import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io

st.set_page_config(layout="wide", page_title="Tumor Segmentation", page_icon="🧠")
st.title("🧠 Medical Image Tumor Segmentation and Analysis")

# Sidebar for user inputs
st.sidebar.header("Input Settings")
uploaded_file = st.sidebar.file_uploader("Upload a medical image", type=["jpg", "png", "jpeg"])
threshold = st.sidebar.slider("Segmentation Threshold", 0.0, 1.0, 0.5, 0.01)

if uploaded_file is not None:
    # Read image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    image_norm = image.astype(np.float32) / 255.0

    # Preprocessing
    smoothed = cv2.GaussianBlur(image_norm, (5, 5), 0)
    segmented = (smoothed > threshold).astype(np.uint8) * 255

    # Finding contours (tumor regions)
    contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare images
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    tumor_cutout = np.zeros_like(overlay)

    total_area = image.shape[0] * image.shape[1]
    tumor_area = 0
    tumor_count = 0

    # Loop through contours and find tumors
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if area > 100 and circularity < 0.9:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(overlay, ellipse, (0, 255, 0), 2)
            cv2.drawContours(tumor_cutout, [cnt], -1, (255, 255, 255), thickness=cv2.FILLED)
            tumor_area += area
            tumor_count += 1

    # Calculate percentages
    tumor_percentage = (tumor_area / total_area) * 100
    non_tumor_percentage = 100 - tumor_percentage

    # Layout: Display images and results in columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔍 Tumor Highlighted")
        st.image(overlay, channels="BGR")

        # Convert image to PNG for downloading
        _, overlay_png = cv2.imencode(".png", overlay)
        st.download_button("📥 Download Highlighted Image", overlay_png.tobytes(), file_name="tumor_highlighted.png", mime="image/png")

    with col2:
        st.subheader("🩺 Tumor Cutout")
        st.image(tumor_cutout, channels="BGR")

        # Convert image to PNG for downloading
        _, cutout_png = cv2.imencode(".png", tumor_cutout)
        st.download_button("📥 Download Tumor Cutout", cutout_png.tobytes(), file_name="tumor_cutout.png", mime="image/png")

    # Tumor analysis summary in a separate expander
    with st.expander("📈 Tumor Analysis Summary"):
        st.metric("Total Detected Tumor Regions", tumor_count)
        st.metric("Tumor Area (%)", f"{tumor_percentage:.2f}%")

        # Pie chart for tumor vs healthy tissue
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            [tumor_percentage, non_tumor_percentage],
            labels=["Tumor", "Healthy Tissue"],
            colors=["red", "lightgreen"],
            autopct="%.2f%%",
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)

        # Downloadable analysis report
        report_text = f"""
        Tumor Segmentation Report

        Total Tumor Regions: {tumor_count}
        Tumor Area: {tumor_area:.2f} pixels
        Total Image Area: {total_area} pixels
        Tumor Coverage: {tumor_percentage:.2f}%
        """

        st.download_button("📥 Download Analysis Report (TXT)", report_text, file_name="tumor_report.txt", mime="text/plain")

    st.success("✅ Analysis Complete")
