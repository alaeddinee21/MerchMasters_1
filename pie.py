import streamlit as st

st.set_page_config(layout="wide")  # Ensure this is the first command

import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import torch
import cv2  # Import OpenCV for color conversion

# Load model function
@st.cache_resource
def load_model():
    model_path = r"C:\Users\hp\Desktop\ProductDet\ramy.pt"  # Ensure correct path
    model = YOLO(model_path)
    return model

model = load_model()

# Function to detect objects
def detect_objects(image):
    results = model(image,conf=0.3)  # Run YOLO detection
    
    # Extract detected classes and counts
    detections = results[0].names  # Get class names
    counts = {}  # Store count of each detected object

    for box in results[0].boxes:
        class_id = int(box.cls)  # Class index
        class_name = detections[class_id].strip().lower()  # Normalize class names
        counts[class_name] = counts.get(class_name, 0) + 1  # Increment count

    # Reset Matplotlib style to default
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    result_img = results[0].plot()

    # Convert BGR to RGB (fixes the blue color issue)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    return result_img, counts  # Return image with bounding boxes + detection summary

# Streamlit UI
st.title("Retail Shelf Detection")

# Sidebar: Upload & Preview Image
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

# Main Layout: Detection Results & Pie Chart
if uploaded_file:
    if st.button("Detect Products"):
        result_img, detections = detect_objects(image)

        col1, col2 = st.columns(2)  # Create two equal columns

        with col1:
            st.subheader("Detection Results")
            st.image(result_img, caption="Processed Image", use_container_width=True)

        with col2:
            st.subheader("Detection Summary")

            # Aggregate counts for case-insensitive classes
            aggregated_counts = {}
            for brand, count in detections.items():
                normalized_brand = brand.capitalize()  # Ensure consistent display
                aggregated_counts[normalized_brand] = aggregated_counts.get(normalized_brand, 0) + count

            total_count = sum(aggregated_counts.values())

            for brand, count in aggregated_counts.items():
                st.write(f"- **{brand}:** {count} detected")

            # Pie chart for all product categories
            if total_count > 0:
                st.subheader("Product Distribution")

                plt.style.use("dark_background")  # Remove white background

                sizes = list(aggregated_counts.values())
                colors = plt.cm.Set2.colors[:len(sizes)]  # Use a softer color map

                fig, ax = plt.subplots(figsize=(2, 2))  # Adjusted size

                # Compute autopct function with jitter to prevent overlap
                def autopct_jitter(pct):
                    if pct > 2:
                        return f"{pct:.1f}%"
                    return ""

                wedges, texts, autotexts = ax.pie(
                    sizes,
                    labels=None,  # No labels
                    autopct=autopct_jitter,  # Jitter function
                    colors=colors,
                    startangle=90,
                    pctdistance=0.85,
                )

                ax.axis("equal")  # Ensures pie is drawn as a circle

                # Show legend separately for clarity
                if len(aggregated_counts) > 1:
                    ax.legend(aggregated_counts.keys(), loc="center left", bbox_to_anchor=(1, 0.5))

                st.pyplot(fig)

            # Pie chart for Ramy vs Competitors
            st.subheader("Ramy vs Competitors")

            ramy_count = sum(count for brand, count in aggregated_counts.items() if "ramy" in brand.lower())
            competitors_count = total_count - ramy_count

            if total_count > 0:
                labels = ["Ramy", "Competitors"]
                sizes = [ramy_count, competitors_count]
                colors = ["#1f77b4", "#ff7f0e"]  # Blue for Ramy, Orange for competitors

                fig2, ax2 = plt.subplots(figsize=(2, 2))
                ax2.pie(
                    sizes,
                    labels=labels,
                    autopct="%1.1f%%",
                    colors=colors,
                    startangle=90,
                    pctdistance=0.85,
                )
                ax2.axis("equal")  # Ensure pie is a circle

                st.pyplot(fig2)
