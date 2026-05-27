# %%writefile app.py
import streamlit as st
import os


st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
        """
        This application demonstrates an end-to-end computer vision project
        focused on image classification and object detection.

        **Project Phases:**
        - Data Preparation
        - Image Classification (VGG16, ResNet50, MobileNetV2, EfficientNetB0)
        - Object Detection (YOLOv8)
        - Model Integration & Pipeline Development
        - Performance Optimization (Quantization, Batched Inference)

        **Use the sidebar to navigate through the different sections of the application.**
        """
    )

st.subheader("Key Features")
st.markdown(
        """
        - **Image Classification:** Classify single objects from uploaded images.
        - **Object Detection:** Detect multiple objects in an image using YOLOv8.
        - **Model Comparison:** Compare the performance of various classification models.
        - **Optimized Inference:** Experience faster predictions with quantized models and batched processing.
        """
    )

st.subheader("How to Use")
st.markdown(
        """
        1. Select a page from the sidebar navigation.
        2. Follow the instructions on each page to upload images, view predictions, or explore model metrics.
        """
    )

    #st.image("https://i.ibb.co/Czd1b56/image.png", width=800)
    #st.image("homepage_banner.png", width=800, caption="SmartVision AI End-to-End System Architecture")

# elif app_page == "Image Classification":
#     st.title("🖼️ Image Classification Platform")
#     st.info("Please use the file directory link in the sidebar menu to launch the dedicated classification script.")

# elif app_page == "Object Detection":
#     st.title("🔍 YOLOv8 Object Detection Hub")
#     st.info("Please click the dedicated detection script link in your sidebar directory map to initialize the YOLO model.")

# elif app_page == "Model Performance Dashboard":
#     st.sidebar.page_link("pages/3_Model_performance.py", label="Manage users")
#     st.title("📊 Architecture Performance & Telemetry")
#     st.info("Head over to the specialized performance sub-file via the sidebar list to see graphs and tables.")

# elif app_page == "Live Webcam Detection":
#     st.title("🎥 Real-Time Live Webcam Stream")
#     st.info("Launch the live webcam utility from the sidebar selection list to link your hardware camera feed.")


