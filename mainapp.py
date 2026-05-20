# %%writefile app.py
import streamlit as st
import os


st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# st.sidebar.title("SmartVision AI")
st.sidebar.markdown("# 👁️ SmartVisionAI")
st.sidebar.write("---")  # Horizontal line separator



# --- 2. DYNAMIC PATH ROUTING SETUP ---
# Automatically resolves the paths whether you run on Windows or Mac/Linux
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAGES_DIR = os.path.join(BASE_DIR, "pages")

# # Dictionary mapping dropdown selection keys to their exact .py script paths
# PAGE_ROUTES = {
#     "Home Overview": None,  # Handled locally inside this file below
#     "Image Classification": os.path.join(PAGES_DIR, "2_Image_Classification.py"),
#     "Object Detection": os.path.join(PAGES_DIR, "3_Object_Detection.py"),
#     "Model Performance Dashboard": os.path.join(PAGES_DIR, "Model_performance.py"),  # Maps directly to your exact filename
#     "Live Webcam Detection": os.path.join(PAGES_DIR, "5_Live_Webcam_Detection.py")
# }

# st.sidebar.markdown("--- App Navigation ---")

# # Creating the drop-down selection box widget in the sidebar
# app_page = st.sidebar.selectbox(
#     "🧭 Quick Navigation Dashboard",
#     [
#         "Home Overview", 
#         "Image Classification", 
#         "Object Detection", 
#         "Model Performance Dashboard", 
#         "Live Webcam Detection"
#     ]
# )


st.sidebar.info("💡 Note: You can also use Streamlit's native file sidebar links below to change pages.")

# --- MAIN PAGE ROUTING CONTENT BASED ON SELECTION ---
# if app_page == "Home Overview":
st.title("Welcome to SmartVision AI!")

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


