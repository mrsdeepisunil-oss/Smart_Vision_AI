# 👁️ SmartVision AI

An end-to-end Computer Vision application combining deep learning image classification and real-time object detection into an interactive Streamlit dashboard.

---

## 🚀 Key Features
- **Image Classification:** Classifies individual target items utilizing optimized EfficientNetB0 backbones.
- **Object Detection:** Detects multiple bounding box configurations simultaneously via YOLOv8.
- **Live Streaming Inference:** Integrates webcam capturing streams with parallel classification verification filters.
- **Performance Evaluation:** Generates graphical comparisons of validation metrics directly inside a performance tracking UI module.

---

## 📂 Project Structure
```text
Smart vision/
├── Home.py                       # Application landing entry point script
├── README.md                     # Project documentation guide
├── yolov8n.pt                    # Saved base detection model architecture weights
├── smartvision_dataset/          # Root data workspace environment
│   └── dataset_metadata.json     # Encoded target class boundaries mapping index
└── pages/                        # Multipage routing sub-modules
    ├── 2_🖼️_Object_Detection_Image.py
    ├── 3_Model_Performance.py
    └── 4_Live_Webcam_Detection.py
