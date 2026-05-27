# #%%writefile pages/4_Live_Webcam_Detection.py
# import streamlit as st
# import cv2
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import os
# import json
# from ultralytics import YOLO

# # --- Configuration & Model Loading ---
# BASE_DIR = "smartvision_dataset"

# # Initialize camera capture variable to prevent scoping/Pylance errors
# cap = None

# try:
#     with open(os.path.join(BASE_DIR, "dataset_metadata.json"), 'r') as f:
#         metadata = json.load(f)

#     selected_coco_classes_dict = metadata.get('selected_coco_classes')
#     if not selected_coco_classes_dict:
#         st.error("Error: 'selected_coco_classes' not found in dataset_metadata.json. Please ensure metadata is complete.")
#         st.stop()
#     custom_class_names = [name for name, _id in sorted(selected_coco_classes_dict.items(), key=lambda item: item[1])]
#     yolo_idx_to_name = {idx: name for idx, name in enumerate(custom_class_names)}

#     class_names_cnn_sorted = sorted(metadata['classes'].keys())
#     class_idx_to_name_classifier = {i: name for i, name in enumerate(class_names_cnn_sorted)}
#     num_classes = len(class_names_cnn_sorted)

# except FileNotFoundError:
#     st.error(f"Error: dataset_metadata.json not found in {BASE_DIR}. Please run the data preparation steps.")
#     st.stop()
# except KeyError as e:
#     st.error(f"Error: Missing key in dataset_metadata.json: {e}. Metadata might be corrupted or incomplete.")
#     st.stop()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # ImageNet statistics for normalization
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# # Transformation for classification model input (EfficientNetB0)
# preprocess_classification_image = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     normalize
# ])

# @st.cache_resource
# def load_inference_models_optimized_streamlit(num_classes_arg, device_arg, base_dir_arg):
#     st.info("Loading YOLOv8 and EfficientNetB0 models for inference...")

#     # 1. Load YOLOv8 Model
#     yolo_model_path = os.path.join("runs", "detect", "yolov8n_custom", "weights", "best.pt")
#     if not os.path.exists(yolo_model_path):
#         st.error(f"YOLOv8 best weights not found at {yolo_model_path}. Please ensure YOLOv8 training was completed.")
#         st.stop()
#     yolov8_model = YOLO(yolo_model_path)
#     yolov8_model.eval()
#     st.success("✅ YOLOv8 model loaded and set to eval mode.")

#     # 2. Load EfficientNetB0 Model (quantized) for CPU inference
#     efficientnet_model = models.efficientnet_b0(weights=None) # Load architecture without weights initially
#     num_ftrs_efficientnet = efficientnet_model.classifier[1].in_features
#     efficientnet_model.classifier[1] = nn.Sequential(
#         nn.Dropout(0.2),
#         nn.Linear(num_ftrs_efficientnet, num_classes_arg)
#     )

#     model_save_path_efficientnet_quantized = os.path.join(base_dir_arg, "efficientnetb0_classifier_quantized.pth")
#     model_save_path_efficientnet_best = os.path.join(base_dir_arg, "efficientnetb0_classifier_best.pth")

#     if os.path.exists(model_save_path_efficientnet_quantized):
#         temp_efficientnet = models.efficientnet_b0(weights=None)
#         temp_efficientnet.classifier[1] = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(num_ftrs_efficientnet, num_classes_arg)
#         )
#         if os.path.exists(model_save_path_efficientnet_best):
#             temp_efficientnet.load_state_dict(torch.load(model_save_path_efficientnet_best, map_location='cpu'))
#             temp_efficientnet.eval()
#             efficientnet_model = torch.quantization.quantize_dynamic(
#                 temp_efficientnet, {torch.nn.Linear}, dtype=torch.qint8
#             )
#             # Keep quantized model on CPU
#             efficientnet_model = efficientnet_model.to('cpu')
#             st.success(f"✅ Quantized EfficientNetB0 model loaded.")
#         else:
#             st.warning(f"⚠️ Warning: Non-quantized EfficientNetB0 weights not found at {model_save_path_efficientnet_best}. Cannot create quantized model. Falling back to non-quantized pre-trained ImageNet weights.")
#             efficientnet_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
#             efficientnet_model.classifier[1] = nn.Sequential(
#                 nn.Dropout(0.2),
#                 nn.Linear(num_ftrs_efficientnet, num_classes_arg)
#             )
#             efficientnet_model = efficientnet_model.to(device_arg) # Fallback to original device
#             st.warning("Using EfficientNetB0 with pre-trained ImageNet weights and modified head.")

#     elif os.path.exists(model_save_path_efficientnet_best):
#         efficientnet_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
#         efficientnet_model.classifier[1] = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(num_ftrs_efficientnet, num_classes_arg)
#         )
#         efficientnet_model.load_state_dict(torch.load(model_save_path_efficientnet_best, map_location=device_arg))
#         efficientnet_model = efficientnet_model.to(device_arg)
#         st.success(f"✅ Non-quantized EfficientNetB0 weights loaded from {model_save_path_efficientnet_best}.")
#     else:
#         st.error(f"⚠️ Error: No EfficientNetB0 weights found at {model_save_path_efficientnet_quantized} or {model_save_path_efficientnet_best}. Cannot load EfficientNetB0. Please ensure training and quantization steps were completed.")
#         st.stop()

#     efficientnet_model.eval()
#     st.success("✅ EfficientNetB0 model loaded and set to eval mode.")

#     return yolov8_model, efficientnet_model

# yolov8_model_optimized, efficientnet_model_optimized = load_inference_models_optimized_streamlit(num_classes, device, BASE_DIR)

# # --- Streamlit App Layout ---
# st.header("🎥 Live Webcam Detection")
# st.write("Experience real-time object detection using your webcam with YOLOv8. Optionally, use EfficientNetB0 for class verification.")

# # Placeholder for the video feed
# video_placeholder = st.empty()

# # Settings
# st.sidebar.subheader("Webcam Detection Settings")
# yolo_conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
# yolo_iou_threshold = st.sidebar.slider("YOLO IOU Threshold (for NMS)", 0.0, 1.0, 0.7, 0.05)

# verify_with_classification = st.sidebar.checkbox("Verify Detections with CNN Classifier (EfficientNetB0)", value=True)

# cnn_conf_threshold = 0.7 # Default
# if verify_with_classification:
#     cnn_conf_threshold = st.sidebar.slider("CNN Verification Confidence Threshold", 0.0, 1.0, 0.7, 0.05)

# start_button = st.sidebar.button("Start Webcam")
# stop_button = st.sidebar.button("Stop Webcam")

# camer_on = False
# if start_button:
#     camera_on = True
#     cap = cv2.VideoCapture(0) # 0 for default webcam
#     if not cap.isOpened():
#         st.error("Error: Could not open webcam. Please ensure it's connected and not in use.")
#         camera_on = False
#     else:
#         st.sidebar.success("Webcam started!")

#     fps_text = st.sidebar.empty()
#     frame_count = 0
#     start_time = 0

#     while camera_on:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Failed to grab frame from webcam.")
#             break

#         if start_time == 0:
#             start_time = cv2.getTickCount()

#         # Convert OpenCV BGR to PIL RGB for YOLO and CNN
#         img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#         # Perform YOLOv8 Detection
#         yolo_results = yolov8_model_optimized.predict(img_pil, conf=yolo_conf_threshold, iou=yolo_iou_threshold, verbose=False)

#         # Prepare frame for drawing
#         frame_display = frame.copy()

#         for r in yolo_results:
#             for *xyxy, conf, cls in r.boxes.data:
#                 x1, y1, x2, y2 = [int(v) for v in xyxy]  # Bounding box coordinates
#                 yolo_label = yolo_idx_to_name[int(cls)]  # YOLO's predicted class name
#                 yolo_conf = conf.item()  # YOLO's confidence score

#                 final_label = yolo_label
#                 final_conf = yolo_conf
#                 color = (0, 0, 255)  # BGR for OpenCV (Red for YOLO detections)

#                 # Optional Classification Verification with EfficientNetB0
#                 if verify_with_classification:
#                     try:
#                         cropped_object_pil = img_pil.crop((x1, y1, x2, y2))
#                         input_tensor = preprocess_classification_image(cropped_object_pil)
#                         input_batch = input_tensor.unsqueeze(0)

#                         # Determine device for CNN model based on whether it's quantized
#                         if efficientnet_model_optimized.training or not isinstance(efficientnet_model_optimized, torch.quantization.QuantizedModule):
#                             # Non-quantized model goes to GPU if available
#                             input_batch = input_batch.to(device)
#                         else:
#                             # Quantized model stays on CPU
#                             input_batch = input_batch.to('cpu')

#                         with torch.no_grad():
#                             output = efficientnet_model_optimized(input_batch)
#                             probabilities = torch.nn.functional.softmax(output[0], dim=0)
#                             cnn_conf_val, cnn_idx = torch.max(probabilities, 0)

#                         cnn_label = class_idx_to_name_classifier[cnn_idx.item()]
#                         cnn_conf_val = cnn_conf_val.item()

#                         if cnn_conf_val >= cnn_conf_threshold and cnn_label == yolo_label:
#                             final_label = f"{cnn_label} (CNN-V)"
#                             final_conf = cnn_conf_val
#                             color = (0, 255, 0) # Green for verified
#                         elif cnn_conf_val >= cnn_conf_threshold and cnn_conf_val > yolo_conf:
#                             final_label = f"{cnn_label} (CNN)"
#                             final_conf = cnn_conf_val
#                             color = (0, 165, 255) # Orange for CNN override

#                     except Exception as e:
#                         # st.warning(f"Warning during CNN verification for an object: {e}") # Too verbose for live feed
#                         pass # Fallback to YOLO if CNN verification fails

#                 # Draw bounding box and label on the OpenCV frame
#                 cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
#                 display_text = f'{final_label} {final_conf:.2f}'
#                 cv2.putText(frame_display, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#         # Calculate FPS
#         frame_count += 1
#         end_time = cv2.getTickCount()
#         time_taken = (end_time - start_time) / cv2.getTickFrequency()
#         current_fps = frame_count / time_taken
#         fps_text.write(f"FPS: {current_fps:.2f}")

#         # Display the frame
#         video_placeholder.image(frame_display, channels="BGR", width=True)

#         if stop_button: # or not ret:
#             camera_on = False
#             #st.sidebar.warning("Webcam stopped.")
#             #cap.release()
#             break
#     # Releasing camera resources when loop breaks
#     if cap is not None:
#         cap.release()
#     st.sidebar.warning("Webcam stopped.")
    
# # Releasing camera resources when loop breaks
#     if cap is not None:
#         cap.release()
#     st.sidebar.warning("Webcam stopped.")

# # elif stop_button:
# #     camera_on = False
# #     st.sidebar.warning("Webcam stopped.")
# #     if 'cap' in locals() and cap.isOpened():
# #         cap.release()

# # if not camera_on:
# #     st.info("Click 'Start Webcam' to begin live object detection.")

# # %%


# %%writefile pages/4_Live_Webcam_Detection.py
import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import json
from ultralytics import YOLO

# --- Dynamic Pathing & Layout Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # Navigates to D:\Smart vision
BASE_DIR = os.path.join(PROJECT_ROOT, "smartvision_dataset")

# 25 Selected Classes (Robust fallback mechanism if metadata parsing fields fail)
FALLBACK_SELECTED_CLASSES = {
    'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4,
    'bus': 5, 'train': 6, 'truck': 7, 'traffic light': 9, 'stop sign': 11,
    'bench': 13, 'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'cow': 19,
    'elephant': 20, 'bottle': 39, 'cup': 41, 'bowl': 45, 'pizza': 53,
    'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59
}

try:
    with open(os.path.join(BASE_DIR, "dataset_metadata.json"), 'r') as f:
        metadata = json.load(f)

    selected_coco_classes_dict = metadata.get('selected_coco_classes')
    if not selected_coco_classes_dict:
        st.warning("⚠️ 'selected_coco_classes' not found in metadata. Using robust class fallback overrides.")
        selected_coco_classes_dict = FALLBACK_SELECTED_CLASSES

    custom_class_names = [name for name, _id in sorted(selected_coco_classes_dict.items(), key=lambda item: item[1])]
    yolo_idx_to_name = {idx: name for idx, name in enumerate(custom_class_names)}

    class_names_cnn_sorted = sorted(metadata.get('classes', {}).keys())
    if not class_names_cnn_sorted:
        class_names_cnn_sorted = sorted(list(FALLBACK_SELECTED_CLASSES.keys()))
        
    class_idx_to_name_classifier = {i: name for i, name in enumerate(class_names_cnn_sorted)}
    num_classes = len(class_names_cnn_sorted)

except FileNotFoundError:
    st.warning(f"⚠️ dataset_metadata.json not found in {BASE_DIR}. Activating local standalone mode.")
    selected_coco_classes_dict = FALLBACK_SELECTED_CLASSES
    custom_class_names = [name for name, _id in sorted(selected_coco_classes_dict.items(), key=lambda item: item[1])]
    yolo_idx_to_name = {idx: name for idx, name in enumerate(custom_class_names)}
    class_names_cnn_sorted = sorted(list(FALLBACK_SELECTED_CLASSES.keys()))
    class_idx_to_name_classifier = {i: name for i, name in enumerate(class_names_cnn_sorted)}
    num_classes = len(class_names_cnn_sorted)
except KeyError as e:
    st.error(f"Error parsing metadata key layout: {e}.")
    st.stop()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

preprocess_classification_image = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

@st.cache_resource
def load_inference_models_optimized_streamlit(num_classes_arg, device_arg, base_dir_arg):
    st.info("Initializing baseline verification models for inference execution...")

    # Dynamic lookups matching your current project folder pathing
    yolo_model_path = os.path.join(PROJECT_ROOT, "yolov8n.pt")
    
    if not os.path.exists(yolo_model_path):
        st.warning("⚠️ YOLOv8 model file missing. Downloading base weights package...")
        yolov8_model = YOLO("yolov8n.pt")
        yolov8_model.save(yolo_model_path)
    else:
        yolov8_model = YOLO(yolo_model_path)
        
    yolov8_model.eval()
    st.success("✅ YOLOv8 real-time model interface successfully initiated.")

    efficientnet_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_ftrs_efficientnet = efficientnet_model.classifier[1].in_features
    efficientnet_model.classifier[1] = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs_efficientnet, num_classes_arg)
    )
    
    model_save_path_efficientnet_best = os.path.join(base_dir_arg, "efficientnetb0_classifier_best.pth")
    if os.path.exists(model_save_path_efficientnet_best):
        try:
            efficientnet_model.load_state_dict(torch.load(model_save_path_efficientnet_best, map_location=device_arg))
            st.success("✅ Custom EfficientNetB0 classification parameters applied.")
        except Exception:
            st.warning("⚠️ Custom weights signature mismatched. Preserving default transfer metrics.")
            
    efficientnet_model = efficientnet_model.to(device_arg)
    efficientnet_model.eval()
    
    return yolov8_model, efficientnet_model

yolov8_model_optimized, efficientnet_model_optimized = load_inference_models_optimized_streamlit(num_classes, device, BASE_DIR)

# --- Streamlit App Layout ---
st.header("🎥 Live Webcam Detection")
st.write("Experience real-time object detection using your webcam with YOLOv8.")

video_placeholder = st.empty()

st.sidebar.subheader("Webcam Detection Settings")
yolo_conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.40, 0.05)
yolo_iou_threshold = st.sidebar.slider("YOLO mAP IOU Match Filter", 0.0, 1.0, 0.60, 0.05)
verify_with_classification = st.sidebar.checkbox("Verify Detections with CNN Classifier", value=True)

cnn_conf_threshold = 0.60
if verify_with_classification:
    cnn_conf_threshold = st.sidebar.slider("CNN Verification Confidence Threshold", 0.0, 1.0, 0.60, 0.05)

start_button = st.sidebar.button("Start Webcam Pipeline")
stop_button = st.sidebar.button("Kill Camera Stream Session")

# Check and preserve camera session tracking variables across renders
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if start_button:
    st.session_state.camera_active = True
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error: Local video device resources are inaccessible or locked by another app.")
        st.session_state.camera_active = False
    else:
        st.sidebar.success("Video interface established.")

    fps_text = st.sidebar.empty()
    frame_count = 0
    start_time = cv2.getTickCount()

    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret or stop_button:
            break

        # Convert channels for parsing engines
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        yolo_results = yolov8_model_optimized.predict(img_pil, conf=yolo_conf_threshold, iou=yolo_iou_threshold, verbose=False)
        frame_display = frame.copy()

        for r in yolo_results:
            for *xyxy, conf, cls in r.boxes.data:
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                
                class_idx = int(cls)
                if class_idx < len(yolo_idx_to_name):
                    yolo_label = yolo_idx_to_name[class_idx]
                else:
                    yolo_label = "Object"
                    
                yolo_conf = conf.item()
                final_label, final_conf = yolo_label, yolo_conf
                color = (0, 0, 255)  # Red default

                if verify_with_classification:
                    try:
                        cropped_object_pil = img_pil.crop((x1, y1, x2, y2))
                        input_tensor = preprocess_classification_image(cropped_object_pil)
                        input_batch = input_tensor.unsqueeze(0).to(device)

                        with torch.no_grad():
                            output = efficientnet_model_optimized(input_batch)
                            probabilities = torch.nn.functional.softmax(output[0], dim=0)
                            cnn_conf_val, cnn_idx = torch.max(probabilities, 0)

                        cnn_idx_val = cnn_idx.item()
                        if cnn_idx_val < len(class_idx_to_name_classifier):
                            cnn_label = class_idx_to_name_classifier[cnn_idx_val]
                            cnn_conf_val = cnn_conf_val.item()

                            if cnn_conf_val >= cnn_conf_threshold and cnn_label == yolo_label:
                                final_label = f"{cnn_label} (CNN-V)"
                                final_conf = cnn_conf_val
                                color = (0, 255, 0)
                            elif cnn_conf_val >= cnn_conf_threshold and cnn_conf_val > yolo_conf:
                                final_label = f"{cnn_label} (CNN)"
                                final_conf = cnn_conf_val
                                color = (0, 165, 255)
                    except Exception:
                        pass

                cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
                display_text = f'{final_label} {final_conf:.2f}'
                cv2.putText(frame_display, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame_count += 1
        end_time = cv2.getTickCount()
        time_taken = (end_time - start_time) / cv2.getTickFrequency()
        current_fps = frame_count / time_taken
        fps_text.write(f"Inference Velocity: {current_fps:.1f} FPS")

        # FIX: Replaced outdated width argument with use_container_width
        video_placeholder.image(frame_display, channels="BGR", use_container_width=True)

    # Clean release execution blocks
    st.session_state.camera_active = False
    if 'cap' in locals() and cap is not None:
        cap.release()
    video_placeholder.empty()
    st.sidebar.info("Camera tracking loop terminated.")
else:
    st.info("Click 'Start Webcam Pipeline' in the left menu option to toggle hardware tracking feeds.")