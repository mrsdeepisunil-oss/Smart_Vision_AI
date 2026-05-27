# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import os
# import json
# from ultralytics import YOLO
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # --- Configuration & Model Loading ---
# # BASE_DIR = "smartvision_dataset"
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # 25 Selected Classes (CORRECT indices from detection-datasets/coco)
# # Added here for robustness in case metadata.json fails to load it
# FALLBACK_SELECTED_CLASSES = {
#     'person': 0,
#     'bicycle': 1,
#     'car': 2,
#     'motorcycle': 3,
#     'airplane': 4,
#     'bus': 5,
#     'train': 6,
#     'truck': 7,
#     'traffic light': 9,
#     'stop sign': 11,
#     'bench': 13,
#     'bird': 14,
#     'cat': 15,
#     'dog': 16,
#     'horse': 17,
#     'cow': 19,
#     'elephant': 20,
#     'bottle': 39,
#     'cup': 41,
#     'bowl': 45,
#     'pizza': 53,
#     'cake': 55,
#     'chair': 56,
#     'couch': 57,
#     'potted plant': 58,
#     'bed': 59
# }

# try:
#     with open(os.path.join(BASE_DIR, "dataset_metadata.json"), 'r') as f:
#         metadata = json.load(f)

#     # Reconstruct custom_class_names for YOLO (ordered by YOLO index 0-25)
#     selected_coco_classes_dict = metadata.get('selected_coco_classes')
#     if not selected_coco_classes_dict:
#         st.warning("Warning: 'selected_coco_classes' not found in dataset_metadata.json. Using fallback class definitions.")
#         selected_coco_classes_dict = FALLBACK_SELECTED_CLASSES

#     custom_class_names = [name for name, _id in sorted(selected_coco_classes_dict.items(), key=lambda item: item[1])]
#     yolo_idx_to_name = {idx: name for idx, name in enumerate(custom_class_names)}

#     # Reconstruct class_idx_to_name_classifier for CNN (alphabetically sorted)
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
#     # Assuming 'yolov8n_custom/weights/best.pt' is the path from training
#     # yolo_model_path = os.path.join("runs", "detect", "yolov8n_custom", "weights", "best.pt")
#     # --- TO THIS CLEAN, RELIABLE PATH ---
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Safeguards the root path
#     yolo_model_path = os.path.join(BASE_DIR, "models", "yolov8n.pt")
#     if not os.path.exists(yolo_model_path):
#         st.error(f"YOLOv8 best weights not found at {yolo_model_path}. Please ensure YOLOv8 training was completed.")
#         st.stop()
#     yolov8_model = YOLO(yolo_model_path)
#     yolov8_model.eval()
#     # YOLOv8 models handle their own device placement internally often to CUDA if available.
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
#         # Load best float weights into an unquantized model, then quantize it dynamically
#         temp_efficientnet = models.efficientnet_b0(weights=None)
#         temp_efficientnet.classifier[1] = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(num_ftrs_efficientnet, num_classes_arg)
#         )
#         if os.path.exists(model_save_path_efficientnet_best):
#             temp_efficientnet.load_state_dict(torch.load(model_save_path_efficientnet_best, map_location='cpu'))
#             temp_efficientnet.eval() # Set to eval before quantization
#             efficientnet_model = torch.quantization.quantize_dynamic(
#                 temp_efficientnet, {torch.nn.Linear}, dtype=torch.qint8
#             )
#             # Keep quantized model on CPU as quantized operations are typically CPU-bound
#             efficientnet_model = efficientnet_model.to('cpu')
#             st.success(f"✅ Quantized EfficientNetB0 model loaded from {model_save_path_efficientnet_quantized}.")
#         else:
#             st.warning(f"⚠️ Warning: Non-quantized EfficientNetB0 weights not found at {model_save_path_efficientnet_best}. Cannot create quantized model. Falling back to non-quantized pre-trained ImageNet weights.")
#             # Fallback for quantized path if base model weights are missing
#             efficientnet_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
#             efficientnet_model.classifier[1] = nn.Sequential(
#                 nn.Dropout(0.2),
#                 nn.Linear(num_ftrs_efficientnet, num_classes_arg)
#             )
#             efficientnet_model = efficientnet_model.to(device_arg) # Move to original device if not quantized
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

# # --- Inference Pipeline Function (Adapted for Streamlit) ---
# def streamlit_inference_pipeline(original_img, yolov8_model, efficientnet_model, verify_with_classification=True,
#                                yolo_conf_threshold=0.25, yolo_iou_threshold=0.7, cnn_conf_threshold=0.7):
#     img_width, img_height = original_img.size

#     fig, ax = plt.subplots(1, figsize=(12, 12))
#     ax.imshow(original_img)
#     ax.set_title(f"YOLOv8 Detections {('with CNN Verification' if verify_with_classification else '')}")
#     ax.axis('off')

#     # 1. YOLOv8 Detection
#     yolo_results = yolov8_model.predict(original_img, conf=yolo_conf_threshold, iou=yolo_iou_threshold, verbose=False)

#     for r in yolo_results:
#         for *xyxy, conf, cls in r.boxes.data:
#             x1, y1, x2, y2 = [int(v) for v in xyxy] # Bounding box coordinates
#             yolo_label = yolo_idx_to_name[int(cls)] # YOLO's predicted class name
#             yolo_conf = conf.item() # YOLO's confidence score

#             final_label = yolo_label
#             final_conf = yolo_conf
#             color = 'red' # Default color for YOLO detections

#             # 2. Optional Classification Verification with EfficientNetB0
#             if verify_with_classification:
#                 try:
#                     # Crop the object region
#                     cropped_object_pil = original_img.crop((x1, y1, x2, y2))

#                     # Preprocess and infer with EfficientNetB0
#                     input_tensor = preprocess_classification_image(cropped_object_pil)
#                     input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by a model

#                     # Determine device for CNN model based on whether it's quantized
#                     if efficientnet_model.training or not isinstance(efficientnet_model, torch.quantization.QuantizedModule):
#                         # Non-quantized model goes to GPU if available
#                         input_batch = input_batch.to(device)
#                     else:
#                         # Quantized model stays on CPU
#                         input_batch = input_batch.to('cpu')

#                     with torch.no_grad():
#                         output = efficientnet_model(input_batch)
#                         probabilities = torch.nn.functional.softmax(output[0], dim=0) # Get probabilities
#                         cnn_conf, cnn_idx = torch.max(probabilities, 0) # Get best class and confidence

#                     cnn_label = class_idx_to_name_classifier[cnn_idx.item()]
#                     cnn_conf = cnn_conf.item()

#                     # Decision logic: Use CNN's prediction if it's more confident or if they agree
#                     if cnn_conf >= cnn_conf_threshold and cnn_label == yolo_label:
#                         final_label = f"{cnn_label} (CNN-V)"
#                         final_conf = cnn_conf
#                         color = 'green' # Indicate CNN verification
#                     elif cnn_conf >= cnn_conf_threshold and cnn_conf > yolo_conf:
#                         final_label = f"{cnn_label} (CNN)"
#                         final_conf = cnn_conf
#                         color = 'orange' # CNN override

#                 except Exception as e:
#                     st.warning(f"Warning during CNN verification for this object: {e}. Falling back to YOLO detection.")
#                     pass # Fallback to YOLO if CNN verification fails

#             # 3. Visualization
#             rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
#                                      linewidth=2, edgecolor=color, facecolor='none')
#             ax.add_patch(rect)

#             display_text = f'{final_label} {final_conf:.2f}'
#             plt.text(x1, y1 - 10, display_text, color='white', fontsize=10,
#                      bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2))

#     st.pyplot(fig)
#     plt.close(fig) # Close the figure to prevent memory issues

# # --- Streamlit App Layout ---
# st.header("🔍 Object Detection")
# st.write("Upload an image to perform object detection using YOLOv8. Optionally, verify detections with an EfficientNetB0 classifier.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert('RGB')
#     st.image(image, caption='Uploaded Image', use_column_width=True)
#     st.write("")

#     st.subheader("Detection Settings")
#     col1, col2 = st.columns(2)
#     with col1:
#         yolo_conf_threshold = st.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
#     with col2:
#         yolo_iou_threshold = st.slider("YOLO IOU Threshold (for NMS)", 0.0, 1.0, 0.7, 0.05)

#     verify_with_classification = st.checkbox("Verify Detections with CNN Classifier (EfficientNetB0)", value=True)

#     cnn_conf_threshold = 0.7 # Default
#     if verify_with_classification:
#         cnn_conf_threshold = st.slider("CNN Verification Confidence Threshold", 0.0, 1.0, 0.7, 0.05)

#     if st.button("Run Object Detection"):
#         with st.spinner("Running detection..."):
#             streamlit_inference_pipeline(
#                 image,
#                 yolov8_model_optimized,
#                 efficientnet_model_optimized,
#                 verify_with_classification,
#                 yolo_conf_threshold,
#                 yolo_iou_threshold,
#                 cnn_conf_threshold
#             )
#         st.success("Detection complete!")
# else:
#     st.info("Upload an image to perform object detection.")


import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Configuration & Model Loading ---
# Resolve paths dynamically relative to where this file lives (D:\Smart vision\pages)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # Goes up one level to D:\Smart vision
BASE_DIR = os.path.join(PROJECT_ROOT, "smartvision_dataset")

# 25 Selected Classes (CORRECT indices from detection-datasets/coco)
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
        st.warning("Warning: 'selected_coco_classes' not found in dataset_metadata.json. Using fallback class definitions.")
        selected_coco_classes_dict = FALLBACK_SELECTED_CLASSES

    custom_class_names = [name for name, _id in sorted(selected_coco_classes_dict.items(), key=lambda item: item[1])]
    yolo_idx_to_name = {idx: name for idx, name in enumerate(custom_class_names)}

    class_names_cnn_sorted = sorted(metadata['classes'].keys())
    class_idx_to_name_classifier = {i: name for i, name in enumerate(class_names_cnn_sorted)}
    num_classes = len(class_names_cnn_sorted)

except FileNotFoundError:
    st.error(f"Error: dataset_metadata.json not found in {BASE_DIR}. Please run the data preparation steps.")
    st.stop()
except KeyError as e:
    st.error(f"Error: Missing key in dataset_metadata.json: {e}. Metadata might be corrupted or incomplete.")
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
    st.info("Loading YOLOv8 and EfficientNetB0 models for inference...")

    # --- FIXED PATH EXTRACTION ---
    # Look for the model directly in the project root directory or create fallback
    yolo_model_path = os.path.join(PROJECT_ROOT, "yolov8n.pt")
    
    # If you put it inside a models folder, use this instead:
    # yolo_model_path = os.path.join(PROJECT_ROOT, "models", "yolov8n.pt")

    if not os.path.exists(yolo_model_path):
        # Fallback: If the file isn't downloaded locally yet, download it automatically
        st.warning(f"⚠️ YOLOv8 weights not found at target path. Automatically downloading baseline yolov8n.pt configuration...")
        yolov8_model = YOLO("yolov8n.pt")
        # Save it locally right away so it doesn't need downloading next time
        yolov8_model.save(yolo_model_path)
    else:
        yolov8_model = YOLO(yolo_model_path)
        
    yolov8_model.eval()
    st.success("✅ YOLOv8 model loaded and set to eval mode.")

    # 2. Load EfficientNetB0 Model (quantized) for CPU inference
    efficientnet_model = models.efficientnet_b0(weights=None)
    num_ftrs_efficientnet = efficientnet_model.classifier[1].in_features
    efficientnet_model.classifier[1] = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs_efficientnet, num_classes_arg)
    )

    model_save_path_efficientnet_quantized = os.path.join(base_dir_arg, "efficientnetb0_classifier_quantized.pth")
    model_save_path_efficientnet_best = os.path.join(base_dir_arg, "efficientnetb0_classifier_best.pth")

    if os.path.exists(model_save_path_efficientnet_quantized):
        temp_efficientnet = models.efficientnet_b0(weights=None)
        temp_efficientnet.classifier[1] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs_efficientnet, num_classes_arg)
        )
        if os.path.exists(model_save_path_efficientnet_best):
            temp_efficientnet.load_state_dict(torch.load(model_save_path_efficientnet_best, map_location='cpu'))
            temp_efficientnet.eval()
            efficientnet_model = torch.quantization.quantize_dynamic(
                temp_efficientnet, {torch.nn.Linear}, dtype=torch.qint8
            )
            efficientnet_model = efficientnet_model.to('cpu')
            st.success(f"✅ Quantized EfficientNetB0 model loaded.")
        else:
            efficientnet_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            efficientnet_model.classifier[1] = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_ftrs_efficientnet, num_classes_arg)
            )
            efficientnet_model = efficientnet_model.to(device_arg)
            st.warning("Using EfficientNetB0 fallback with pre-trained ImageNet weights.")

    elif os.path.exists(model_save_path_efficientnet_best):
        efficientnet_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        efficientnet_model.classifier[1] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs_efficientnet, num_classes_arg)
        )
        efficientnet_model.load_state_dict(torch.load(model_save_path_efficientnet_best, map_location=device_arg))
        efficientnet_model = efficientnet_model.to(device_arg)
        st.success(f"✅ Non-quantized EfficientNetB0 weights loaded.")
    else:
        st.error(f"⚠️ Error: No EfficientNetB0 weights found. Falling back to base pre-trained network model.")
        efficientnet_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        efficientnet_model.classifier[1] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs_efficientnet, num_classes_arg)
        )
        efficientnet_model = efficientnet_model.to(device_arg)

    efficientnet_model.eval()
    return yolov8_model, efficientnet_model

yolov8_model_optimized, efficientnet_model_optimized = load_inference_models_optimized_streamlit(num_classes, device, BASE_DIR)

# --- Inference Pipeline Function ---
def streamlit_inference_pipeline(original_img, yolov8_model, efficientnet_model, verify_with_classification=True,
                               yolo_conf_threshold=0.25, yolo_iou_threshold=0.7, cnn_conf_threshold=0.7):
    img_width, img_height = original_img.size

    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(original_img)
    ax.set_title(f"YOLOv8 Detections {('with CNN Verification' if verify_with_classification else '')}")
    ax.axis('off')

    yolo_results = yolov8_model.predict(original_img, conf=yolo_conf_threshold, iou=yolo_iou_threshold, verbose=False)

    for r in yolo_results:
        for *xyxy, conf, cls in r.boxes.data:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            
            # Bound check indices to protect against index parsing crashes
            class_idx = int(cls)
            if class_idx < len(yolo_idx_to_name):
                yolo_label = yolo_idx_to_name[class_idx]
            else:
                yolo_label = "unknown"
                
            yolo_conf = conf.item()

            final_label = yolo_label
            final_conf = yolo_conf
            color = 'red'

            if verify_with_classification:
                try:
                    cropped_object_pil = original_img.crop((x1, y1, x2, y2))
                    input_tensor = preprocess_classification_image(cropped_object_pil)
                    input_batch = input_tensor.unsqueeze(0)

                    if isinstance(efficientnet_model, torch.quantization.QuantizedModule):
                        input_batch = input_batch.to('cpu')
                    else:
                        input_batch = input_batch.to(device)

                    with torch.no_grad():
                        output = efficientnet_model(input_batch)
                        probabilities = torch.nn.functional.softmax(output[0], dim=0)
                        cnn_conf, cnn_idx = torch.max(probabilities, 0)

                    cnn_label = class_idx_to_name_classifier[cnn_idx.item()]
                    cnn_conf = cnn_conf.item()

                    if cnn_conf >= cnn_conf_threshold and cnn_label == yolo_label:
                        final_label = f"{cnn_label} (CNN-V)"
                        final_conf = cnn_conf
                        color = 'green'
                    elif cnn_conf >= cnn_conf_threshold and cnn_conf > yolo_conf:
                        final_label = f"{cnn_label} (CNN)"
                        final_conf = cnn_conf
                        color = 'orange'

                except Exception as e:
                    pass

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            display_text = f'{final_label} {final_conf:.2f}'
            plt.text(x1, y1 - 10, display_text, color='white', fontsize=10,
                     bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2))

    st.pyplot(fig)
    plt.close(fig)

# --- Streamlit App Layout ---
st.header("🔍 Object Detection")
st.write("Upload an image to perform object detection using YOLOv8.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # FIX: Updated to modern use_container_width flag
    st.image(image, caption='Uploaded Image', use_container_width=True)

    st.subheader("Detection Settings")
    col1, col2 = st.columns(2)
    with col1:
        yolo_conf_threshold = st.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    with col2:
        yolo_iou_threshold = st.slider("YOLO IOU Threshold (for NMS)", 0.0, 1.0, 0.7, 0.05)

    verify_with_classification = st.checkbox("Verify Detections with CNN Classifier (EfficientNetB0)", value=True)

    cnn_conf_threshold = 0.7
    if verify_with_classification:
        cnn_conf_threshold = st.slider("CNN Verification Confidence Threshold", 0.0, 1.0, 0.7, 0.05)

    if st.button("Run Object Detection"):
        with st.spinner("Running detection..."):
            streamlit_inference_pipeline(
                image,
                yolov8_model_optimized,
                efficientnet_model_optimized,
                verify_with_classification,
                yolo_conf_threshold,
                yolo_iou_threshold,
                cnn_conf_threshold
            )
        st.success("Detection complete!")