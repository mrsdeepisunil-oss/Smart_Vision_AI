#  %%writefile pages/1_Image_Classification.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import json # To load metadata

# --- Configuration & Model Loading ---
# Assuming BASE_DIR is consistent, loading SELECTED_CLASSES from metadata.json
BASE_DIR = "smartvision_dataset"

try:
    with open(os.path.join(BASE_DIR, "dataset_metadata.json"), 'r') as f:
        metadata = json.load(f)
    # Reconstruct class_idx_to_name_classifier from metadata
    class_names_list = sorted(metadata['classes'].keys())
    num_classes = len(class_names_list)
    class_idx_to_name_classifier = {i: name for i, name in enumerate(class_names_list)}

except FileNotFoundError:
    st.error(f"Error: dataset_metadata.json not found in {BASE_DIR}. Please run the data preparation steps.")
    st.stop()
except KeyError:
    st.error("Error: 'classes' key not found in dataset_metadata.json. Metadata might be corrupted.")
    st.stop()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ImageNet statistics for normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

@st.cache_resource # Cache the models to avoid reloading on each rerun
def load_classification_models(num_classes, device, base_dir):
    models_dict = {}

    # VGG16
    model_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for param in model_vgg.features.parameters():
        param.requires_grad = False
    num_ftrs_vgg = model_vgg.classifier[6].in_features
    model_vgg.classifier[6] = nn.Sequential(
        nn.Linear(num_ftrs_vgg, 512),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    vgg_path = os.path.join(base_dir, "vgg16_classifier_best.pth")
    if os.path.exists(vgg_path):
        model_vgg.load_state_dict(torch.load(vgg_path, map_location=device))
    model_vgg = model_vgg.to(device);
    model_vgg.eval()
    models_dict['VGG16'] = model_vgg

    # ResNet50
    model_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model_resnet.parameters():
        param.requires_grad = False
    num_ftrs_resnet = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_ftrs_resnet, num_classes)
    resnet_path = os.path.join(base_dir, "resnet50_classifier_best.pth")
    if os.path.exists(resnet_path):
        model_resnet.load_state_dict(torch.load(resnet_path, map_location=device))
    model_resnet = model_resnet.to(device);
    model_resnet.eval()
    models_dict['ResNet50'] = model_resnet

    # MobileNetV2
    # Using pretrained=True due to torchvision version issues encountered in notebook
    model_mobilenet = models.mobilenet_v2(pretrained=True)
    for param in model_mobilenet.features.parameters():
        param.requires_grad = False
    num_ftrs_mobilenet = model_mobilenet.classifier[1].in_features
    model_mobilenet.classifier[1] = nn.Linear(num_ftrs_mobilenet, num_classes)
    mobilenet_path = os.path.join(base_dir, "mobilenetv2_classifier_best.pth")
    if os.path.exists(mobilenet_path):
        model_mobilenet.load_state_dict(torch.load(mobilenet_path, map_location=device))
    model_mobilenet = model_mobilenet.to(device);
    model_mobilenet.eval()
    models_dict['MobileNetV2'] = model_mobilenet

    # EfficientNetB0 (Non-quantized)
    model_efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model_efficientnet.features.parameters():
        param.requires_grad = False
    num_ftrs_efficientnet = model_efficientnet.classifier[1].in_features
    model_efficientnet.classifier[1] = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs_efficientnet, num_classes)
    )
    efficientnet_path = os.path.join(base_dir, "efficientnetb0_classifier_best.pth")
    if os.path.exists(efficientnet_path):
        model_efficientnet.load_state_dict(torch.load(efficientnet_path, map_location=device))
    model_efficientnet = model_efficientnet.to(device);
    model_efficientnet.eval()
    models_dict['EfficientNetB0'] = model_efficientnet

    # EfficientNetB0 (Quantized)
    # Load the base model, then apply dynamic quantization
    model_efficientnet_quantized_base = models.efficientnet_b0(weights=None)
    for param in model_efficientnet_quantized_base.features.parameters():
        param.requires_grad = False
    num_ftrs_efficientnet_quantized = model_efficientnet_quantized_base.classifier[1].in_features
    model_efficientnet_quantized_base.classifier[1] = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs_efficientnet_quantized, num_classes)
    )
    # Load original best weights before quantization
    if os.path.exists(efficientnet_path):
        model_efficientnet_quantized_base.load_state_dict(torch.load(efficientnet_path, map_location='cpu'))
    model_efficientnet_quantized_base.eval() # Set to eval before quantization

    # Apply dynamic quantization
    model_efficientnet_quantized = torch.quantization.quantize_dynamic(
        model_efficientnet_quantized_base, {torch.nn.Linear}, dtype=torch.qint8
    )
    # Keep quantized model on CPU as quantized operations are typically CPU-bound
    models_dict['EfficientNetB0 (Quantized)'] = model_efficientnet_quantized.to('cpu')

    return models_dict

classification_models = load_classification_models(num_classes, device, BASE_DIR)

st.header("🧠 Image Classification")
st.write("Upload an image to classify objects using various pre-trained CNN models. See how different architectures perform on the same input!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width=700)
    st.write("")
    st.subheader("Classification Results")

    # Prepare input tensor once for all models
    base_input_tensor = val_test_transforms(image).unsqueeze(0)

    results_df = []

    for model_name, model_instance in classification_models.items():
        if "Quantized" in model_name: # Quantized model runs on CPU
            input_tensor = base_input_tensor.to('cpu')
        else: # Other models run on configured device (GPU if available)
            input_tensor = base_input_tensor.to(device)

        with torch.no_grad():
            outputs = model_instance(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            top5_prob, top5_idx = torch.topk(probabilities, 5)

            model_results = {'Model': model_name}
            for i in range(5):
                label = class_idx_to_name_classifier[top5_idx[i].item()]
                prob = top5_prob[i].item()
                model_results[f'Top {i+1} Label'] = label
                model_results[f'Top {i+1} Confidence'] = f"{prob:.2%}"
            results_df.append(model_results)

    st.table(results_df)

    # Detailed comparison (optional, could be in a separate expander)
    st.subheader("Detailed Model Output")
    for model_name, model_instance in classification_models.items():
        with st.expander(f"Show {model_name} Details"):
            st.write(f"**Model:** {model_name}")
            # Prepare input tensor for expander section
            if "Quantized" in model_name:
                input_tensor = base_input_tensor.to('cpu')
            else:
                input_tensor = base_input_tensor.to(device)

            with torch.no_grad():
                outputs = model_instance(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                top5_prob, top5_idx = torch.topk(probabilities, 5)

                for i in range(len(top5_idx)):
                    label = class_idx_to_name_classifier[top5_idx[i].item()]
                    prob = top5_prob[i].item()
                    st.write(f"**{label}**: {prob:.2%}")

else:
    st.info("Upload an image to get classification predictions.")
