# /breast-cancer-api/app/inference.py

import torch
import numpy as np
import cv2
import base64
import io
from PIL import Image
from .model import MultiTaskUnet

# --- Configuration ---
MODEL_PATH = "weights/best_multitask_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256 # Must match the training image size

# For classification output
LABEL_MAP = {0: 'benign', 1: 'malignant', 2: 'normal'}

# --- Model Loading ---
def load_model():
    """Loads the model and weights only once at startup."""
    model = MultiTaskUnet(
        encoder_name='timm-efficientnet-b0',
        encoder_weights=None, # Set to None, as we are loading our own trained weights
        in_channels=3,
        num_classes=len(LABEL_MAP)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {MODEL_PATH} and moved to {DEVICE}.")
    return model

# --- Preprocessing and Post-processing ---
def preprocess_image(image_bytes: bytes):
    """Converts raw image bytes to a model-ready tensor."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_np = np.array(image, dtype=np.float32)
    # Transpose from (H, W, C) to (C, H, W) and normalize
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1) / 255.0
    # Add batch dimension
    return image_tensor.unsqueeze(0)

def postprocess_mask(mask_tensor: torch.Tensor) -> str:
    """Converts the model's mask tensor to a Base64 encoded string."""
    # Apply sigmoid, threshold, and convert to NumPy array on CPU
    pred_mask = (torch.sigmoid(mask_tensor) > 0.5).float().cpu().numpy().squeeze()
    
    # Convert binary mask to an 8-bit image (0 or 255)
    mask_img_np = (pred_mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_img_np, mode='L') # 'L' for grayscale

    # Save to an in-memory buffer
    buffer = io.BytesIO()
    mask_pil.save(buffer, format="PNG")
    
    # Encode buffer to Base64
    mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return mask_base64

# --- Core Prediction Function ---
def get_prediction(model: MultiTaskUnet, image_bytes: bytes):
    """Runs a single image through the full inference pipeline."""
    # 1. Preprocess
    tensor = preprocess_image(image_bytes).to(DEVICE)
    
    # 2. Inference
    with torch.no_grad():
        seg_logits, class_logits = model(tensor)
    
    # 3. Post-process Classification
    pred_class_id = torch.argmax(class_logits, dim=1).item()
    pred_class_name = LABEL_MAP[pred_class_id]
    
    # 4. Post-process Segmentation Mask
    mask_b64 = postprocess_mask(seg_logits)

    return {
        "predicted_class_id": pred_class_id,
        "predicted_class_name": pred_class_name,
        "mask_base64": mask_b64
    }
    