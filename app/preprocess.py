import cv2
import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional, Union
from PIL import Image, ImageEnhance


def preprocess_segmentation(
    image: Union[str, np.ndarray],
    target_size: Tuple[int, int] = (256, 256),
    normalize: bool = True,
    clahe: bool = True
) -> torch.Tensor:
    """
    Preprocess image for segmentation task.
    
    Args:
        image: Input image path or numpy array (H, W, C) in BGR format
        target_size: Target size as (height, width)
        normalize: Whether to normalize to [0, 1]
        clahe: Whether to apply CLAHE for contrast enhancement
        
    Returns:
        Preprocessed image as torch.Tensor of shape (C, H, W)
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Could not read image: {image}")
    
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply CLAHE to each channel if image is RGB
    if clahe and len(image.shape) == 3:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(image)
        l = clahe.apply(l)
        image = cv2.merge((l, a, b))
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    elif clahe and len(image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
    
    # Resize image
    image = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Convert to float32 and normalize
    image = image.astype(np.float32)
    
    # Normalize to [0, 1] if needed
    if normalize:
        if image.max() > 1.0:
            image = image / 255.0
    
    # Convert to CHW format
    if len(image.shape) == 2:
        # Grayscale image
        image = np.expand_dims(image, axis=0)  # Add channel dimension
    else:
        # RGB image
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    
    return torch.from_numpy(image)


def preprocess_classification(
    image: Union[str, np.ndarray],
    target_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Preprocess image for classification task.
    
    Args:
        image: Input image path or numpy array (H, W, C) in BGR format
        target_size: Target size as (height, width)
        
    Returns:
        Preprocessed image as torch.Tensor of shape (C, H, W)
    """
    # Reuse segmentation preprocessing but with different default size
    return preprocess_segmentation(
        image=image,
        target_size=target_size,
        normalize=True,
        clahe=True
    )


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor back to a numpy array in the range [0, 255].
    
    Args:
        tensor: Input tensor of shape (C, H, W) or (1, C, H, W)
        
    Returns:
        Denormalized numpy array in HWC format with values in [0, 255]
    """
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    
    # Convert to numpy and transpose to HWC
    image = tensor.cpu().numpy()
    if len(image.shape) == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # Scale to [0, 255] and convert to uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    return image
