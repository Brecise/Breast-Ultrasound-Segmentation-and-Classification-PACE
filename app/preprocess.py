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
        Preprocessed image as torch.Tensor of shape (1, 1, H, W)
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    elif len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    if clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
    
    # Resize image
    image = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Normalize to [0, 1]
    if normalize:
        image = image.astype(np.float32) / 255.0
    
    # Add channel and batch dimensions
    image = np.expand_dims(image, axis=0)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    return torch.from_numpy(image.astype(np.float32))


def preprocess_classification(
    image: Union[str, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """
    Preprocess image for classification task using standard ImageNet normalization.
    
    Args:
        image: Input image path or numpy array (H, W, C) in BGR format
        target_size: Target size as (height, width)
        normalize: Whether to apply ImageNet normalization
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Preprocessed image as torch.Tensor of shape (1, 3, H, W)
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 3:  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    image = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Convert to float32 and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    if normalize:
        image = (image - np.array(mean)) / np.array(std)
    
    # Convert to CHW format and add batch dimension
    image = image.transpose(2, 0, 1)  # HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    return torch.from_numpy(image.astype(np.float32))
