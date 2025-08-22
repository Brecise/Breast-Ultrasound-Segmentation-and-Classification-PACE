import torch
import numpy as np
from typing import Dict, Any, Tuple, Union


def postprocess_segmentation(
    seg_output: torch.Tensor,
    threshold: float = 0.5,
    min_region_size: int = 50
) -> np.ndarray:
    """
    Postprocess segmentation output to create clean binary masks.
    
    Args:
        seg_output: Raw model output tensor of shape (1, 1, H, W)
        threshold: Probability threshold for binary classification
        min_region_size: Minimum region size in pixels (smaller regions will be removed)
        
    Returns:
        Binary mask as numpy array of shape (H, W) with values in {0, 255}
    """
    # Convert to numpy and squeeze batch and channel dimensions
    seg_np = seg_output.squeeze().cpu().numpy()
    
    # Apply threshold
    binary_mask = (seg_np > threshold).astype(np.uint8) * 255
    
    # Remove small regions using connected components
    if min_region_size > 0:
        from skimage import measure, morphology
        
        # Label connected components
        labeled = measure.label(binary_mask > 0)
        
        # Calculate region properties
        regions = measure.regionprops(labeled)
        
        # Create mask of regions to keep
        mask = np.zeros_like(binary_mask, dtype=bool)
        for region in regions:
            if region.area >= min_region_size:
                mask[labeled == region.label] = True
        
        binary_mask = (mask * 255).astype(np.uint8)
    
    return binary_mask


def postprocess_classification(
    cls_output: torch.Tensor,
    class_names: Tuple[str, ...] = ('Normal', 'Benign', 'Malignant'),
    threshold: float = 0.5
) -> Dict[str, Union[str, float]]:
    """
    Postprocess classification output to get predicted class and confidence.
    
    Args:
        cls_output: Raw model output tensor of shape (1, num_classes)
        class_names: Tuple of class names in order of model output
        threshold: Minimum confidence threshold for positive prediction
        
    Returns:
        Dictionary containing:
            - 'class': Predicted class name
            - 'confidence': Confidence score [0, 1]
            - 'is_abnormal': Whether the prediction is abnormal (not 'Normal')
    """
    # Apply softmax to get probabilities
    probs = torch.softmax(cls_output, dim=1).squeeze().cpu().numpy()
    
    # Get predicted class and confidence
    pred_idx = np.argmax(probs)
    confidence = float(probs[pred_idx])
    
    # Get class name
    class_name = class_names[pred_idx]
    
    # Determine if abnormal (not 'Normal' class)
    is_abnormal = class_name != 'Normal' and confidence >= threshold
    
    return {
        'class': class_name,
        'confidence': confidence,
        'is_abnormal': is_abnormal
    }
