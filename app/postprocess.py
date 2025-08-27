import torch
import numpy as np
from typing import Dict, Any, Tuple, Union, Optional


def postprocess_segmentation(
    seg_output: torch.Tensor,
    threshold: float = 0.5,
    min_region_size: int = 50,
    return_probs: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Postprocess segmentation output to create clean binary masks.
    
    Args:
        seg_output: Raw model output tensor of shape (1, 1, H, W) or (1, H, W)
        threshold: Probability threshold for binary classification (0-1)
        min_region_size: Minimum region size in pixels (smaller regions will be removed)
        return_probs: If True, returns both binary mask and probability map
        
    Returns:
        Binary mask as numpy array of shape (H, W) with values in {0, 255}
        If return_probs is True, also returns the probability map (H, W) in [0, 1]
    """
    # Ensure we're working with numpy
    if torch.is_tensor(seg_output):
        seg_output = seg_output.detach().cpu().numpy()
    
    # Remove batch and channel dimensions if they exist
    while len(seg_output.shape) > 2:
        seg_output = seg_output.squeeze(0)
    
    # Get probability map (apply sigmoid if logits)
    if seg_output.max() > 1.0 or seg_output.min() < 0.0:
        prob_map = 1 / (1 + np.exp(-seg_output))  # Sigmoid
    else:
        prob_map = seg_output
    
    # Apply threshold to get binary mask
    binary_mask = (prob_map > threshold).astype(np.uint8) * 255
    
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
    
    if return_probs:
        return binary_mask, prob_map
    return binary_mask


def postprocess_classification(
    cls_output: torch.Tensor,
    class_names: Optional[Tuple[str, ...]] = None,
    return_logits: bool = False
) -> Dict[str, Any]:
    """
    Postprocess classification output to get class probabilities and predictions.
    
    Args:
        cls_output: Raw model output tensor of shape (1, num_classes) or (num_classes,)
        class_names: Optional tuple of class names
        return_logits: If True, include raw logits in the output
        
    Returns:
        Dictionary containing:
        - 'probabilities': List of class probabilities
        - 'predicted_class': Index of predicted class
        - 'predicted_label': Name of predicted class (if class_names provided)
        - 'class_names': List of class names (if provided)
        - 'logits': Raw logits (if return_logits=True)
    """
    # Ensure we're working with numpy
    if torch.is_tensor(cls_output):
        cls_output = cls_output.detach().cpu().numpy()
    
    # Remove batch dimension if present
    if len(cls_output.shape) > 1:
        cls_output = cls_output.squeeze(0)
    
    # Get probabilities (softmax if logits)
    if cls_output.min() < 0 or cls_output.max() > 1.0:
        # Apply softmax to get probabilities
        exp_scores = np.exp(cls_output - np.max(cls_output))  # Numerical stability
        probs = exp_scores / exp_scores.sum()
    else:
        probs = cls_output
    
    # Get predicted class
    predicted_class = int(np.argmax(probs))
    
    # Prepare result
    result = {
        'probabilities': probs.tolist(),
        'predicted_class': predicted_class,
    }
    
    # Add class names if provided
    if class_names is not None:
        if len(class_names) != len(probs):
            raise ValueError(f"Number of class names ({len(class_names)}) "
                           f"does not match number of classes ({len(probs)})")
        result['predicted_label'] = class_names[predicted_class]
        result['class_names'] = list(class_names)
    
    # Add logits if requested
    if return_logits:
        result['logits'] = cls_output.tolist()
    
    return result


def combine_results(
    seg_output: Optional[torch.Tensor] = None,
    cls_output: Optional[torch.Tensor] = None,
    class_names: Optional[Tuple[str, ...]] = None,
    seg_threshold: float = 0.5,
    min_region_size: int = 50
) -> Dict[str, Any]:
    """
    Combine segmentation and classification results into a single dictionary.
    
    Args:
        seg_output: Raw segmentation output tensor
        cls_output: Raw classification output tensor
        class_names: Optional tuple of class names
        seg_threshold: Threshold for segmentation binarization
        min_region_size: Minimum region size for segmentation postprocessing
        
    Returns:
        Dictionary containing combined results
    """
    result = {}
    
    # Process segmentation if provided
    if seg_output is not None:
        binary_mask, prob_map = postprocess_segmentation(
            seg_output,
            threshold=seg_threshold,
            min_region_size=min_region_size,
            return_probs=True
        )
        result['segmentation'] = {
            'binary_mask': binary_mask,
            'probability_map': prob_map,
            'threshold': seg_threshold
        }
    
    # Process classification if provided
    if cls_output is not None:
        cls_result = postprocess_classification(
            cls_output,
            class_names=class_names,
            return_logits=False
        )
        result['classification'] = cls_result
    
    return result
