#!/usr/bin/env python3
"""
PACE 2025 - Breast Cancer Detection System
Main inference script for segmentation and classification of breast ultrasound images.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import numpy as np
from PIL import Image
import cv2

# Add the parent directory to path to allow importing from app
sys.path.append(str(Path(__file__).parent.parent))

from app.model import MultiTaskUnet
from app.preprocess import preprocess_segmentation, preprocess_classification
from app.postprocess import postprocess_segmentation, postprocess_classification

# Constants
DEFAULT_MODEL_PATH = "app/checkpoints/best_model.pt"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ("Normal", "Benign", "Malignant")


def load_model(
    model_path: Union[str, Path], 
    device: str = DEFAULT_DEVICE,
    num_classes: int = 3
) -> MultiTaskUnet:
    """
    Load the trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on ('cuda' or 'cpu')
        num_classes: Number of output classes for classification
        
    Returns:
        Loaded PyTorch model
    """
    # Initialize model with the correct architecture
    model = MultiTaskUnet(
        encoder_name='timm-efficientnet-b0',
        encoder_weights=None,  # We'll load our own weights
        in_channels=3,
        num_classes=num_classes
    )
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Move model to the specified device
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Model device: {next(model.parameters()).device}")
    
    return model


def process_image(
    model: torch.nn.Module,
    image_path: Union[str, Path],
    device: str = DEFAULT_DEVICE,
    task: str = 'both'
) -> Dict[str, any]:
    """
    Process a single image with the model.
    
    Args:
        model: Loaded PyTorch model
        image_path: Path to the input image
        device: Device to run inference on
        task: Task to perform ('seg', 'cls', or 'both')
        
    Returns:
        Dictionary containing the model outputs
    """
    # Read and preprocess the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess for the model
    image_tensor = preprocess_segmentation(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        if task == 'seg':
            seg_logits, _ = model(image_tensor)
            cls_logits = None
        elif task == 'cls':
            _, cls_logits = model(image_tensor)
            seg_logits = None
        else:  # both
            seg_logits, cls_logits = model(image_tensor)
    
    # Postprocess results
    results = {}
    
    if seg_logits is not None:
        results['segmentation'] = postprocess_segmentation(seg_logits)
    
    if cls_logits is not None:
        probs = torch.softmax(cls_logits, dim=1)
        results['classification'] = {
            'probabilities': probs.cpu().numpy()[0],
            'predicted_class': torch.argmax(probs, dim=1).item(),
            'class_names': CLASS_NAMES
        }
    
    return results


def save_results(
    results: Dict[str, any],
    output_dir: Union[str, Path],
    image_id: str
) -> None:
    """
    Save the model outputs to disk.
    
    Args:
        results: Dictionary containing model outputs
        output_dir: Directory to save outputs
        image_id: Base name of the input image (without extension)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save segmentation mask if available
    if 'segmentation' in results:
        mask = results['segmentation']
        mask_path = output_dir / f"{image_id}_mask.png"
        cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
    
    # Save classification results if available
    if 'classification' in results:
        cls = results['classification']
        cls_path = output_dir / f"{image_id}_cls.json"
        with open(cls_path, 'w') as f:
            json.dump({
                'probabilities': cls['probabilities'].tolist(),
                'predicted_class': int(cls['predicted_class']),
                'predicted_label': CLASS_NAMES[cls['predicted_class']],
                'class_names': list(CLASS_NAMES)
            }, f, indent=2)


def process_directory(
    model: torch.nn.Module,
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    device: str = DEFAULT_DEVICE,
    task: str = 'both'
) -> None:
    """
    Process all images in a directory.
    
    Args:
        model: Loaded PyTorch model
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
        device: Device to run inference on
        task: Task to perform ('seg', 'cls', or 'both')
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_exts]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Processing {len(image_files)} images from {input_dir}")
    
    for img_path in image_files:
        try:
            image_id = img_path.stem
            print(f"Processing {image_id}...")
            
            # Process the image
            results = process_image(model, img_path, device, task)
            
            # Save the results
            save_results(results, output_dir, image_id)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Breast Cancer Detection Inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save outputs (default: output/)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'Path to model checkpoint (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE,
                        help=f'Device to run inference on (default: {DEFAULT_DEVICE})')
    parser.add_argument('--task', type=str, default='both',
                        choices=['seg', 'cls', 'both'],
                        help='Task to perform: segmentation, classification, or both (default: both)')
    
    args = parser.parse_args()
    
    # Load the model
    try:
        model = load_model(args.model, args.device)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)
    
    # Check if input is a file or directory
    input_path = Path(args.input)
    if input_path.is_file():
        # Process single image
        try:
            results = process_image(model, input_path, args.device, args.task)
            save_results(results, args.output, input_path.stem)
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            sys.exit(1)
    elif input_path.is_dir():
        # Process directory
        process_directory(model, input_path, args.output, args.device, args.task)
        print(f"All results saved to {args.output}")
    else:
        print(f"Input path does not exist: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()