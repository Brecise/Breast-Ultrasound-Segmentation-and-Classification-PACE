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

from app.model import MultiTaskModel
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
) -> MultiTaskModel:
    """
    Load the trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on ('cuda' or 'cpu')
        num_classes: Number of output classes for classification
        
    Returns:
        Loaded PyTorch model
    """
    # Initialize model
    model = MultiTaskModel(num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
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
    if task in ['seg', 'both']:
        seg_input = preprocess_segmentation(str(image_path))
        seg_input = seg_input.to(device)
    
    if task in ['cls', 'both']:
        cls_input = preprocess_classification(str(image_path))
        cls_input = cls_input.to(device)
    
    # Run inference
    with torch.no_grad():
        if task == 'seg':
            seg_output = model(seg_input)['segmentation']
            mask = postprocess_segmentation(seg_output)
            return {'segmentation': mask}
            
        elif task == 'cls':
            cls_output = model(cls_input)['classification']
            result = postprocess_classification(cls_output, CLASS_NAMES)
            return {'classification': result}
            
        else:  # both
            seg_output = model(seg_input)['segmentation']
            cls_output = model(cls_input)['classification']
            
            mask = postprocess_segmentation(seg_output)
            cls_result = postprocess_classification(cls_output, CLASS_NAMES)
            
            return {
                'segmentation': mask,
                'classification': cls_result
            }


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
    
    # Save segmentation mask if available
    if 'segmentation' in results:
        seg_dir = output_dir / 'segmentation'
        seg_dir.mkdir(parents=True, exist_ok=True)
        
        mask = results['segmentation']
        mask_path = seg_dir / f"{image_id}_mask.png"
        cv2.imwrite(str(mask_path), mask)
    
    # Save classification results if available
    if 'classification' in results:
        cls_dir = output_dir / 'classification'
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        cls_result = results['classification']
        cls_path = cls_dir / 'predictions.csv'
        
        # Check if CSV exists to append or create new
        if cls_path.exists():
            import pandas as pd
            df = pd.read_csv(cls_path)
            new_row = {
                'image_id': image_id,
                'label': cls_result['class'],
                'confidence': cls_result['confidence'],
                'is_abnormal': cls_result['is_abnormal']
            }
            df = df.append(new_row, ignore_index=True)
            df.to_csv(cls_path, index=False)
        else:
            import pandas as pd
            df = pd.DataFrame([{
                'image_id': image_id,
                'label': cls_result['class'],
                'confidence': cls_result['confidence'],
                'is_abnormal': cls_result['is_abnormal']
            }])
            df.to_csv(cls_path, index=False)


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
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get all image files in the input directory
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_path.glob(f'*{ext}'))
        image_paths.extend(input_path.glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_paths)} images in {input_dir}")
    
    # Process each image
    for img_path in image_paths:
        try:
            print(f"Processing {img_path.name}...")
            
            # Get image ID (filename without extension)
            img_id = img_path.stem
            
            # Process the image
            results = process_image(
                model=model,
                image_path=img_path,
                device=device,
                task=task
            )
            
            # Save the results
            save_results(results, output_path, img_id)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='PACE 2025 - Breast Cancer Detection')
    
    # Required arguments
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input directory containing images')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path to output directory')
    
    # Optional arguments
    parser.add_argument('-m', '--model', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'Path to model checkpoint (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('-t', '--task', type=str, default='both',
                        choices=['seg', 'cls', 'both'],
                        help='Task to perform: seg for segmentation, cls for classification, both (default)')
    parser.add_argument('-d', '--device', type=str, default=DEFAULT_DEVICE,
                        choices=['cuda', 'cpu'],
                        help=f'Device to use for inference (default: {DEFAULT_DEVICE})')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        device = 'cpu'
    
    # Check if input directory exists
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Check if model file exists
    if not os.path.isfile(args.model):
        print(f"Error: Model file '{args.model}' does not exist.")
        sys.exit(1)
    
    try:
        # Load the model
        print(f"Loading model from {args.model}...")
        model = load_model(args.model, device)
        print(f"Model loaded successfully on device: {device}")
        
        # Process the input directory
        print(f"Processing images in {args.input}...")
        process_directory(
            model=model,
            input_dir=args.input,
            output_dir=args.output,
            device=device,
            task=args.task
        )
        
        print(f"\nProcessing complete. Results saved to {args.output}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()