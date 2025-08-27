# /breast-cancer-api/app/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple, Optional, Union
import segmentation_models_pytorch as smp

class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    def __init__(self, mode='binary'):
        super(DiceLoss, self).__init__()
        self.mode = mode
        
    def forward(self, pred, target):
        smooth = 1.0
        if self.mode == 'binary':
            pred = torch.sigmoid(pred)
        
        # Flatten predictions and targets
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice

class MultiTaskModel(nn.Module):
    """
    Multi-task model for breast cancer segmentation and classification.
    Uses a shared encoder with task-specific decoders.
    """
    
    def __init__(
        self,
        num_classes: int = 3,  # Normal, Benign, Malignant
        encoder_name: str = 'resnet34',
        pretrained: bool = True
    ):
        """
        Initialize the multi-task model.
        
        Args:
            num_classes: Number of classification classes
            encoder_name: Backbone architecture name (resnet18, resnet34, efficientnet-b0, etc.)
            pretrained: Whether to use pretrained weights for the encoder
        """
        super().__init__()
        self.num_classes = num_classes
        self.encoder_name = encoder_name
        
        # Initialize encoder
        self.encoder = self._create_encoder(encoder_name, pretrained)
        
        # Segmentation decoder (simplified U-Net like decoder)
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_encoder(self, encoder_name: str, pretrained: bool = True) -> nn.Module:
        """Create the encoder backbone."""
        if 'resnet' in encoder_name:
            if encoder_name == 'resnet18':
                encoder = models.resnet18(pretrained=pretrained)
                # Remove the fully connected layer
                modules = list(encoder.children())[:-2]
                return nn.Sequential(*modules)
            elif encoder_name == 'resnet34':
                encoder = models.resnet34(pretrained=pretrained)
                # Remove the fully connected layer
                modules = list(encoder.children())[:-2]
                return nn.Sequential(*modules)
        
        # Default to ResNet34 if specified encoder is not supported
        encoder = models.resnet34(pretrained=pretrained)
        return nn.Sequential(*list(encoder.children())[:-2])
    
    def _initialize_weights(self):
        """Initialize weights for the decoder and classifier."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the multi-task model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Dictionary containing:
                - 'segmentation': Segmentation output (batch_size, 1, H, W)
                - 'classification': Classification logits (batch_size, num_classes)
        """
        # Shared encoder
        features = self.encoder(x)
        
        # Segmentation decoder
        seg_output = self.seg_decoder(features)
        
        # Classification head
        cls_output = self.classifier(features)
        
        return {
            'segmentation': seg_output,
            'classification': cls_output
        }
    
    def predict(
        self, 
        x: torch.Tensor,
        task: str = 'both',
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with the model.
        
        Args:
            x: Input tensor or batch of tensors
            task: Task to perform ('seg', 'cls', or 'both')
            threshold: Threshold for segmentation
            
        Returns:
            Dictionary containing the requested predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            result = {}
            
            if task in ['seg', 'both']:
                result['segmentation'] = (outputs['segmentation'] > threshold).float()
                
            if task in ['cls', 'both']:
                result['classification'] = F.softmax(outputs['classification'], dim=1)
            
            return result

class MultiTaskUnet(nn.Module):
    """
    A U-Net based model with two heads:
    1. Segmentation Head: Outputs a pixel-wise mask (using the U-Net++ decoder).
    2. Classification Head: Outputs a class prediction (benign, malignant, normal).
    """
    def __init__(self, encoder_name, encoder_weights, in_channels, num_classes):
        super(MultiTaskUnet, self).__init__()

        # --- Segmentation Branch ---
        self.segmentation_model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,  # Binary segmentation
            activation=None
        )

        # --- Classification Branch ---
        # Get the number of output channels from the encoder's last stage
        encoder_out_channels = self.segmentation_model.encoder.out_channels[-1]
        
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(encoder_out_channels, num_classes)
        )

    def forward(self, images, masks=None, labels=None):
        # --- Segmentation Path ---
        segmentation_logits = self.segmentation_model(images)

        # --- Classification Path ---
        encoder_features = self.segmentation_model.encoder(images)[-1]
        classification_logits = self.classification_head(encoder_features)

        # --- Loss Calculation (during training) ---
        if masks is not None and labels is not None:
            seg_loss_dice = DiceLoss(mode='binary')(segmentation_logits, masks)
            seg_loss_bce = nn.BCEWithLogitsLoss()(segmentation_logits, masks)
            segmentation_loss = seg_loss_dice + seg_loss_bce

            classification_loss = nn.CrossEntropyLoss()(classification_logits, labels)
            total_loss = segmentation_loss + classification_loss

            return segmentation_logits, classification_logits, total_loss

        # During inference, just return the predictions
        return segmentation_logits, classification_logits