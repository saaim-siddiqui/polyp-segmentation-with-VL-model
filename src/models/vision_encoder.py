"""
Vision Encoder Module.

This module handles visual feature extraction using pretrained CNN backbones
(ResNet, EfficientNet) with support for multi-scale feature extraction.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict, Optional
from collections import OrderedDict


class ResNetEncoder(nn.Module):
    """
    ResNet-based encoder for feature extraction.
    
    Extracts multi-scale features from different stages of ResNet
    for use in U-Net style decoder with skip connections.
    """
    
    # Channel dimensions for different ResNet variants
    CHANNEL_CONFIGS = {
        "resnet18": [64, 64, 128, 256, 512],
        "resnet34": [64, 64, 128, 256, 512],
        "resnet50": [64, 256, 512, 1024, 2048],
        "resnet101": [64, 256, 512, 1024, 2048],
    }
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        freeze_bn: bool = False,
    ):
        """
        Initialize ResNet encoder.
        
        Args:
            encoder_name: Name of ResNet variant
            pretrained: Whether to use pretrained ImageNet weights
            freeze_bn: Whether to freeze batch normalization layers
        """
        super().__init__()
        
        self.encoder_name = encoder_name
        self.channels = self.CHANNEL_CONFIGS[encoder_name]
        
        # Load pretrained ResNet
        if encoder_name == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif encoder_name == "resnet34":
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        elif encoder_name == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        elif encoder_name == "resnet101":
            resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")
        
        # Extract layers for multi-scale features
        # Stage 0: conv1 + bn1 + relu (stride 2) -> H/2
        self.stage0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        
        # Pooling layer (stride 2) -> H/4
        self.pool = resnet.maxpool
        
        # Stage 1-4: ResNet blocks
        self.stage1 = resnet.layer1  # H/4
        self.stage2 = resnet.layer2  # H/8
        self.stage3 = resnet.layer3  # H/16
        self.stage4 = resnet.layer4  # H/32
        
        # Freeze batch norm if specified
        if freeze_bn:
            self._freeze_bn()
    
    def _freeze_bn(self):
        """Freeze batch normalization layers."""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input image tensor [B, 3, H, W]
            
        Returns:
            Dictionary of multi-scale features:
            - 'stage0': [B, C0, H/2, W/2]
            - 'stage1': [B, C1, H/4, W/4]
            - 'stage2': [B, C2, H/8, W/8]
            - 'stage3': [B, C3, H/16, W/16]
            - 'stage4': [B, C4, H/32, W/32] (bottleneck)
        """
        features = OrderedDict()
        
        # Stage 0: Initial convolution
        x = self.stage0(x)
        features['stage0'] = x  # H/2
        
        # Pooling
        x = self.pool(x)
        
        # Stages 1-4
        x = self.stage1(x)
        features['stage1'] = x  # H/4
        
        x = self.stage2(x)
        features['stage2'] = x  # H/8
        
        x = self.stage3(x)
        features['stage3'] = x  # H/16
        
        x = self.stage4(x)
        features['stage4'] = x  # H/32 (bottleneck)
        
        return features
    
    def get_channels(self) -> List[int]:
        """Return channel dimensions for each stage."""
        return self.channels


class VisionEncoder(nn.Module):
    """
    Vision encoder wrapper with optional feature projection.
    
    Wraps the backbone encoder and adds projection layers
    to map features to the desired dimension for fusion.
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        feature_dim: int = 256,
        freeze_bn: bool = False,
    ):
        """
        Initialize vision encoder.
        
        Args:
            encoder_name: Name of backbone encoder
            pretrained: Whether to use pretrained weights
            feature_dim: Output feature dimension (for bottleneck)
            freeze_bn: Whether to freeze batch normalization
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Initialize backbone
        if encoder_name.startswith("resnet"):
            self.backbone = ResNetEncoder(
                encoder_name=encoder_name,
                pretrained=pretrained,
                freeze_bn=freeze_bn,
            )
            self.encoder_channels = self.backbone.get_channels()
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")
        
        # Bottleneck projection (stage4 -> feature_dim)
        bottleneck_channels = self.encoder_channels[-1]
        self.bottleneck_projection = nn.Sequential(
            nn.Conv2d(bottleneck_channels, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through vision encoder.
        
        Args:
            x: Input image tensor [B, 3, H, W]
            
        Returns:
            Dictionary of features:
            - 'skip_features': List of skip connection features
            - 'bottleneck': Projected bottleneck features [B, D, H/32, W/32]
        """
        # Get multi-scale features from backbone
        features = self.backbone(x)
        
        # Extract skip connection features (stages 0-3)
        skip_features = [
            features['stage0'],
            features['stage1'],
            features['stage2'],
            features['stage3'],
        ]
        
        # Project bottleneck features
        bottleneck = self.bottleneck_projection(features['stage4'])
        
        return {
            'skip_features': skip_features,
            'bottleneck': bottleneck,
            'raw_bottleneck': features['stage4'],  # For potential use
        }
    
    def get_skip_channels(self) -> List[int]:
        """Return channel dimensions for skip connections."""
        return self.encoder_channels[:-1]  # Exclude bottleneck
    
    def get_bottleneck_channels(self) -> int:
        """Return bottleneck channel dimension (after projection)."""
        return self.feature_dim


# For testing
if __name__ == "__main__":
    # Test encoder
    encoder = VisionEncoder(
        encoder_name="resnet34",
        pretrained=False,  # Set False for quick testing
        feature_dim=256,
    )
    
    # Create dummy input
    x = torch.randn(2, 3, 256, 256)
    
    # Forward pass
    outputs = encoder(x)
    
    print("Vision Encoder Output Shapes:")
    print(f"Bottleneck: {outputs['bottleneck'].shape}")
    print(f"Skip features:")
    for i, skip in enumerate(outputs['skip_features']):
        print(f"  Stage {i}: {skip.shape}")
    
    print(f"\nSkip channels: {encoder.get_skip_channels()}")
    print(f"Bottleneck channels: {encoder.get_bottleneck_channels()}")