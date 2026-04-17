"""
Vision-Language Segmentation Model with Uncertainty Estimation.

This is the main model that combines:
- Vision Encoder (ResNet)
- Text Encoder (PubMedBERT)
- Cross-Attention Fusion
- U-Net Decoder with MC Dropout
- Uncertainty Estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

from .vision_encoder import VisionEncoder
from .text_encoder import TextEncoder, CaptionGenerator
from .fusion import CrossAttentionFusion, FiLMFusion, MultiScaleFusion
from .decoder import SegmentationDecoder, UncertaintyEstimator


class VLSegmentationModel(nn.Module):
    """
    Vision-Language Segmentation Model.
    
    Combines vision and text encoders with cross-attention fusion
    for text-guided medical image segmentation with uncertainty estimation.
    """
    
    def __init__(
        self,
        # Vision encoder config
        vision_encoder_name: str = "resnet34",
        vision_pretrained: bool = True,
        # Text encoder config
        text_encoder_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        text_max_length: int = 128,
        freeze_text_encoder: bool = True,
        # Fusion config
        fusion_type: str = "cross_attention",
        feature_dim: int = 256,
        num_attention_heads: int = 8,
        fusion_dropout: float = 0.1,
        # Decoder config
        decoder_channels: List[int] = None,
        num_classes: int = 1,
        # Uncertainty config
        mc_dropout_rate: float = 0.1,
        use_mc_dropout: bool = True,
        num_mc_samples: int = 10,
        # Mode
        use_text: bool = True,
    ):
        """
        Initialize VL Segmentation Model.
        
        Args:
            vision_encoder_name: Name of vision backbone
            vision_pretrained: Whether to use pretrained vision weights
            text_encoder_name: HuggingFace model name for text encoder
            text_max_length: Maximum text sequence length
            freeze_text_encoder: Whether to freeze text encoder
            fusion_type: Type of fusion ('cross_attention', 'film', 'concat')
            feature_dim: Feature dimension for fusion
            num_attention_heads: Number of attention heads
            fusion_dropout: Dropout rate in fusion module
            decoder_channels: Decoder channel dimensions
            num_classes: Number of output classes
            mc_dropout_rate: Dropout rate for uncertainty estimation
            use_mc_dropout: Whether to use MC dropout
            num_mc_samples: Number of MC samples for uncertainty
            use_text: Whether to use text conditioning (False for vision-only baseline)
        """
        super().__init__()
        
        self.use_text = use_text
        self.use_mc_dropout = use_mc_dropout
        self.num_mc_samples = num_mc_samples
        self.feature_dim = feature_dim
        
        # Default decoder channels
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]
        
        # ============ Vision Encoder ============
        self.vision_encoder = VisionEncoder(
            encoder_name=vision_encoder_name,
            pretrained=vision_pretrained,
            feature_dim=feature_dim,
        )
        
        # Get encoder channel info
        self.skip_channels = self.vision_encoder.get_skip_channels()
        self.bottleneck_channels = self.vision_encoder.get_bottleneck_channels()
        
        # ============ Text Encoder ============
        if use_text:
            self.text_encoder = TextEncoder(
                encoder_name=text_encoder_name,
                projection_dim=feature_dim,
                max_length=text_max_length,
                freeze_encoder=freeze_text_encoder,
            )
            
            # ============ Fusion Module ============
            self.fusion = CrossAttentionFusion(
                vision_dim=feature_dim,
                text_dim=feature_dim,
                num_heads=num_attention_heads,
                dropout=fusion_dropout,
            )
        else:
            self.text_encoder = None
            self.fusion = None
        
        # ============ Decoder ============
        # Adjust skip channels order: [stage3, stage2, stage1, stage0]
        decoder_skip_channels = self.skip_channels[::-1]
        
        self.decoder = SegmentationDecoder(
            bottleneck_channels=self.bottleneck_channels,
            skip_channels=decoder_skip_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            dropout_rate=mc_dropout_rate,
            use_mc_dropout=use_mc_dropout,
        )
        
        # ============ Uncertainty Estimator ============
        self.uncertainty_estimator = UncertaintyEstimator(
            num_mc_samples=num_mc_samples,
            uncertainty_type='both',
        )
    
    def encode_image(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode images using vision encoder.
        
        Args:
            images: Input images [B, 3, H, W]
            
        Returns:
            Dictionary with 'bottleneck' and 'skip_features'
        """
        return self.vision_encoder(images)
    
    def encode_text(
        self,
        texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Encode text using text encoder.
        
        Args:
            texts: List of caption strings
            input_ids: Pre-tokenized input IDs
            attention_mask: Attention mask
            
        Returns:
            Dictionary with 'sequence', 'pooled', 'attention_mask'
            or None if use_text is False
        """
        if not self.use_text or self.text_encoder is None:
            return None
        
        return self.text_encoder(
            texts=texts,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    
    def fuse_features(
        self,
        vision_features: torch.Tensor,
        text_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Fuse vision and text features.
        
        Args:
            vision_features: Bottleneck vision features [B, D, H, W]
            text_features: Text encoder outputs (or None)
            
        Returns:
            Fused features [B, D, H, W]
        """
        if not self.use_text or text_features is None or self.fusion is None:
            return vision_features
        
        return self.fusion(
            vision_features=vision_features,
            text_features=text_features['sequence'],
            text_mask=text_features['attention_mask'],
        )
    
    def forward(
        self,
        images: torch.Tensor,
        texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            images: Input images [B, 3, H, W]
            texts: List of caption strings (optional)
            input_ids: Pre-tokenized input IDs (optional)
            attention_mask: Attention mask (optional)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
            - 'logits': Segmentation logits [B, 1, H, W]
            - 'probs': Segmentation probabilities [B, 1, H, W]
            - (optional) 'vision_features', 'text_features', 'fused_features'
        """
        # Encode image
        vision_outputs = self.encode_image(images)
        bottleneck = vision_outputs['bottleneck']
        skip_features = vision_outputs['skip_features']
        
        # Encode text (if using text)
        text_features = None
        if self.use_text and (texts is not None or input_ids is not None):
            text_features = self.encode_text(
                texts=texts,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        # Fuse features
        fused_features = self.fuse_features(bottleneck, text_features)
        
        # Decode
        logits = self.decoder(fused_features, skip_features)
        
        # Build output
        outputs = {
            'logits': logits,
            'probs': torch.sigmoid(logits),
        }
        
        if return_features:
            outputs['vision_features'] = bottleneck
            outputs['text_features'] = text_features
            outputs['fused_features'] = fused_features
        
        return outputs
    
    def predict_with_uncertainty(
        self,
        images: torch.Tensor,
        texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty estimation using MC Dropout.
        
        Args:
            images: Input images [B, 3, H, W]
            texts: List of caption strings
            input_ids: Pre-tokenized input IDs
            attention_mask: Attention mask
            
        Returns:
            Dictionary containing:
            - 'prediction': Mean prediction [B, 1, H, W]
            - 'entropy': Predictive entropy [B, 1, H, W]
            - 'variance': Predictive variance [B, 1, H, W]
            - 'samples': MC samples [T, B, 1, H, W]
        """
        # Prepare forward kwargs
        forward_kwargs = {}
        if texts is not None:
            forward_kwargs['texts'] = texts
        if input_ids is not None:
            forward_kwargs['input_ids'] = input_ids
        if attention_mask is not None:
            forward_kwargs['attention_mask'] = attention_mask
        
        # Create wrapper that returns just probabilities
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, images, **kwargs):
                outputs = self.model(images, **kwargs)
                return outputs['logits']
            
            def enable_mc_dropout(self):
                self.model.decoder.enable_mc_dropout()
            
            def disable_mc_dropout(self):
                self.model.decoder.disable_mc_dropout()
        
        wrapper = ModelWrapper(self)
        
        return self.uncertainty_estimator.estimate_uncertainty(
            wrapper, images, **forward_kwargs
        )
    
    def enable_mc_dropout(self):
        """Enable MC dropout for uncertainty estimation."""
        self.decoder.enable_mc_dropout()
    
    def disable_mc_dropout(self):
        """Disable MC dropout."""
        self.decoder.disable_mc_dropout()
    
    def get_trainable_parameters(self) -> List[Dict]:
        """
        Get parameter groups for optimizer with different learning rates.
        
        Returns:
            List of parameter dictionaries for optimizer
        """
        param_groups = []
        
        # Vision encoder (lower learning rate if pretrained)
        param_groups.append({
            'params': self.vision_encoder.parameters(),
            'lr_scale': 0.1,  # Lower LR for pretrained backbone
        })
        
        # Text encoder (if trainable)
        if self.use_text and self.text_encoder is not None:
            # Only add parameters that require grad
            text_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
            if text_params:
                param_groups.append({
                    'params': text_params,
                    'lr_scale': 0.1,
                })
        
        # Fusion module (full learning rate)
        if self.fusion is not None:
            param_groups.append({
                'params': self.fusion.parameters(),
                'lr_scale': 1.0,
            })
        
        # Decoder (full learning rate)
        param_groups.append({
            'params': self.decoder.parameters(),
            'lr_scale': 1.0,
        })
        
        return param_groups


class VisionOnlyModel(nn.Module):
    """
    Vision-only baseline model (no text conditioning).
    
    Simplified version of VLSegmentationModel without text components.
    Used for baseline comparison.
    """
    
    def __init__(
        self,
        vision_encoder_name: str = "resnet34",
        vision_pretrained: bool = True,
        feature_dim: int = 256,
        decoder_channels: List[int] = None,
        num_classes: int = 1,
        mc_dropout_rate: float = 0.1,
        use_mc_dropout: bool = True,
        num_mc_samples: int = 10,
    ):
        """Initialize vision-only model."""
        super().__init__()
        
        self.use_mc_dropout = use_mc_dropout
        self.num_mc_samples = num_mc_samples
        
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(
            encoder_name=vision_encoder_name,
            pretrained=vision_pretrained,
            feature_dim=feature_dim,
        )
        
        skip_channels = self.vision_encoder.get_skip_channels()
        bottleneck_channels = self.vision_encoder.get_bottleneck_channels()
        
        # Decoder
        self.decoder = SegmentationDecoder(
            bottleneck_channels=bottleneck_channels,
            skip_channels=skip_channels[::-1],
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            dropout_rate=mc_dropout_rate,
            use_mc_dropout=use_mc_dropout,
        )
        
        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(
            num_mc_samples=num_mc_samples,
            uncertainty_type='both',
        )
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        vision_outputs = self.vision_encoder(images)
        logits = self.decoder(
            vision_outputs['bottleneck'],
            vision_outputs['skip_features']
        )
        
        return {
            'logits': logits,
            'probs': torch.sigmoid(logits),
        }
    
    def predict_with_uncertainty(
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict with uncertainty estimation using MC Dropout."""
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, images, **kwargs):
                outputs = self.model(images)
                return outputs['logits']

            def enable_mc_dropout(self):
                self.model.decoder.enable_mc_dropout()

            def disable_mc_dropout(self):
                self.model.decoder.disable_mc_dropout()

        wrapper = ModelWrapper(self)
        return self.uncertainty_estimator.estimate_uncertainty(wrapper, images)

    def enable_mc_dropout(self):
        self.decoder.enable_mc_dropout()
    
    def disable_mc_dropout(self):
        self.decoder.disable_mc_dropout()


def create_model(config) -> nn.Module:
    """
    Factory function to create model based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Initialized model
    """
    if config.experiment.vision_only or not config.experiment.use_text:
        return VisionOnlyModel(
            vision_encoder_name=config.vision.encoder_name,
            vision_pretrained=config.vision.pretrained,
            feature_dim=config.vision.feature_dim,
            decoder_channels=config.decoder.decoder_channels,
            num_classes=1,
            mc_dropout_rate=config.decoder.dropout_rate,
            use_mc_dropout=config.decoder.mc_dropout,
            num_mc_samples=config.uncertainty.num_mc_samples,
        )
    else:
        return VLSegmentationModel(
            vision_encoder_name=config.vision.encoder_name,
            vision_pretrained=config.vision.pretrained,
            text_encoder_name=config.text.encoder_name,
            text_max_length=config.text.max_length,
            freeze_text_encoder=config.text.freeze_encoder,
            fusion_type=config.fusion.fusion_type,
            feature_dim=config.vision.feature_dim,
            num_attention_heads=config.fusion.num_heads,
            fusion_dropout=config.fusion.dropout,
            decoder_channels=config.decoder.decoder_channels,
            num_classes=1,
            mc_dropout_rate=config.decoder.dropout_rate,
            use_mc_dropout=config.decoder.mc_dropout,
            num_mc_samples=config.uncertainty.num_mc_samples,
            use_text=config.experiment.use_text,
        )


# For testing
if __name__ == "__main__":
    print("Testing VLSegmentationModel...")
    
    # Create model (without loading actual text encoder for quick test)
    model = VLSegmentationModel(
        vision_encoder_name="resnet34",
        vision_pretrained=False,
        text_encoder_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        use_text=False,  # Disable text for quick test
    )
    
    # Test forward pass
    images = torch.randn(2, 3, 256, 256)
    outputs = model(images)
    
    print(f"Input shape: {images.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Output probs shape: {outputs['probs'].shape}")
    
    # Test vision-only model
    print("\nTesting VisionOnlyModel...")
    vision_model = VisionOnlyModel(
        vision_encoder_name="resnet34",
        vision_pretrained=False,
    )
    
    outputs = vision_model(images)
    print(f"Output logits shape: {outputs['logits'].shape}")
    
    print("\nAll tests passed!")