"""
Models package for Vision-Language Segmentation.

This package contains:
- VisionEncoder: ResNet-based image encoder
- TextEncoder: PubMedBERT-based text encoder with caption generation
- CrossAttentionFusion: Vision-language fusion module
- SegmentationDecoder: U-Net style decoder with MC Dropout
- VLSegmentationModel: Main model combining all components
- UncertaintyEstimator: MC Dropout based uncertainty estimation
"""

from .vision_encoder import VisionEncoder, ResNetEncoder
from .text_encoder import TextEncoder, CaptionGenerator
from .fusion import (
    CrossAttentionFusion,
    FiLMFusion,
    ConcatFusion,
    MultiScaleFusion,
)
from .decoder import (
    SegmentationDecoder,
    DecoderBlock,
    ConvBlock,
    UncertaintyEstimator,
)
from .vl_segmentation import (
    VLSegmentationModel,
    VisionOnlyModel,
    create_model,
)

__all__ = [
    # Vision
    "VisionEncoder",
    "ResNetEncoder",
    # Text
    "TextEncoder",
    "CaptionGenerator",
    # Fusion
    "CrossAttentionFusion",
    "FiLMFusion",
    "ConcatFusion",
    "MultiScaleFusion",
    # Decoder
    "SegmentationDecoder",
    "DecoderBlock",
    "ConvBlock",
    "UncertaintyEstimator",
    # Main models
    "VLSegmentationModel",
    "VisionOnlyModel",
    "create_model",
]