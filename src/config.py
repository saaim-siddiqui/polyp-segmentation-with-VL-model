"""
Configuration for Vision-Language Segmentation with Uncertainty Estimation.

This file contains all hyperparameters and settings for the project.
Modify these values to experiment with different configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset configuration."""
    # Paths
    data_root: str = "./data"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Image settings
    image_size: int = 256  # Images will be resized to (image_size, image_size)
    num_workers: int = 4
    
    # Augmentation
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    rotation_limit: int = 30
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2


@dataclass
class TextConfig:
    """Text encoder configuration."""
    # Model selection
    encoder_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    # Alternative: "emilyalsentzer/Bio_ClinicalBERT"
    # Alternative: "openai/clip-vit-base-patch32" (for CLIP text encoder)
    
    # Tokenization
    max_length: int = 128
    
    # Projection
    projection_dim: int = 256  # Project text features to this dimension
    
    # Freezing
    freeze_encoder: bool = True  # Freeze text encoder weights
    unfreeze_layers: int = 0  # Number of layers to unfreeze from top (0 = all frozen)


@dataclass
class VisionConfig:
    """Vision encoder configuration."""
    # Model selection
    encoder_name: str = "resnet34"  # Options: resnet18, resnet34, resnet50, efficientnet-b0
    pretrained: bool = True
    
    # Feature dimensions (depends on encoder)
    # ResNet34: [64, 128, 256, 512] at different stages
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    
    # Output
    feature_dim: int = 256  # Bottleneck feature dimension


@dataclass
class FusionConfig:
    """Cross-attention fusion configuration."""
    # Attention settings
    num_heads: int = 8
    dropout: float = 0.1
    
    # Fusion location
    fusion_stages: List[str] = field(default_factory=lambda: ["bottleneck"])
    # Options: ["bottleneck"], ["bottleneck", "decoder_1"], etc.
    
    # Fusion type
    fusion_type: str = "cross_attention"  # Options: cross_attention, film, concat


@dataclass
class DecoderConfig:
    """Decoder configuration."""
    # Architecture
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    
    # Dropout for uncertainty estimation
    mc_dropout: bool = True
    dropout_rate: float = 0.1  # Applied during both training and inference for MC Dropout


@dataclass
class UncertaintyConfig:
    """Uncertainty estimation configuration."""
    # MC Dropout settings
    num_mc_samples: int = 10  # Number of forward passes for MC Dropout
    
    # Uncertainty type
    uncertainty_type: str = "both"  # Options: entropy, variance, both
    
    # Boundary extraction for spatial uncertainty
    boundary_kernel_size: int = 5  # For morphological operations


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic settings
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Learning rate schedule
    scheduler: str = "cosine"  # Options: cosine, step, plateau
    warmup_epochs: int = 5
    
    # Loss function
    loss_type: str = "dice_bce"  # Options: dice, bce, dice_bce, focal
    dice_weight: float = 0.5
    bce_weight: float = 0.5
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4
    
    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: str = "./checkpoints"
    
    # Logging
    log_interval: int = 10  # Log every N batches
    val_interval: int = 1  # Validate every N epochs


@dataclass
class ExperimentConfig:
    """Experiment configuration for ablation studies."""
    # Experiment name
    name: str = "full_vl_model"
    
    # Ablation settings
    use_text: bool = True
    text_attributes: List[str] = field(default_factory=lambda: [
        "shape", "size", "location", "boundary", "pathology"
    ])
    # Set to subset for ablation, e.g., ["shape"] or ["size", "location"]
    
    # Baseline comparison
    vision_only: bool = False  # If True, ignore text completely
    
    # Random seed
    seed: int = 42
    
    # Device
    device: str = "cuda"  # Options: cuda, cpu


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    text: TextConfig = field(default_factory=TextConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure dimensions match
        assert self.text.projection_dim == self.vision.feature_dim, \
            "Text projection dim must match vision feature dim for fusion"
        
        # Ensure valid split ratios
        total_split = self.data.train_split + self.data.val_split + self.data.test_split
        assert abs(total_split - 1.0) < 1e-6, "Data splits must sum to 1.0"


def get_config(experiment_name: str = "default") -> Config:
    """
    Get configuration for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment preset
        
    Returns:
        Config object with appropriate settings
    """
    config = Config()
    
    if experiment_name == "vision_only":
        config.experiment.name = "vision_only_baseline"
        config.experiment.use_text = False
        config.experiment.vision_only = True
        
    elif experiment_name == "text_shape_only":
        config.experiment.name = "ablation_shape"
        config.experiment.text_attributes = ["shape"]
        
    elif experiment_name == "text_size_only":
        config.experiment.name = "ablation_size"
        config.experiment.text_attributes = ["size"]
        
    elif experiment_name == "text_boundary_only":
        config.experiment.name = "ablation_boundary"
        config.experiment.text_attributes = ["boundary"]
        
    elif experiment_name == "text_location_only":
        config.experiment.name = "ablation_location"
        config.experiment.text_attributes = ["location"]
        
    elif experiment_name == "full":
        config.experiment.name = "full_vl_model"
        # Default settings, all attributes
        
    return config


# Ablation experiment configurations
ABLATION_EXPERIMENTS = [
    "vision_only",
    "text_shape_only", 
    "text_size_only",
    "text_boundary_only",
    "text_location_only",
    "full"
]