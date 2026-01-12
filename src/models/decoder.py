"""
Decoder Module with Uncertainty Estimation.

This module implements a U-Net style decoder with skip connections
and MC Dropout for uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ConvBlock(nn.Module):
    """
    Convolutional block with optional dropout for uncertainty estimation.
    
    Conv -> BN -> ReLU -> Conv -> BN -> ReLU (-> Dropout)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
        use_mc_dropout: bool = False,
    ):
        """
        Initialize convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dropout_rate: Dropout probability
            use_mc_dropout: If True, apply dropout during inference too
        """
        super().__init__()
        
        self.use_mc_dropout = use_mc_dropout
        self.dropout_rate = dropout_rate
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional MC dropout."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        if self.dropout is not None:
            if self.use_mc_dropout:
                # Always apply dropout (for MC dropout during inference)
                x = F.dropout2d(x, p=self.dropout_rate, training=True)
            else:
                x = self.dropout(x)
        
        return x


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling and skip connection.
    
    Upsample -> Concat with skip -> ConvBlock
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
        use_mc_dropout: bool = False,
    ):
        """
        Initialize decoder block.
        
        Args:
            in_channels: Number of input channels (from previous decoder stage)
            skip_channels: Number of channels from skip connection
            out_channels: Number of output channels
            dropout_rate: Dropout probability
            use_mc_dropout: If True, apply dropout during inference
        """
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        
        self.conv_block = ConvBlock(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            dropout_rate=dropout_rate,
            use_mc_dropout=use_mc_dropout,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through decoder block.
        
        Args:
            x: Input features from previous decoder stage
            skip: Skip connection features from encoder
            
        Returns:
            Upsampled and fused features
        """
        x = self.upsample(x)
        
        if skip is not None:
            # Handle size mismatch due to odd input sizes
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        
        return self.conv_block(x)


class SegmentationDecoder(nn.Module):
    """
    U-Net style segmentation decoder with MC Dropout support.
    
    Takes bottleneck features and skip connections from encoder,
    progressively upsamples to original resolution.
    """
    
    def __init__(
        self,
        bottleneck_channels: int,
        skip_channels: List[int],
        decoder_channels: List[int],
        num_classes: int = 1,
        dropout_rate: float = 0.1,
        use_mc_dropout: bool = True,
    ):
        """
        Initialize segmentation decoder.
        
        Args:
            bottleneck_channels: Number of channels in bottleneck
            skip_channels: List of skip connection channels [stage3, stage2, stage1, stage0]
            decoder_channels: List of decoder output channels
            num_classes: Number of output classes (1 for binary)
            dropout_rate: Dropout probability for MC dropout
            use_mc_dropout: Whether to use MC dropout for uncertainty
        """
        super().__init__()
        
        self.use_mc_dropout = use_mc_dropout
        self.dropout_rate = dropout_rate
        
        # Verify configuration
        assert len(skip_channels) == len(decoder_channels), \
            "Number of skip channels must match decoder channels"
        
        # Build decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        in_ch = bottleneck_channels
        for i, (skip_ch, out_ch) in enumerate(zip(skip_channels, decoder_channels)):
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    dropout_rate=dropout_rate,
                    use_mc_dropout=use_mc_dropout,
                )
            )
            in_ch = out_ch
        
        # Final upsampling to original resolution
        self.final_upsample = nn.ConvTranspose2d(
            decoder_channels[-1], decoder_channels[-1],
            kernel_size=2, stride=2
        )
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1),
        )
    
    def forward(
        self,
        bottleneck: torch.Tensor,
        skip_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            bottleneck: Bottleneck features [B, C, H/32, W/32]
            skip_features: List of skip features [stage0, stage1, stage2, stage3]
                          (from encoder, ordered from shallow to deep)
            
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        # Reverse skip features to match decoder order (deep to shallow)
        skip_features = skip_features[::-1]  # [stage3, stage2, stage1, stage0]
        
        x = bottleneck
        
        # Decoder stages
        for i, (decoder_block, skip) in enumerate(zip(self.decoder_blocks, skip_features)):
            x = decoder_block(x, skip)
        
        # Final upsampling
        x = self.final_upsample(x)
        
        # Segmentation head
        logits = self.seg_head(x)
        
        return logits
    
    def enable_mc_dropout(self):
        """Enable MC dropout mode for uncertainty estimation."""
        self.use_mc_dropout = True
        for block in self.decoder_blocks:
            block.conv_block.use_mc_dropout = True
    
    def disable_mc_dropout(self):
        """Disable MC dropout mode."""
        self.use_mc_dropout = False
        for block in self.decoder_blocks:
            block.conv_block.use_mc_dropout = False


class UncertaintyEstimator(nn.Module):
    """
    Uncertainty estimation module using MC Dropout.
    
    Performs multiple forward passes with dropout enabled
    to estimate predictive uncertainty.
    """
    
    def __init__(
        self,
        num_mc_samples: int = 10,
        uncertainty_type: str = "both",
    ):
        """
        Initialize uncertainty estimator.
        
        Args:
            num_mc_samples: Number of MC dropout samples
            uncertainty_type: Type of uncertainty to compute
                            ('entropy', 'variance', 'both')
        """
        super().__init__()
        
        self.num_mc_samples = num_mc_samples
        self.uncertainty_type = uncertainty_type
    
    def compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute predictive entropy.
        
        Args:
            probs: Mean predicted probabilities [B, C, H, W]
            
        Returns:
            Entropy map [B, 1, H, W]
        """
        # For binary segmentation, compute binary entropy
        # H = -p*log(p) - (1-p)*log(1-p)
        eps = 1e-8
        entropy = -probs * torch.log(probs + eps) - (1 - probs) * torch.log(1 - probs + eps)
        return entropy
    
    def compute_variance(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Compute predictive variance (epistemic uncertainty).
        
        Args:
            samples: MC samples [T, B, C, H, W]
            
        Returns:
            Variance map [B, C, H, W]
        """
        return samples.var(dim=0)
    
    @torch.no_grad()
    def estimate_uncertainty(
        self,
        model: nn.Module,
        images: torch.Tensor,
        **forward_kwargs,
    ) -> dict:
        """
        Estimate uncertainty using MC Dropout.
        
        Args:
            model: The segmentation model (must have mc_forward method or 
                   support training mode dropout)
            images: Input images [B, 3, H, W]
            **forward_kwargs: Additional arguments for model forward pass
            
        Returns:
            Dictionary containing:
            - 'prediction': Mean prediction [B, 1, H, W]
            - 'entropy': Entropy uncertainty [B, 1, H, W] (if applicable)
            - 'variance': Variance uncertainty [B, 1, H, W] (if applicable)
            - 'samples': All MC samples [T, B, 1, H, W]
        """
        # Store original training mode
        was_training = model.training
        
        # Enable MC dropout
        if hasattr(model, 'enable_mc_dropout'):
            model.enable_mc_dropout()
        
        # Set to eval but keep dropout active
        model.eval()
        
        # Collect MC samples
        samples = []
        for _ in range(self.num_mc_samples):
            logits = model(images, **forward_kwargs)
            probs = torch.sigmoid(logits)
            samples.append(probs)
        
        # Stack samples: [T, B, C, H, W]
        samples = torch.stack(samples, dim=0)
        
        # Compute mean prediction
        mean_probs = samples.mean(dim=0)
        
        # Compute uncertainties
        results = {
            'prediction': mean_probs,
            'samples': samples,
        }
        
        if self.uncertainty_type in ['entropy', 'both']:
            results['entropy'] = self.compute_entropy(mean_probs)
        
        if self.uncertainty_type in ['variance', 'both']:
            results['variance'] = self.compute_variance(samples)
        
        # Restore model state
        if hasattr(model, 'disable_mc_dropout'):
            model.disable_mc_dropout()
        
        if was_training:
            model.train()
        
        return results


# For testing
if __name__ == "__main__":
    # Test decoder
    print("Testing SegmentationDecoder...")
    
    decoder = SegmentationDecoder(
        bottleneck_channels=256,
        skip_channels=[256, 128, 64, 64],  # From stage3 to stage0
        decoder_channels=[256, 128, 64, 32],
        num_classes=1,
        dropout_rate=0.1,
        use_mc_dropout=True,
    )
    
    # Create dummy inputs
    bottleneck = torch.randn(2, 256, 8, 8)  # For 256x256 input
    skip_features = [
        torch.randn(2, 64, 128, 128),   # stage0: H/2
        torch.randn(2, 64, 64, 64),     # stage1: H/4
        torch.randn(2, 128, 32, 32),    # stage2: H/8
        torch.randn(2, 256, 16, 16),    # stage3: H/16
    ]
    
    # Forward pass
    output = decoder(bottleneck, skip_features)
    print(f"Input bottleneck shape: {bottleneck.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test uncertainty estimation
    print("\nTesting UncertaintyEstimator...")
    
    # Create a simple mock model
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, 3, padding=1)
            self.dropout = nn.Dropout2d(0.1)
        
        def forward(self, x):
            return self.dropout(self.conv(x))
        
        def enable_mc_dropout(self):
            pass
        
        def disable_mc_dropout(self):
            pass
    
    mock_model = MockModel()
    estimator = UncertaintyEstimator(num_mc_samples=5, uncertainty_type='both')
    
    images = torch.randn(2, 3, 64, 64)
    uncertainty_results = estimator.estimate_uncertainty(mock_model, images)
    
    print(f"Prediction shape: {uncertainty_results['prediction'].shape}")
    print(f"Entropy shape: {uncertainty_results['entropy'].shape}")
    print(f"Variance shape: {uncertainty_results['variance'].shape}")
    print(f"Samples shape: {uncertainty_results['samples'].shape}")