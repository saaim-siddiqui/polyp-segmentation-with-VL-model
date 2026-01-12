"""
Cross-Attention Fusion Module.

This module implements the fusion mechanism between vision and text features
using cross-attention, FiLM conditioning, or simple concatenation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import math


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention module for vision-language fusion.
    
    Allows image features to attend to text features,
    enabling text-guided feature modulation.
    """
    
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        """
        Initialize cross-attention fusion.
        
        Args:
            vision_dim: Dimension of vision features
            text_dim: Dimension of text features
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_residual: Whether to use residual connection
        """
        super().__init__()
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        
        assert vision_dim % num_heads == 0, "vision_dim must be divisible by num_heads"
        self.head_dim = vision_dim // num_heads
        
        # Query projection (from vision)
        self.query_proj = nn.Linear(vision_dim, vision_dim)
        
        # Key and Value projections (from text)
        self.key_proj = nn.Linear(text_dim, vision_dim)
        self.value_proj = nn.Linear(text_dim, vision_dim)
        
        # Output projection
        self.output_proj = nn.Linear(vision_dim, vision_dim)
        
        # Layer normalization
        self.norm_vision = nn.LayerNorm(vision_dim)
        self.norm_text = nn.LayerNorm(text_dim)
        self.norm_out = nn.LayerNorm(vision_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(vision_dim, vision_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim * 4, vision_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(vision_dim)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through cross-attention.
        
        Args:
            vision_features: Vision features [B, H*W, D] or [B, D, H, W]
            text_features: Text features [B, L, D_text]
            text_mask: Attention mask for text [B, L]
            
        Returns:
            Fused vision features with same shape as input
        """
        # Handle spatial features (B, D, H, W) -> (B, H*W, D)
        spatial_input = False
        if vision_features.dim() == 4:
            spatial_input = True
            B, D, H, W = vision_features.shape
            vision_features = vision_features.permute(0, 2, 3, 1).reshape(B, H * W, D)
        
        B, N, D = vision_features.shape
        _, L, _ = text_features.shape
        
        # Normalize inputs
        vision_normed = self.norm_vision(vision_features)
        text_normed = self.norm_text(text_features)
        
        # Compute Q, K, V
        Q = self.query_proj(vision_normed)  # [B, N, D]
        K = self.key_proj(text_normed)      # [B, L, D]
        V = self.value_proj(text_normed)    # [B, L, D]
        
        # Reshape for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, H, N, L]
        
        # Apply mask if provided
        if text_mask is not None:
            # Expand mask for heads: [B, L] -> [B, 1, 1, L]
            mask = text_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, H, N, d]
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        
        # Output projection
        attn_output = self.output_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Residual connection
        if self.use_residual:
            vision_features = vision_features + attn_output
        else:
            vision_features = attn_output
        
        # Feed-forward with residual
        ffn_out = self.ffn(self.norm_ffn(vision_features))
        vision_features = vision_features + ffn_out
        
        # Reshape back to spatial if needed
        if spatial_input:
            vision_features = vision_features.reshape(B, H, W, D).permute(0, 3, 1, 2)
        
        return vision_features


class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for vision-language fusion.
    
    Modulates vision features using learned scale and shift from text.
    Simpler than cross-attention but effective for global conditioning.
    """
    
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
    ):
        """
        Initialize FiLM fusion.
        
        Args:
            vision_dim: Dimension of vision features
            text_dim: Dimension of text features (pooled)
        """
        super().__init__()
        
        self.vision_dim = vision_dim
        
        # Generate scale (gamma) and shift (beta) from text
        self.film_generator = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, vision_dim * 2),  # gamma and beta
        )
        
        # Layer norm before modulation
        self.norm = nn.LayerNorm(vision_dim)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through FiLM.
        
        Args:
            vision_features: Vision features [B, D, H, W] or [B, N, D]
            text_features: Pooled text features [B, D_text]
            
        Returns:
            Modulated vision features with same shape as input
        """
        # Handle spatial features
        spatial_input = False
        if vision_features.dim() == 4:
            spatial_input = True
            B, D, H, W = vision_features.shape
            vision_features = vision_features.permute(0, 2, 3, 1)  # [B, H, W, D]
        
        # Generate FiLM parameters
        film_params = self.film_generator(text_features)  # [B, 2*D]
        gamma, beta = film_params.chunk(2, dim=-1)  # [B, D] each
        
        # Expand for broadcasting
        if spatial_input:
            gamma = gamma.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, D]
            beta = beta.unsqueeze(1).unsqueeze(1)
        else:
            gamma = gamma.unsqueeze(1)  # [B, 1, D]
            beta = beta.unsqueeze(1)
        
        # Apply FiLM: y = gamma * x + beta
        vision_features = self.norm(vision_features)
        vision_features = gamma * vision_features + beta
        
        # Reshape back if needed
        if spatial_input:
            vision_features = vision_features.permute(0, 3, 1, 2)  # [B, D, H, W]
        
        return vision_features


class ConcatFusion(nn.Module):
    """
    Simple concatenation-based fusion.
    
    Concatenates global text features to vision features
    and uses a projection layer to fuse.
    """
    
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
    ):
        """
        Initialize concat fusion.
        
        Args:
            vision_dim: Dimension of vision features
            text_dim: Dimension of text features (pooled)
        """
        super().__init__()
        
        self.vision_dim = vision_dim
        
        # Projection after concatenation
        self.fusion_proj = nn.Sequential(
            nn.Linear(vision_dim + text_dim, vision_dim),
            nn.LayerNorm(vision_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through concat fusion.
        
        Args:
            vision_features: Vision features [B, D, H, W] or [B, N, D]
            text_features: Pooled text features [B, D_text]
            
        Returns:
            Fused vision features with same shape as input
        """
        # Handle spatial features
        spatial_input = False
        if vision_features.dim() == 4:
            spatial_input = True
            B, D, H, W = vision_features.shape
            vision_features = vision_features.permute(0, 2, 3, 1).reshape(B, H * W, D)
        
        B, N, D = vision_features.shape
        
        # Expand text features for concatenation
        text_expanded = text_features.unsqueeze(1).expand(-1, N, -1)  # [B, N, D_text]
        
        # Concatenate and project
        fused = torch.cat([vision_features, text_expanded], dim=-1)
        fused = self.fusion_proj(fused)
        
        # Reshape back if needed
        if spatial_input:
            fused = fused.reshape(B, H, W, D).permute(0, 3, 1, 2)
        
        return fused


class MultiScaleFusion(nn.Module):
    """
    Multi-scale fusion module.
    
    Applies fusion at multiple decoder stages using the specified fusion type.
    """
    
    def __init__(
        self,
        vision_dims: list,
        text_dim: int,
        fusion_type: str = "cross_attention",
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-scale fusion.
        
        Args:
            vision_dims: List of vision feature dimensions at each scale
            text_dim: Dimension of text features
            fusion_type: Type of fusion (cross_attention, film, concat)
            num_heads: Number of attention heads (for cross-attention)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # Create fusion modules for each scale
        self.fusion_modules = nn.ModuleList()
        
        for vision_dim in vision_dims:
            if fusion_type == "cross_attention":
                module = CrossAttentionFusion(
                    vision_dim=vision_dim,
                    text_dim=text_dim,
                    num_heads=min(num_heads, vision_dim // 32),  # Adjust heads for smaller dims
                    dropout=dropout,
                )
            elif fusion_type == "film":
                module = FiLMFusion(
                    vision_dim=vision_dim,
                    text_dim=text_dim,
                )
            elif fusion_type == "concat":
                module = ConcatFusion(
                    vision_dim=vision_dim,
                    text_dim=text_dim,
                )
            else:
                raise ValueError(f"Unknown fusion type: {fusion_type}")
            
            self.fusion_modules.append(module)
    
    def forward(
        self,
        vision_features_list: list,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> list:
        """
        Forward pass through multi-scale fusion.
        
        Args:
            vision_features_list: List of vision features at different scales
            text_features: Text features (sequence for cross-attn, pooled for film/concat)
            text_mask: Optional attention mask
            
        Returns:
            List of fused features at each scale
        """
        fused_features = []
        
        for i, (vision_feat, fusion_module) in enumerate(zip(vision_features_list, self.fusion_modules)):
            if self.fusion_type == "cross_attention":
                fused = fusion_module(vision_feat, text_features, text_mask)
            else:
                # FiLM and concat use pooled text features
                if text_features.dim() == 3:
                    # Use mean pooling if sequence features provided
                    if text_mask is not None:
                        mask_expanded = text_mask.unsqueeze(-1).float()
                        pooled = (text_features * mask_expanded).sum(1) / mask_expanded.sum(1)
                    else:
                        pooled = text_features.mean(1)
                else:
                    pooled = text_features
                fused = fusion_module(vision_feat, pooled)
            
            fused_features.append(fused)
        
        return fused_features


# For testing
if __name__ == "__main__":
    # Test cross-attention fusion
    print("Testing CrossAttentionFusion...")
    cross_attn = CrossAttentionFusion(
        vision_dim=256,
        text_dim=256,
        num_heads=8,
    )
    
    vision_feat = torch.randn(2, 256, 8, 8)  # Spatial features
    text_feat = torch.randn(2, 20, 256)       # Sequence features
    text_mask = torch.ones(2, 20)             # All tokens valid
    
    fused = cross_attn(vision_feat, text_feat, text_mask)
    print(f"Input shape: {vision_feat.shape}")
    print(f"Output shape: {fused.shape}")
    
    # Test FiLM fusion
    print("\nTesting FiLMFusion...")
    film = FiLMFusion(vision_dim=256, text_dim=256)
    
    pooled_text = torch.randn(2, 256)
    fused_film = film(vision_feat, pooled_text)
    print(f"Input shape: {vision_feat.shape}")
    print(f"Output shape: {fused_film.shape}")
    
    # Test concat fusion
    print("\nTesting ConcatFusion...")
    concat = ConcatFusion(vision_dim=256, text_dim=256)
    
    fused_concat = concat(vision_feat, pooled_text)
    print(f"Input shape: {vision_feat.shape}")
    print(f"Output shape: {fused_concat.shape}")