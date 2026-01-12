"""
Text Encoder Module.

This module handles text encoding using pretrained language models
(PubMedBERT, BioClinicalBERT, or CLIP text encoder).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """
    Text encoder using pretrained language models.
    
    Supports:
    - PubMedBERT (recommended for medical domain)
    - BioClinicalBERT
    - CLIP text encoder
    
    Includes a projection layer to match vision feature dimensions.
    """
    
    def __init__(
        self,
        encoder_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        projection_dim: int = 256,
        max_length: int = 128,
        freeze_encoder: bool = True,
        unfreeze_layers: int = 0,
    ):
        """
        Initialize text encoder.
        
        Args:
            encoder_name: HuggingFace model name or path
            projection_dim: Output dimension after projection
            max_length: Maximum sequence length for tokenization
            freeze_encoder: Whether to freeze encoder weights
            unfreeze_layers: Number of top layers to unfreeze (0 = all frozen)
        """
        super().__init__()
        
        self.encoder_name = encoder_name
        self.max_length = max_length
        self.projection_dim = projection_dim
        
        # Load pretrained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        
        # Get hidden size from encoder config
        self.hidden_size = self.encoder.config.hidden_size
        
        # Projection layer: map from hidden_size to projection_dim
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Global pooling projection (for conditioning)
        self.global_projection = nn.Sequential(
            nn.Linear(self.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        
        # Freeze encoder if specified
        if freeze_encoder:
            self._freeze_encoder(unfreeze_layers)
    
    def _freeze_encoder(self, unfreeze_layers: int = 0):
        """
        Freeze encoder parameters, optionally unfreezing top layers.
        
        Args:
            unfreeze_layers: Number of top layers to keep trainable
        """
        # Freeze all parameters first
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze top layers if specified
        if unfreeze_layers > 0:
            # Get encoder layers (works for BERT-like models)
            if hasattr(self.encoder, 'encoder'):
                layers = self.encoder.encoder.layer
                for layer in layers[-unfreeze_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            
            # Also unfreeze pooler if it exists
            if hasattr(self.encoder, 'pooler') and self.encoder.pooler is not None:
                for param in self.encoder.pooler.parameters():
                    param.requires_grad = True
    
    def tokenize(
        self,
        texts: list,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            texts: List of text strings
            device: Device to place tensors on
            
        Returns:
            Dictionary with input_ids, attention_mask
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        
        return encoded
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through text encoder.
        
        Args:
            input_ids: Tokenized input IDs [B, L]
            attention_mask: Attention mask [B, L]
            texts: Raw text strings (alternative to input_ids)
            
        Returns:
            Dictionary containing:
            - 'sequence': Projected sequence features [B, L, D]
            - 'pooled': Projected pooled (global) features [B, D]
            - 'attention_mask': Attention mask for cross-attention
        """
        # Tokenize if raw texts provided
        if texts is not None:
            device = next(self.parameters()).device
            encoded = self.tokenize(texts, device)
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
        
        # Encode text
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get sequence output [B, L, H]
        sequence_output = outputs.last_hidden_state
        
        # Get pooled output [B, H]
        # Use [CLS] token if pooler not available
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = sequence_output[:, 0, :]  # [CLS] token
        
        # Project to target dimension
        sequence_projected = self.projection(sequence_output)  # [B, L, D]
        pooled_projected = self.global_projection(pooled_output)  # [B, D]
        
        return {
            'sequence': sequence_projected,
            'pooled': pooled_projected,
            'attention_mask': attention_mask,
        }


class CaptionGenerator:
    """
    Generate structured captions from metadata attributes.
    
    This class creates templated captions from polyp metadata for training.
    """
    
    # Templates for different attributes
    SHAPE_TEMPLATES = {
        "regular": "regular, smooth-edged",
        "irregular": "irregular, with uneven boundaries",
        "pedunculated": "pedunculated, with a stalk-like base",
        "sessile": "sessile, flat against the mucosal surface",
    }
    
    SIZE_TEMPLATES = {
        "small": "small-sized (less than 5mm in diameter)",
        "medium": "medium-sized (between 5-10mm in diameter)",
        "large": "large-sized (greater than 10mm in diameter)",
    }
    
    LOCATION_TEMPLATES = {
        "central": "positioned centrally in the field of view",
        "peripheral": "located at the periphery of the image",
        "near_fold": "situated near a mucosal fold",
        "partially_visible": "partially visible at the edge of the frame",
    }
    
    BOUNDARY_TEMPLATES = {
        "clear": "with clearly defined boundaries",
        "ambiguous": "with ambiguous, hard-to-distinguish margins",
        "obscured": "with boundaries partially obscured by debris or fluid",
    }
    
    PATHOLOGY_TEMPLATES = {
        "hyperplastic": "appearing consistent with a hyperplastic polyp",
        "adenoma": "suggestive of an adenomatous polyp",
        "adenoma_low_grade": "suggestive of a low-grade adenoma",
        "adenoma_high_grade": "suggestive of a high-grade adenoma",
        "carcinoma": "with features concerning for carcinoma",
        "unknown": "of indeterminate pathological type",
    }
    
    def __init__(self, attributes_to_include: list = None):
        """
        Initialize caption generator.
        
        Args:
            attributes_to_include: List of attribute names to include in captions.
                                  If None, include all attributes.
        """
        self.attributes_to_include = attributes_to_include or [
            "shape", "size", "location", "boundary", "pathology"
        ]
    
    def generate_caption(
        self,
        shape: Optional[str] = None,
        size: Optional[str] = None,
        location: Optional[str] = None,
        boundary: Optional[str] = None,
        pathology: Optional[str] = None,
    ) -> str:
        """
        Generate a structured caption from attributes.
        
        Args:
            shape: Shape attribute value
            size: Size attribute value
            location: Location attribute value
            boundary: Boundary clarity attribute value
            pathology: Pathology type attribute value
            
        Returns:
            Generated caption string
        """
        parts = ["A polyp is visible"]
        
        # Add location if specified
        if "location" in self.attributes_to_include and location:
            loc_text = self.LOCATION_TEMPLATES.get(location, f"located {location}")
            parts.append(loc_text)
        
        # Start describing the lesion
        lesion_parts = []
        
        if "size" in self.attributes_to_include and size:
            size_text = self.SIZE_TEMPLATES.get(size, f"{size}-sized")
            lesion_parts.append(size_text)
        
        if "shape" in self.attributes_to_include and shape:
            shape_text = self.SHAPE_TEMPLATES.get(shape, shape)
            lesion_parts.append(shape_text)
        
        if lesion_parts:
            parts.append("The lesion is " + ", ".join(lesion_parts))
        
        # Add boundary information
        if "boundary" in self.attributes_to_include and boundary:
            bound_text = self.BOUNDARY_TEMPLATES.get(boundary, f"with {boundary} boundaries")
            parts.append(bound_text)
        
        # Add pathology
        if "pathology" in self.attributes_to_include and pathology:
            path_text = self.PATHOLOGY_TEMPLATES.get(pathology, f"{pathology} type")
            parts.append(path_text)
        
        # Combine into coherent caption
        caption = ". ".join(parts) + "."
        
        return caption
    
    def generate_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Generate caption from a metadata dictionary.
        
        Args:
            metadata: Dictionary containing attribute keys and values
            
        Returns:
            Generated caption string
        """
        return self.generate_caption(
            shape=metadata.get("shape"),
            size=metadata.get("size"),
            location=metadata.get("location"),
            boundary=metadata.get("boundary"),
            pathology=metadata.get("pathology"),
        )


# For testing
if __name__ == "__main__":
    # Test caption generation
    generator = CaptionGenerator()
    
    sample_metadata = {
        "shape": "irregular",
        "size": "medium",
        "location": "central",
        "boundary": "ambiguous",
        "pathology": "adenoma",
    }
    
    caption = generator.generate_from_metadata(sample_metadata)
    print("Generated caption:")
    print(caption)
    print()
    
    # Test ablation (only shape)
    generator_shape_only = CaptionGenerator(attributes_to_include=["shape"])
    caption_shape = generator_shape_only.generate_from_metadata(sample_metadata)
    print("Shape-only caption:")
    print(caption_shape)