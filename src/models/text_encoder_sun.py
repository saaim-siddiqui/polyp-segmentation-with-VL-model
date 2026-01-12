"""
Text Encoder Module - Updated for SUN Colonoscopy Database.

This module handles text encoding using pretrained language models
(PubMedBERT, BioClinicalBERT, or CLIP text encoder).

Updated to support Paris Classification morphology and anatomical locations.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
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


class SUNCaptionGenerator:
    """
    Generate structured captions from SUN database metadata.
    
    Handles Paris Classification morphology and anatomical locations
    specific to the SUN Colonoscopy Video Database.
    """
    
    # Paris Classification morphology descriptions
    # Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3749928/
    SHAPE_TEMPLATES = {
        # Polypoid (protruding) lesions
        "Ip": "pedunculated polypoid with a distinct stalk",
        "Isp": "semi-pedunculated polypoid with a broad-based stalk",
        "Is": "sessile polypoid, elevated without a stalk",
        # Non-polypoid (flat) lesions
        "IIa": "slightly elevated flat lesion",
        "IIa(LST-NG)": "laterally spreading tumor, non-granular type, flat elevated",
        "IIb": "completely flat lesion",
        "IIc": "slightly depressed flat lesion",
        # Combined
        "IIa+IIc": "flat lesion with both elevated and depressed components",
        # Default for unknown
        "unknown": "lesion of indeterminate morphology",
    }
    
    # Size categorization (clinical relevance)
    SIZE_CATEGORIES = {
        "diminutive": (0, 5),      # ≤5mm - typically hyperplastic or low-risk
        "small": (6, 9),           # 6-9mm - intermediate risk
        "large": (10, 19),         # 10-19mm - higher risk
        "giant": (20, float('inf'))  # ≥20mm - highest risk
    }
    
    SIZE_TEMPLATES = {
        "diminutive": "diminutive (5mm or smaller), typically lower risk",
        "small": "small-sized (6-9mm), intermediate risk",
        "large": "large (10-19mm), higher malignancy risk",
        "giant": "giant (20mm or larger), highest risk requiring careful evaluation",
    }
    
    # Anatomical location descriptions (proximal to distal)
    LOCATION_TEMPLATES = {
        "Cecum": "located in the cecum, the proximal-most part of the colon",
        "Ascending colon": "located in the ascending colon, right side of the abdomen",
        "Transverse colon": "located in the transverse colon, crossing the upper abdomen",
        "Descending colon": "located in the descending colon, left side of the abdomen",
        "Sigmoid colon": "located in the sigmoid colon, the S-shaped distal segment",
        "Rectum": "located in the rectum, the terminal portion before the anus",
        # Grouped locations for analysis
        "proximal": "located in the proximal (right) colon",
        "distal": "located in the distal (left) colon",
    }
    
    # Pathology descriptions with clinical significance
    PATHOLOGY_TEMPLATES = {
        "Low-grade adenoma": "consistent with low-grade adenoma, a precancerous lesion with mild dysplasia",
        "High-grade adenoma": "suggestive of high-grade adenoma with severe dysplasia, higher malignancy risk",
        "Hyperplastic polyp": "appearing as a hyperplastic polyp, typically benign with low malignant potential",
        "Sessile serrated lesion": "features of sessile serrated lesion, requires complete removal due to malignant potential",
        "Traditional serrated adenoma": "consistent with traditional serrated adenoma, a serrated pathway precursor lesion",
        "Invasive cancer (T1b)": "concerning for invasive adenocarcinoma with submucosal invasion",
        "Invasive cancer": "concerning for invasive adenocarcinoma",
        # Aliases
        "SSA": "features of sessile serrated adenoma, requires complete removal",
        "TSA": "consistent with traditional serrated adenoma",
        "SSL": "features of sessile serrated lesion, requires complete removal due to malignant potential",
    }
    
    def __init__(self, attributes_to_include: List[str] = None):
        """
        Initialize caption generator for SUN database.
        
        Args:
            attributes_to_include: List of attribute names to include in captions.
                                  Options: ["shape", "size", "location", "pathology"]
                                  If None, include all attributes.
        """
        self.attributes_to_include = attributes_to_include or [
            "shape", "size", "location", "pathology"
        ]
    
    def _parse_size(self, size_str: str) -> tuple:
        """
        Parse size string to get numeric value and category.
        
        Args:
            size_str: Size string like "3mm", "15mm-", "10mm"
            
        Returns:
            Tuple of (size_mm, category)
        """
        if size_str is None:
            return None, "unknown"
        
        # Handle "15mm-" format (means >= 15mm)
        size_str = str(size_str).strip().lower()
        
        if size_str.endswith("mm-"):
            size_mm = int(size_str.replace("mm-", ""))
        elif size_str.endswith("mm"):
            size_mm = int(size_str.replace("mm", ""))
        else:
            try:
                size_mm = int(size_str)
            except ValueError:
                return None, "unknown"
        
        # Categorize
        for category, (min_size, max_size) in self.SIZE_CATEGORIES.items():
            if min_size <= size_mm <= max_size:
                return size_mm, category
        
        return size_mm, "unknown"
    
    def _get_location_group(self, location: str) -> str:
        """
        Group location into proximal vs distal colon.
        
        Proximal (right) colon: Cecum, Ascending, Transverse (proximal 2/3)
        Distal (left) colon: Transverse (distal 1/3), Descending, Sigmoid, Rectum
        
        Args:
            location: Anatomical location string
            
        Returns:
            "proximal" or "distal"
        """
        proximal = ["Cecum", "Ascending colon"]
        # Transverse is technically middle, but often grouped with proximal
        
        if location in proximal:
            return "proximal"
        elif location == "Transverse colon":
            return "mid"  # Can go either way clinically
        else:
            return "distal"
    
    def generate_caption(
        self,
        shape: Optional[str] = None,
        size: Optional[str] = None,
        location: Optional[str] = None,
        pathology: Optional[str] = None,
    ) -> str:
        """
        Generate a structured caption from SUN database attributes.
        
        Args:
            shape: Paris Classification (Is, Ip, Isp, IIa, etc.)
            size: Size in mm (e.g., "3mm", "15mm-")
            location: Anatomical location (e.g., "Sigmoid colon")
            pathology: Pathological diagnosis
            
        Returns:
            Generated caption string
        """
        parts = ["A colorectal polyp is identified"]
        
        # Add location
        if "location" in self.attributes_to_include and location:
            loc_text = self.LOCATION_TEMPLATES.get(
                location, 
                f"located in the {location}"
            )
            parts.append(loc_text)
        
        # Describe morphology
        lesion_parts = []
        
        if "shape" in self.attributes_to_include and shape:
            shape_text = self.SHAPE_TEMPLATES.get(
                shape, 
                self.SHAPE_TEMPLATES.get("unknown")
            )
            lesion_parts.append(f"The lesion is {shape_text}")
        
        if "size" in self.attributes_to_include and size:
            size_mm, size_category = self._parse_size(size)
            if size_mm:
                size_text = self.SIZE_TEMPLATES.get(
                    size_category, 
                    f"measuring approximately {size_mm}mm"
                )
                lesion_parts.append(f"It is {size_text}")
        
        parts.extend(lesion_parts)
        
        # Add pathology
        if "pathology" in self.attributes_to_include and pathology:
            path_text = self.PATHOLOGY_TEMPLATES.get(
                pathology, 
                f"with pathology suggesting {pathology}"
            )
            parts.append(f"Histologically {path_text}")
        
        # Combine into coherent caption
        caption = ". ".join(parts) + "."
        
        return caption
    
    def generate_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Generate caption from a metadata dictionary.
        
        Args:
            metadata: Dictionary containing attribute keys and values
                     Expected keys: shape, size, location, pathology
            
        Returns:
            Generated caption string
        """
        return self.generate_caption(
            shape=metadata.get("shape") or metadata.get("Shape"),
            size=metadata.get("size") or metadata.get("Size"),
            location=metadata.get("location") or metadata.get("Location"),
            pathology=metadata.get("pathology") or metadata.get("Pathological diagnosis"),
        )


# Backward compatibility alias
CaptionGenerator = SUNCaptionGenerator


# For testing
if __name__ == "__main__":
    # Test caption generation with SUN database examples
    generator = SUNCaptionGenerator()
    
    # Example from SUN database (ID 1)
    sample_1 = {
        "shape": "Is",
        "size": "6mm",
        "location": "Cecum",
        "pathology": "Low-grade adenoma",
    }
    
    # Example from SUN database (ID 35) - High grade
    sample_35 = {
        "shape": "Ip",
        "size": "15mm-",
        "location": "Sigmoid colon",
        "pathology": "High-grade adenoma",
    }
    
    # Example from SUN database (ID 63) - Cancer
    sample_63 = {
        "shape": "Is",
        "size": "7mm",
        "location": "Rectum",
        "pathology": "Invasive cancer (T1b)",
    }
    
    print("=" * 60)
    print("SUN Database Caption Generation Examples")
    print("=" * 60)
    
    print("\nSample 1 (Low-grade adenoma):")
    print(generator.generate_from_metadata(sample_1))
    
    print("\nSample 35 (High-grade adenoma, large):")
    print(generator.generate_from_metadata(sample_35))
    
    print("\nSample 63 (Invasive cancer):")
    print(generator.generate_from_metadata(sample_63))
    
    # Test ablation (only shape)
    print("\n" + "=" * 60)
    print("Ablation: Shape-only caption")
    print("=" * 60)
    generator_shape_only = SUNCaptionGenerator(attributes_to_include=["shape"])
    print(generator_shape_only.generate_from_metadata(sample_1))
    
    # Test ablation (only pathology)
    print("\n" + "=" * 60)
    print("Ablation: Pathology-only caption")
    print("=" * 60)
    generator_path_only = SUNCaptionGenerator(attributes_to_include=["pathology"])
    print(generator_path_only.generate_from_metadata(sample_1))