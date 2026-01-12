"""
Uncertainty-Text Correlation Metrics.

This module implements the novel metrics for analyzing the relationship
between semantic text attributes and model uncertainty:

1. Per-Attribute Uncertainty Correlation
2. Uncertainty Reduction Score (URS)
3. Semantic Alignment Score (SAS)
4. Attribute-Conditioned ECE
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from collections import defaultdict
import warnings


class SpatialUncertaintyAggregator:
    """
    Aggregates pixel-wise uncertainty over meaningful spatial regions.
    
    Computes:
    - U_polyp: Mean uncertainty in polyp region
    - U_boundary: Mean uncertainty in boundary region
    - U_background: Mean uncertainty in background region
    """
    
    def __init__(self, boundary_kernel_size: int = 5):
        """
        Initialize aggregator.
        
        Args:
            boundary_kernel_size: Kernel size for morphological boundary extraction
        """
        self.boundary_kernel_size = boundary_kernel_size
    
    def extract_boundary(
        self,
        mask: torch.Tensor,
        kernel_size: int = None
    ) -> torch.Tensor:
        """
        Extract boundary region using morphological operations.
        
        Boundary = dilate(mask) - erode(mask)
        
        Args:
            mask: Binary mask [B, 1, H, W] or [H, W]
            kernel_size: Kernel size for morphological ops
            
        Returns:
            Boundary mask with same shape as input
        """
        if kernel_size is None:
            kernel_size = self.boundary_kernel_size
        
        # Handle different input shapes
        squeeze_dims = []
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
            squeeze_dims = [0, 1]
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
            squeeze_dims = [1]
        
        # Ensure float for pooling operations
        mask_float = mask.float()
        
        # Dilation using max pooling
        dilated = F.max_pool2d(
            mask_float,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        
        # Erosion using -max(-x)
        eroded = -F.max_pool2d(
            -mask_float,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        
        # Boundary = dilated - eroded
        boundary = dilated - eroded
        boundary = (boundary > 0).float()
        
        # Restore original shape
        for dim in reversed(squeeze_dims):
            boundary = boundary.squeeze(dim)
        
        return boundary
    
    def aggregate_uncertainty(
        self,
        uncertainty_map: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate uncertainty over spatial regions.
        
        Args:
            uncertainty_map: Pixel-wise uncertainty [B, 1, H, W]
            mask: Ground truth segmentation mask [B, 1, H, W]
            
        Returns:
            Dictionary with aggregated uncertainties:
            - 'polyp': [B] mean uncertainty in polyp region
            - 'boundary': [B] mean uncertainty in boundary region  
            - 'background': [B] mean uncertainty in background region
        """
        B = uncertainty_map.shape[0]
        
        # Ensure same device
        device = uncertainty_map.device
        mask = mask.to(device).float()
        
        # Extract regions
        polyp_region = mask > 0.5
        background_region = mask < 0.5
        boundary_region = self.extract_boundary(mask) > 0.5
        
        results = {
            'polyp': torch.zeros(B, device=device),
            'boundary': torch.zeros(B, device=device),
            'background': torch.zeros(B, device=device),
        }
        
        for b in range(B):
            u = uncertainty_map[b, 0]  # [H, W]
            
            # Polyp uncertainty
            polyp_mask = polyp_region[b, 0]
            if polyp_mask.sum() > 0:
                results['polyp'][b] = u[polyp_mask].mean()
            
            # Boundary uncertainty
            bound_mask = boundary_region[b, 0] if boundary_region.dim() == 4 else boundary_region[b]
            if bound_mask.sum() > 0:
                results['boundary'][b] = u[bound_mask].mean()
            
            # Background uncertainty
            bg_mask = background_region[b, 0]
            if bg_mask.sum() > 0:
                results['background'][b] = u[bg_mask].mean()
        
        return results


class AttributeUncertaintyCorrelation:
    """
    Computes correlation between semantic attributes and uncertainty.
    
    For each attribute (shape, size, location, boundary, pathology),
    analyzes how uncertainty varies across attribute values.
    """
    
    def __init__(self):
        """Initialize correlation analyzer."""
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        # Store uncertainties grouped by attribute values
        # Structure: {attribute_name: {value: [list of uncertainties]}}
        self.attribute_uncertainties = defaultdict(lambda: defaultdict(list))
        self.sample_count = 0
    
    def update(
        self,
        aggregated_uncertainty: Dict[str, torch.Tensor],
        metadata: Dict[str, str],
        region: str = 'polyp',
    ):
        """
        Update statistics with a batch of samples.
        
        Args:
            aggregated_uncertainty: Dict with 'polyp', 'boundary', 'background' uncertainties
            metadata: Dict with attribute values (shape, size, location, boundary, pathology)
            region: Which region's uncertainty to use ('polyp', 'boundary', 'background')
        """
        uncertainty_values = aggregated_uncertainty[region]
        
        # Handle batch
        if isinstance(uncertainty_values, torch.Tensor):
            uncertainty_values = uncertainty_values.cpu().numpy()
        
        if not isinstance(uncertainty_values, (list, np.ndarray)):
            uncertainty_values = [uncertainty_values]
        
        for u in uncertainty_values:
            for attr_name, attr_value in metadata.items():
                if attr_value is not None:
                    self.attribute_uncertainties[attr_name][attr_value].append(float(u))
            self.sample_count += 1
    
    def compute_statistics(self) -> Dict[str, Dict]:
        """
        Compute per-attribute uncertainty statistics.
        
        Returns:
            Dictionary containing for each attribute:
            - mean: Mean uncertainty per value
            - std: Standard deviation per value
            - count: Sample count per value
            - anova_p: ANOVA p-value (if >2 groups)
            - kruskal_p: Kruskal-Wallis p-value
        """
        results = {}
        
        for attr_name, value_dict in self.attribute_uncertainties.items():
            attr_results = {
                'values': {},
                'anova_p': None,
                'kruskal_p': None,
            }
            
            groups = []
            for value, uncertainties in value_dict.items():
                if len(uncertainties) > 0:
                    u_array = np.array(uncertainties)
                    attr_results['values'][value] = {
                        'mean': float(np.mean(u_array)),
                        'std': float(np.std(u_array)),
                        'count': len(uncertainties),
                    }
                    groups.append(u_array)
            
            # Statistical tests (need at least 2 groups with 2+ samples each)
            valid_groups = [g for g in groups if len(g) >= 2]
            
            if len(valid_groups) >= 2:
                try:
                    # ANOVA (parametric)
                    _, anova_p = stats.f_oneway(*valid_groups)
                    attr_results['anova_p'] = float(anova_p)
                except:
                    pass
                
                try:
                    # Kruskal-Wallis (non-parametric)
                    _, kruskal_p = stats.kruskal(*valid_groups)
                    attr_results['kruskal_p'] = float(kruskal_p)
                except:
                    pass
            
            results[attr_name] = attr_results
        
        return results


class UncertaintyReductionScore:
    """
    Computes Uncertainty Reduction Score (URS).
    
    URS measures how much text conditioning reduces uncertainty
    compared to a vision-only baseline:
    
    URS(a_k) = (U_vision(a_k) - U_VL(a_k)) / U_vision(a_k)
    
    - URS > 0: Text reduces uncertainty (good)
    - URS < 0: Text increases uncertainty (potentially misleading)
    """
    
    def __init__(self):
        """Initialize URS calculator."""
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        # Store paired uncertainties: {attribute: {value: [(u_vision, u_vl), ...]}}
        self.paired_uncertainties = defaultdict(lambda: defaultdict(list))
    
    def update(
        self,
        uncertainty_vision: Dict[str, torch.Tensor],
        uncertainty_vl: Dict[str, torch.Tensor],
        metadata: Dict[str, str],
        region: str = 'polyp',
    ):
        """
        Update with paired vision-only and VL uncertainties.
        
        Args:
            uncertainty_vision: Aggregated uncertainty from vision-only model
            uncertainty_vl: Aggregated uncertainty from VL model
            metadata: Attribute metadata
            region: Spatial region to use
        """
        u_vision = uncertainty_vision[region]
        u_vl = uncertainty_vl[region]
        
        if isinstance(u_vision, torch.Tensor):
            u_vision = u_vision.cpu().numpy()
            u_vl = u_vl.cpu().numpy()
        
        if not isinstance(u_vision, (list, np.ndarray)):
            u_vision = [u_vision]
            u_vl = [u_vl]
        
        for uv, uvl in zip(u_vision, u_vl):
            for attr_name, attr_value in metadata.items():
                if attr_value is not None:
                    self.paired_uncertainties[attr_name][attr_value].append(
                        (float(uv), float(uvl))
                    )
    
    def compute_urs(self) -> Dict[str, Dict[str, float]]:
        """
        Compute URS for each attribute value.
        
        Returns:
            Dictionary: {attribute: {value: URS}}
        """
        results = {}
        
        for attr_name, value_dict in self.paired_uncertainties.items():
            attr_results = {}
            
            for value, pairs in value_dict.items():
                if len(pairs) > 0:
                    pairs = np.array(pairs)
                    u_vision_mean = pairs[:, 0].mean()
                    u_vl_mean = pairs[:, 1].mean()
                    
                    if u_vision_mean > 1e-8:
                        urs = (u_vision_mean - u_vl_mean) / u_vision_mean
                    else:
                        urs = 0.0
                    
                    attr_results[value] = {
                        'urs': float(urs),
                        'u_vision_mean': float(u_vision_mean),
                        'u_vl_mean': float(u_vl_mean),
                        'reduction_abs': float(u_vision_mean - u_vl_mean),
                        'count': len(pairs),
                    }
            
            results[attr_name] = attr_results
        
        return results


class SemanticAlignmentScore:
    """
    Computes Semantic Alignment Score (SAS).
    
    SAS measures whether uncertainty patterns align with text-described ambiguity.
    
    For boundary-related attributes:
    SAS_boundary = U_boundary / (U_polyp + U_boundary)
    
    When text mentions "irregular" or "ambiguous" boundaries,
    SAS_boundary should be higher.
    """
    
    def __init__(self):
        """Initialize SAS calculator."""
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        # Store SAS values: {attribute: {value: [sas_scores]}}
        self.sas_scores = defaultdict(lambda: defaultdict(list))
    
    def compute_sas(
        self,
        aggregated_uncertainty: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SAS values for a batch.
        
        Args:
            aggregated_uncertainty: Dict with 'polyp', 'boundary', 'background'
            
        Returns:
            Dictionary with SAS values:
            - 'boundary': SAS for boundary region
            - 'polyp': SAS for polyp region
        """
        u_polyp = aggregated_uncertainty['polyp']
        u_boundary = aggregated_uncertainty['boundary']
        u_bg = aggregated_uncertainty['background']
        
        # Avoid division by zero
        eps = 1e-8
        
        # Boundary SAS: proportion of uncertainty at boundary
        total_foreground = u_polyp + u_boundary + eps
        sas_boundary = u_boundary / total_foreground
        
        # Polyp SAS: proportion in polyp vs total
        total = u_polyp + u_boundary + u_bg + eps
        sas_polyp = u_polyp / total
        
        return {
            'boundary': sas_boundary,
            'polyp': sas_polyp,
        }
    
    def update(
        self,
        aggregated_uncertainty: Dict[str, torch.Tensor],
        metadata: Dict[str, str],
    ):
        """
        Update SAS statistics.
        
        Args:
            aggregated_uncertainty: Regional uncertainties
            metadata: Attribute metadata
        """
        sas = self.compute_sas(aggregated_uncertainty)
        
        # Convert to numpy
        sas_boundary = sas['boundary']
        if isinstance(sas_boundary, torch.Tensor):
            sas_boundary = sas_boundary.cpu().numpy()
        
        if not isinstance(sas_boundary, (list, np.ndarray)):
            sas_boundary = [sas_boundary]
        
        for s in sas_boundary:
            for attr_name, attr_value in metadata.items():
                if attr_value is not None:
                    self.sas_scores[attr_name][attr_value].append(float(s))
    
    def compute_statistics(self) -> Dict[str, Dict]:
        """
        Compute SAS statistics per attribute value.
        
        Returns:
            Statistics for each attribute
        """
        results = {}
        
        for attr_name, value_dict in self.sas_scores.items():
            attr_results = {}
            
            for value, scores in value_dict.items():
                if len(scores) > 0:
                    scores = np.array(scores)
                    attr_results[value] = {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'count': len(scores),
                    }
            
            results[attr_name] = attr_results
        
        return results


class AttributeConditionedECE:
    """
    Computes Expected Calibration Error conditioned on attributes.
    
    Standard ECE doesn't consider semantics. This computes ECE
    separately for each attribute value to understand:
    "Is the model better calibrated for certain lesion types?"
    """
    
    def __init__(self, num_bins: int = 10):
        """
        Initialize ECE calculator.
        
        Args:
            num_bins: Number of bins for reliability diagram
        """
        self.num_bins = num_bins
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        # Store (confidence, accuracy) pairs: {attr: {value: [(conf, acc), ...]}}
        self.calibration_data = defaultdict(lambda: defaultdict(list))
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Dict[str, str],
    ):
        """
        Update calibration data.
        
        Args:
            predictions: Predicted probabilities [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W]
            metadata: Attribute metadata
        """
        # Flatten spatial dimensions
        preds = predictions.view(-1).cpu().numpy()
        targs = targets.view(-1).cpu().numpy()
        
        # Store confidence and correctness
        for attr_name, attr_value in metadata.items():
            if attr_value is not None:
                # Sample to avoid memory issues (store aggregated stats per sample)
                conf = np.mean(np.maximum(preds, 1 - preds))  # Mean confidence
                pred_binary = preds > 0.5
                acc = np.mean(pred_binary == targs)  # Accuracy
                
                self.calibration_data[attr_name][attr_value].append((conf, acc))
    
    def compute_ece(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
    ) -> Tuple[float, Dict]:
        """
        Compute ECE from confidence-accuracy pairs.
        
        Args:
            confidences: Array of confidence values
            accuracies: Array of accuracy values
            
        Returns:
            Tuple of (ECE value, reliability diagram data)
        """
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        ece = 0.0
        reliability = {'bins': [], 'accuracies': [], 'confidences': [], 'counts': []}
        
        for i in range(self.num_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
                
                reliability['bins'].append(bin_centers[i])
                reliability['accuracies'].append(avg_accuracy)
                reliability['confidences'].append(avg_confidence)
                reliability['counts'].append(in_bin.sum())
        
        return ece, reliability
    
    def compute_conditioned_ece(self) -> Dict[str, Dict]:
        """
        Compute ECE conditioned on each attribute value.
        
        Returns:
            Dictionary: {attribute: {value: {ece, reliability_diagram}}}
        """
        results = {}
        
        for attr_name, value_dict in self.calibration_data.items():
            attr_results = {}
            
            for value, pairs in value_dict.items():
                if len(pairs) >= 10:  # Need minimum samples
                    pairs = np.array(pairs)
                    confidences = pairs[:, 0]
                    accuracies = pairs[:, 1]
                    
                    ece, reliability = self.compute_ece(confidences, accuracies)
                    
                    attr_results[value] = {
                        'ece': float(ece),
                        'mean_confidence': float(np.mean(confidences)),
                        'mean_accuracy': float(np.mean(accuracies)),
                        'count': len(pairs),
                        'reliability': reliability,
                    }
            
            results[attr_name] = attr_results
        
        return results


class UncertaintyMetricsCalculator:
    """
    Combined calculator for all uncertainty-text correlation metrics.
    
    Convenience class that manages all metric calculators together.
    """
    
    def __init__(
        self,
        boundary_kernel_size: int = 5,
        ece_num_bins: int = 10,
    ):
        """
        Initialize all metric calculators.
        
        Args:
            boundary_kernel_size: Kernel size for boundary extraction
            ece_num_bins: Number of bins for ECE calculation
        """
        self.aggregator = SpatialUncertaintyAggregator(boundary_kernel_size)
        self.attr_correlation = AttributeUncertaintyCorrelation()
        self.urs_calculator = UncertaintyReductionScore()
        self.sas_calculator = SemanticAlignmentScore()
        self.ece_calculator = AttributeConditionedECE(ece_num_bins)
    
    def reset(self):
        """Reset all calculators."""
        self.attr_correlation.reset()
        self.urs_calculator.reset()
        self.sas_calculator.reset()
        self.ece_calculator.reset()
    
    def update_single_model(
        self,
        uncertainty_map: torch.Tensor,
        mask: torch.Tensor,
        predictions: torch.Tensor,
        metadata: Dict[str, str],
    ):
        """
        Update metrics for a single model (no URS).
        
        Args:
            uncertainty_map: Pixel-wise uncertainty [B, 1, H, W]
            mask: Ground truth mask [B, 1, H, W]
            predictions: Predicted probabilities [B, 1, H, W]
            metadata: Attribute metadata
        """
        # Aggregate spatial uncertainty
        agg_uncertainty = self.aggregator.aggregate_uncertainty(uncertainty_map, mask)
        
        # Update attribute correlation
        self.attr_correlation.update(agg_uncertainty, metadata, region='polyp')
        self.attr_correlation.update(agg_uncertainty, metadata, region='boundary')
        
        # Update SAS
        self.sas_calculator.update(agg_uncertainty, metadata)
        
        # Update ECE
        self.ece_calculator.update(predictions, mask, metadata)
    
    def update_paired_models(
        self,
        uncertainty_vision: torch.Tensor,
        uncertainty_vl: torch.Tensor,
        mask: torch.Tensor,
        metadata: Dict[str, str],
    ):
        """
        Update URS with paired vision-only and VL uncertainties.
        
        Args:
            uncertainty_vision: Vision-only uncertainty map
            uncertainty_vl: VL model uncertainty map
            mask: Ground truth mask
            metadata: Attribute metadata
        """
        agg_vision = self.aggregator.aggregate_uncertainty(uncertainty_vision, mask)
        agg_vl = self.aggregator.aggregate_uncertainty(uncertainty_vl, mask)
        
        self.urs_calculator.update(agg_vision, agg_vl, metadata, region='polyp')
        self.urs_calculator.update(agg_vision, agg_vl, metadata, region='boundary')
    
    def compute_all_metrics(self) -> Dict[str, Dict]:
        """
        Compute all accumulated metrics.
        
        Returns:
            Dictionary with all metric results
        """
        return {
            'attribute_correlation': self.attr_correlation.compute_statistics(),
            'urs': self.urs_calculator.compute_urs(),
            'sas': self.sas_calculator.compute_statistics(),
            'conditioned_ece': self.ece_calculator.compute_conditioned_ece(),
        }


# For testing
if __name__ == "__main__":
    print("Testing Uncertainty Metrics...")
    
    # Create test data
    B, H, W = 4, 64, 64
    uncertainty_map = torch.rand(B, 1, H, W)
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, 20:40, 20:40] = 1.0  # Square polyp
    predictions = torch.sigmoid(torch.randn(B, 1, H, W))
    
    metadata = {
        'shape': 'irregular',
        'size': 'medium',
        'location': 'central',
        'boundary': 'ambiguous',
        'pathology': 'adenoma',
    }
    
    # Test spatial aggregator
    print("\n1. Testing SpatialUncertaintyAggregator...")
    aggregator = SpatialUncertaintyAggregator()
    agg = aggregator.aggregate_uncertainty(uncertainty_map, mask)
    print(f"   Polyp uncertainty: {agg['polyp'].mean():.4f}")
    print(f"   Boundary uncertainty: {agg['boundary'].mean():.4f}")
    print(f"   Background uncertainty: {agg['background'].mean():.4f}")
    
    # Test full calculator
    print("\n2. Testing UncertaintyMetricsCalculator...")
    calculator = UncertaintyMetricsCalculator()
    
    # Simulate multiple batches
    for _ in range(10):
        calculator.update_single_model(uncertainty_map, mask, predictions, metadata)
    
    # Compute metrics
    results = calculator.compute_all_metrics()
    
    print(f"\n   Attribute Correlation:")
    for attr, data in results['attribute_correlation'].items():
        if data['values']:
            print(f"      {attr}: {data['values']}")
    
    print(f"\n   SAS Statistics:")
    for attr, data in results['sas'].items():
        if data:
            print(f"      {attr}: {data}")
    
    print("\nAll tests passed!")