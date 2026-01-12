"""
Evaluation Script for Vision-Language Segmentation.

Computes segmentation metrics and uncertainty-text correlation metrics.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from .config import Config, get_config
from .models import create_model, VLSegmentationModel, VisionOnlyModel, UncertaintyEstimator
from .data import create_dataloaders
from .metrics import UncertaintyMetricsCalculator


class Evaluator:
    """Evaluator for VL Segmentation models with uncertainty analysis."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: str = "cuda",
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            config: Configuration object
            device: Device to use
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(
            num_mc_samples=config.uncertainty.num_mc_samples,
            uncertainty_type='both',
        )
        
        # Metrics calculator
        self.metrics_calculator = UncertaintyMetricsCalculator(
            boundary_kernel_size=config.uncertainty.boundary_kernel_size,
        )
    
    def compute_segmentation_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute standard segmentation metrics."""
        preds = (predictions > threshold).float()
        targets = targets.float()
        
        # Flatten
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        # True positives, etc.
        tp = (preds_flat * targets_flat).sum()
        fp = (preds_flat * (1 - targets_flat)).sum()
        fn = ((1 - preds_flat) * targets_flat).sum()
        tn = ((1 - preds_flat) * (1 - targets_flat)).sum()
        
        # Metrics
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)
        
        return {
            'dice': dice.item(),
            'iou': iou.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'specificity': specificity.item(),
        }
    
    @torch.no_grad()
    def evaluate_single_model(
        self,
        test_loader,
        compute_uncertainty: bool = True,
    ) -> Dict:
        """
        Evaluate a single model.
        
        Args:
            test_loader: Test data loader
            compute_uncertainty: Whether to compute uncertainty metrics
            
        Returns:
            Dictionary of evaluation results
        """
        self.model.eval()
        self.metrics_calculator.reset()
        
        all_metrics = []
        
        pbar = tqdm(test_loader, desc="Evaluating")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            captions = batch['caption']
            metadata_list = batch.get('metadata', [{}] * images.size(0))
            
            # Regular prediction
            if isinstance(self.model, VLSegmentationModel) and self.config.experiment.use_text:
                outputs = self.model(images, texts=captions)
            else:
                outputs = self.model(images)
            
            probs = outputs['probs']
            
            # Compute segmentation metrics
            batch_metrics = self.compute_segmentation_metrics(probs, masks)
            all_metrics.append(batch_metrics)
            
            # Compute uncertainty if requested
            if compute_uncertainty:
                # Get uncertainty via MC Dropout
                if isinstance(self.model, VLSegmentationModel) and self.config.experiment.use_text:
                    uncertainty_results = self.model.predict_with_uncertainty(
                        images, texts=captions
                    )
                else:
                    uncertainty_results = self.model.predict_with_uncertainty(images)
                
                # Use entropy as uncertainty measure
                uncertainty_map = uncertainty_results['entropy']
                
                # Update metrics for each sample
                for i in range(images.size(0)):
                    metadata = metadata_list[i] if i < len(metadata_list) else {}
                    self.metrics_calculator.update_single_model(
                        uncertainty_map[i:i+1],
                        masks[i:i+1],
                        probs[i:i+1],
                        metadata,
                    )
        
        # Aggregate segmentation metrics
        seg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            seg_metrics[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            }
        
        results = {
            'segmentation': seg_metrics,
        }
        
        # Add uncertainty metrics
        if compute_uncertainty:
            results['uncertainty'] = self.metrics_calculator.compute_all_metrics()
        
        return results
    
    @torch.no_grad()
    def evaluate_paired_models(
        self,
        vision_model: nn.Module,
        vl_model: nn.Module,
        test_loader,
    ) -> Dict:
        """
        Evaluate paired vision-only and VL models for URS computation.
        
        Args:
            vision_model: Vision-only baseline model
            vl_model: Vision-language model
            test_loader: Test data loader
            
        Returns:
            Dictionary with comparative results
        """
        vision_model.eval()
        vl_model.eval()
        
        vision_model = vision_model.to(self.device)
        vl_model = vl_model.to(self.device)
        
        self.metrics_calculator.reset()
        
        vision_metrics = []
        vl_metrics = []
        
        pbar = tqdm(test_loader, desc="Paired Evaluation")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            captions = batch['caption']
            metadata_list = batch.get('metadata', [{}] * images.size(0))
            
            # Vision-only predictions
            vision_outputs = vision_model(images)
            vision_probs = vision_outputs['probs']
            
            # VL predictions
            vl_outputs = vl_model(images, texts=captions)
            vl_probs = vl_outputs['probs']
            
            # Segmentation metrics
            vision_metrics.append(self.compute_segmentation_metrics(vision_probs, masks))
            vl_metrics.append(self.compute_segmentation_metrics(vl_probs, masks))
            
            # Uncertainty estimation
            vision_uncertainty = vision_model.predict_with_uncertainty(images)
            vl_uncertainty = vl_model.predict_with_uncertainty(images, texts=captions)
            
            # Update URS metrics
            for i in range(images.size(0)):
                metadata = metadata_list[i] if i < len(metadata_list) else {}
                self.metrics_calculator.update_paired_models(
                    vision_uncertainty['entropy'][i:i+1],
                    vl_uncertainty['entropy'][i:i+1],
                    masks[i:i+1],
                    metadata,
                )
                
                # Also update single model metrics for VL
                self.metrics_calculator.update_single_model(
                    vl_uncertainty['entropy'][i:i+1],
                    masks[i:i+1],
                    vl_probs[i:i+1],
                    metadata,
                )
        
        # Aggregate metrics
        def aggregate_metrics(metrics_list):
            result = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list]
                result[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                }
            return result
        
        results = {
            'vision_only': aggregate_metrics(vision_metrics),
            'vl_model': aggregate_metrics(vl_metrics),
            'uncertainty_metrics': self.metrics_calculator.compute_all_metrics(),
        }
        
        # Compute improvement
        results['improvement'] = {
            'dice': results['vl_model']['dice']['mean'] - results['vision_only']['dice']['mean'],
            'iou': results['vl_model']['iou']['mean'] - results['vision_only']['iou']['mean'],
        }
        
        return results


def visualize_results(
    results: Dict,
    output_dir: str,
):
    """
    Generate visualizations for evaluation results.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if uncertainty_metrics exists (from paired) or uncertainty (from single)
    uncertainty = results.get('uncertainty_metrics', results.get('uncertainty', {}))
    
    # 1. Attribute-Uncertainty Box Plot
    if 'attribute_correlation' in uncertainty:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        attr_data = uncertainty['attribute_correlation']
        for idx, (attr_name, attr_values) in enumerate(attr_data.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            if attr_values.get('values'):
                names = list(attr_values['values'].keys())
                means = [attr_values['values'][n]['mean'] for n in names]
                stds = [attr_values['values'][n]['std'] for n in names]
                
                ax.bar(names, means, yerr=stds, capsize=5, alpha=0.7)
                ax.set_title(f'{attr_name.capitalize()} vs Uncertainty')
                ax.set_ylabel('Mean Uncertainty')
                ax.tick_params(axis='x', rotation=45)
                
                # Add significance indicator
                if attr_values.get('kruskal_p') is not None:
                    p_val = attr_values['kruskal_p']
                    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                    ax.set_xlabel(f'p={p_val:.4f} ({sig})')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'attribute_uncertainty_correlation.png', dpi=150)
        plt.close()
    
    # 2. URS heatmap
    if 'urs' in uncertainty:
        urs_data = uncertainty['urs']
        
        # Prepare data for heatmap
        all_attrs = list(urs_data.keys())
        all_values = set()
        for attr_vals in urs_data.values():
            all_values.update(attr_vals.keys())
        all_values = sorted(list(all_values))
        
        if all_attrs and all_values:
            matrix = np.zeros((len(all_attrs), len(all_values)))
            matrix[:] = np.nan
            
            for i, attr in enumerate(all_attrs):
                for j, val in enumerate(all_values):
                    if val in urs_data[attr]:
                        matrix[i, j] = urs_data[attr][val].get('urs', np.nan)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(
                matrix,
                xticklabels=all_values,
                yticklabels=all_attrs,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                center=0,
                ax=ax,
                mask=np.isnan(matrix),
            )
            ax.set_title('Uncertainty Reduction Score (URS) by Attribute')
            ax.set_xlabel('Attribute Value')
            ax.set_ylabel('Attribute')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'urs_heatmap.png', dpi=150)
            plt.close()
    
    # 3. SAS comparison
    if 'sas' in uncertainty:
        sas_data = uncertainty['sas']
        
        # Focus on boundary attribute
        if 'boundary' in sas_data:
            boundary_sas = sas_data['boundary']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            names = list(boundary_sas.keys())
            means = [boundary_sas[n]['mean'] for n in names]
            stds = [boundary_sas[n]['std'] for n in names]
            
            colors = ['green' if 'clear' in n else 'orange' if 'ambiguous' in n else 'red' 
                     for n in names]
            
            ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
            ax.set_title('Semantic Alignment Score by Boundary Type')
            ax.set_ylabel('SAS (Boundary Uncertainty Proportion)')
            ax.set_xlabel('Boundary Attribute')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'sas_boundary.png', dpi=150)
            plt.close()
    
    # 4. Model Comparison (if paired results)
    if 'vision_only' in results and 'vl_model' in results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        metrics = ['dice', 'iou']
        x = np.arange(len(metrics))
        width = 0.35
        
        vision_means = [results['vision_only'][m]['mean'] for m in metrics]
        vision_stds = [results['vision_only'][m]['std'] for m in metrics]
        vl_means = [results['vl_model'][m]['mean'] for m in metrics]
        vl_stds = [results['vl_model'][m]['std'] for m in metrics]
        
        ax = axes[0]
        ax.bar(x - width/2, vision_means, width, yerr=vision_stds, label='Vision-Only', capsize=5)
        ax.bar(x + width/2, vl_means, width, yerr=vl_stds, label='VL Model', capsize=5)
        ax.set_ylabel('Score')
        ax.set_title('Segmentation Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Improvement
        ax = axes[1]
        improvements = [results['improvement'][m] for m in metrics]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        ax.bar(metrics, improvements, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Improvement')
        ax.set_title('VL Model Improvement over Vision-Only')
        ax.set_xticklabels([m.upper() for m in metrics])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png', dpi=150)
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def visualize_uncertainty_maps(
    model: nn.Module,
    test_loader,
    output_dir: str,
    num_samples: int = 10,
    device: str = "cuda",
    use_text: bool = True,
):
    """
    Generate uncertainty map visualizations.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        output_dir: Output directory
        num_samples: Number of samples to visualize
        device: Device to use
        use_text: Whether model uses text
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    model = model.to(device)
    
    samples_saved = 0
    
    for batch in test_loader:
        if samples_saved >= num_samples:
            break
        
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        captions = batch['caption']
        image_ids = batch['image_id']
        
        # Get predictions with uncertainty
        with torch.no_grad():
            if use_text and isinstance(model, VLSegmentationModel):
                uncertainty_results = model.predict_with_uncertainty(images, texts=captions)
            else:
                uncertainty_results = model.predict_with_uncertainty(images)
        
        # Visualize each sample
        for i in range(min(images.size(0), num_samples - samples_saved)):
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            
            # Original image (denormalize)
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            axes[0].imshow(img)
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(masks[i, 0].cpu().numpy(), cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            pred = uncertainty_results['prediction'][i, 0].cpu().numpy()
            axes[2].imshow(pred, cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            # Entropy uncertainty
            entropy = uncertainty_results['entropy'][i, 0].cpu().numpy()
            im = axes[3].imshow(entropy, cmap='hot')
            axes[3].set_title('Entropy Uncertainty')
            axes[3].axis('off')
            plt.colorbar(im, ax=axes[3], fraction=0.046)
            
            # Variance uncertainty
            variance = uncertainty_results['variance'][i, 0].cpu().numpy()
            im = axes[4].imshow(variance, cmap='hot')
            axes[4].set_title('Variance Uncertainty')
            axes[4].axis('off')
            plt.colorbar(im, ax=axes[4], fraction=0.046)
            
            # Add caption as suptitle
            if use_text:
                caption_text = captions[i] if len(captions[i]) <= 100 else captions[i][:100] + "..."
                fig.suptitle(f"Caption: {caption_text}", fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'uncertainty_sample_{image_ids[i]}.png', dpi=150)
            plt.close()
            
            samples_saved += 1
    
    print(f"Saved {samples_saved} uncertainty visualizations to {output_dir}")


def run_full_evaluation(
    checkpoint_path: str,
    data_root: str,
    output_dir: str,
    config: Config = None,
    compare_vision_only: bool = True,
    vision_checkpoint_path: str = None,
):
    """
    Run full evaluation pipeline.
    
    Args:
        checkpoint_path: Path to VL model checkpoint
        data_root: Path to data directory
        output_dir: Directory to save results
        config: Configuration (loaded from checkpoint if None)
        compare_vision_only: Whether to compare with vision-only baseline
        vision_checkpoint_path: Path to vision-only checkpoint
        
    Returns:
        Evaluation results dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if config is None:
        config = get_config("full")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create test loader
    _, _, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=config.training.batch_size,
        image_size=config.data.image_size,
        num_workers=config.data.num_workers,
        caption_attributes=config.experiment.text_attributes if config.experiment.use_text else None,
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Load VL model
    vl_model = create_model(config)
    vl_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = Evaluator(vl_model, config, device)
    
    if compare_vision_only and vision_checkpoint_path:
        # Load vision-only model
        vision_config = get_config("vision_only")
        vision_model = create_model(vision_config)
        vision_checkpoint = torch.load(vision_checkpoint_path, map_location='cpu')
        vision_model.load_state_dict(vision_checkpoint['model_state_dict'])
        
        # Paired evaluation
        results = evaluator.evaluate_paired_models(vision_model, vl_model, test_loader)
    else:
        # Single model evaluation
        results = evaluator.evaluate_single_model(test_loader, compute_uncertainty=True)
    
    # Save results
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")
    
    # Generate visualizations
    visualize_results(results, output_dir / 'plots')
    
    # Generate uncertainty maps
    visualize_uncertainty_maps(
        vl_model, test_loader, output_dir / 'uncertainty_maps',
        num_samples=10, device=device, use_text=config.experiment.use_text
    )
    
    return results


# ============ Main ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate VL Segmentation Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory")
    parser.add_argument("--vision_checkpoint", type=str, default=None, help="Vision-only checkpoint for comparison")
    
    args = parser.parse_args()
    
    results = run_full_evaluation(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        compare_vision_only=args.vision_checkpoint is not None,
        vision_checkpoint_path=args.vision_checkpoint,
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    if 'segmentation' in results:
        print("\nSegmentation Metrics:")
        for metric, values in results['segmentation'].items():
            print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    if 'vl_model' in results:
        print("\nVL Model Performance:")
        for metric, values in results['vl_model'].items():
            print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
        
        print("\nImprovement over Vision-Only:")
        for metric, value in results['improvement'].items():
            sign = '+' if value > 0 else ''
            print(f"  {metric}: {sign}{value:.4f}")
    
    print("\nEvaluation complete!")