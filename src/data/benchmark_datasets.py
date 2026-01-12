"""
Benchmark Dataset Loaders for Kvasir-SEG and CVC-ClinicDB.

These datasets don't have metadata, so they're used for:
1. Cross-dataset generalization testing
2. Comparison with published literature results

No text conditioning is used for these datasets.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path


class BenchmarkPolypDataset(Dataset):
    """
    Generic dataset loader for benchmark polyp datasets without metadata.
    
    Supports:
    - Kvasir-SEG
    - CVC-ClinicDB
    - CVC-ColonDB
    - ETIS-Larib
    - Any dataset with images/ and masks/ folders
    """
    
    def __init__(
        self,
        data_root: str,
        image_size: int = 256,
        use_augmentation: bool = False,
        generic_caption: str = "A colorectal polyp is identified in the colonoscopy image.",
    ):
        """
        Initialize benchmark dataset.
        
        Args:
            data_root: Root directory containing images and masks
            image_size: Target image size
            use_augmentation: Whether to apply augmentation
            generic_caption: Generic caption to use (same for all images)
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.generic_caption = generic_caption
        
        # Find images and masks directories
        self.images_dir, self.masks_dir = self._find_directories()
        
        # Build sample list
        self.samples = self._build_sample_list()
        
        # Setup transforms
        self.transform = self._get_transforms(use_augmentation)
        
        print(f"Loaded {len(self.samples)} samples from {data_root}")
    
    def _find_directories(self) -> Tuple[Path, Path]:
        """Find images and masks directories with flexible naming."""
        
        # Common naming conventions
        image_names = ["images", "image", "Original", "original", "imgs", "img"]
        mask_names = ["masks", "mask", "Ground Truth", "groundtruth", "gt", "GT", 
                      "ground_truth", "segmentation", "annotations"]
        
        images_dir = None
        masks_dir = None
        
        # Try to find images directory
        for name in image_names:
            candidate = self.data_root / name
            if candidate.exists():
                images_dir = candidate
                break
        
        # If no images subfolder, check if images are directly in root
        if images_dir is None:
            # Check if root contains image files directly
            root_files = list(self.data_root.glob("*.jpg")) + list(self.data_root.glob("*.png"))
            if root_files:
                images_dir = self.data_root
        
        # Try to find masks directory
        for name in mask_names:
            candidate = self.data_root / name
            if candidate.exists():
                masks_dir = candidate
                break
        
        if images_dir is None:
            raise ValueError(f"Could not find images directory in {self.data_root}")
        if masks_dir is None:
            raise ValueError(f"Could not find masks directory in {self.data_root}")
        
        return images_dir, masks_dir
    
    def _build_sample_list(self) -> List[Dict]:
        """Build list of (image_path, mask_path) samples."""
        samples = []
        
        # Get all image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.images_dir.glob(ext)))
            image_files.extend(list(self.images_dir.glob(ext.upper())))
        
        image_files = sorted(set(image_files))
        
        for img_path in image_files:
            # Find corresponding mask
            mask_path = self._find_mask(img_path)
            
            if mask_path is not None:
                samples.append({
                    'image_path': str(img_path),
                    'mask_path': str(mask_path),
                    'image_id': img_path.stem,
                })
        
        return samples
    
    def _find_mask(self, image_path: Path) -> Optional[Path]:
        """Find mask file corresponding to an image."""
        stem = image_path.stem
        
        # Try different extensions and naming conventions
        mask_extensions = ['.png', '.jpg', '.tif', '.tiff', '.bmp']
        mask_prefixes = ['', 'mask_', 'gt_']
        mask_suffixes = ['', '_mask', '_gt', '_segmentation']
        
        for ext in mask_extensions:
            for prefix in mask_prefixes:
                for suffix in mask_suffixes:
                    # Try exact name
                    candidate = self.masks_dir / f"{prefix}{stem}{suffix}{ext}"
                    if candidate.exists():
                        return candidate
                    
                    # Try uppercase extension
                    candidate = self.masks_dir / f"{prefix}{stem}{suffix}{ext.upper()}"
                    if candidate.exists():
                        return candidate
        
        return None
    
    def _get_transforms(self, use_augmentation: bool) -> A.Compose:
        """Get preprocessing transforms."""
        if use_augmentation:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load image
        image = np.array(Image.open(sample['image_path']).convert('RGB'))
        
        # Load mask
        mask = np.array(Image.open(sample['mask_path']).convert('L'))
        
        # Binarize mask (handle different mask formats)
        if mask.max() > 1:
            mask = (mask > 127).astype(np.float32)
        else:
            mask = mask.astype(np.float32)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Ensure mask has channel dimension
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask.float(),
            'caption': self.generic_caption,
            'metadata': {},  # No metadata for benchmark datasets
            'image_id': sample['image_id'],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate function."""
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'mask': torch.stack([item['mask'] for item in batch]),
        'caption': [item['caption'] for item in batch],
        'metadata': [item['metadata'] for item in batch],
        'image_id': [item['image_id'] for item in batch],
    }


def create_benchmark_loader(
    data_root: str,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4,
    use_augmentation: bool = False,
    generic_caption: str = "A colorectal polyp is identified in the colonoscopy image.",
) -> DataLoader:
    """
    Create a dataloader for benchmark dataset.
    
    Args:
        data_root: Path to dataset root
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of workers
        use_augmentation: Whether to use augmentation
        generic_caption: Generic caption for all images
        
    Returns:
        DataLoader for the benchmark dataset
    """
    dataset = BenchmarkPolypDataset(
        data_root=data_root,
        image_size=image_size,
        use_augmentation=use_augmentation,
        generic_caption=generic_caption,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return loader


def create_kvasir_loader(
    data_root: str,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4,
) -> DataLoader:
    """
    Convenience function for Kvasir-SEG dataset.
    
    Expected structure:
    data_root/
    ├── images/
    │   ├── cju0qkwl35piu0993l0dewei2.jpg
    │   └── ...
    └── masks/
        ├── cju0qkwl35piu0993l0dewei2.jpg
        └── ...
    """
    return create_benchmark_loader(
        data_root=data_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        generic_caption="A colorectal polyp is identified in this colonoscopy image.",
    )


def create_cvc_clinicdb_loader(
    data_root: str,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4,
) -> DataLoader:
    """
    Convenience function for CVC-ClinicDB dataset.
    
    Expected structure:
    data_root/
    ├── Original/
    │   ├── 1.tif
    │   └── ...
    └── Ground Truth/
        ├── 1.tif
        └── ...
    """
    return create_benchmark_loader(
        data_root=data_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        generic_caption="A colorectal polyp is identified in this colonoscopy image.",
    )


# ============ Evaluation Functions ============

def evaluate_on_benchmark(
    model,
    benchmark_loader: DataLoader,
    device: str = "cuda",
    use_text: bool = False,
) -> Dict[str, float]:
    """
    Evaluate model on a benchmark dataset.
    
    Args:
        model: Trained model
        benchmark_loader: Benchmark dataset loader
        device: Device to use
        use_text: Whether to use text conditioning
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    model = model.to(device)
    
    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []
    
    with torch.no_grad():
        for batch in benchmark_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            captions = batch['caption']
            
            # Forward pass
            if use_text and hasattr(model, 'forward'):
                try:
                    outputs = model(images, texts=captions)
                except TypeError:
                    outputs = model(images)
            else:
                outputs = model(images)
            
            probs = outputs['probs']
            preds = (probs > 0.5).float()
            
            # Compute metrics for each sample
            for i in range(images.size(0)):
                pred = preds[i].view(-1)
                target = masks[i].view(-1)
                
                tp = (pred * target).sum()
                fp = (pred * (1 - target)).sum()
                fn = ((1 - pred) * target).sum()
                
                dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
                iou = tp / (tp + fp + fn + 1e-6)
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                
                all_dice.append(dice.item())
                all_iou.append(iou.item())
                all_precision.append(precision.item())
                all_recall.append(recall.item())
    
    return {
        'dice': {'mean': np.mean(all_dice), 'std': np.std(all_dice)},
        'iou': {'mean': np.mean(all_iou), 'std': np.std(all_iou)},
        'precision': {'mean': np.mean(all_precision), 'std': np.std(all_precision)},
        'recall': {'mean': np.mean(all_recall), 'std': np.std(all_recall)},
        'num_samples': len(all_dice),
    }


def run_benchmark_evaluation(
    model,
    benchmark_datasets: Dict[str, str],
    device: str = "cuda",
    batch_size: int = 8,
    image_size: int = 256,
) -> Dict[str, Dict]:
    """
    Run evaluation on multiple benchmark datasets.
    
    Args:
        model: Trained model
        benchmark_datasets: Dict mapping dataset name to data path
                           e.g., {"kvasir_seg": "./data/kvasir", "cvc_clinic": "./data/cvc"}
        device: Device to use
        batch_size: Batch size for evaluation
        image_size: Image size
        
    Returns:
        Dictionary of results per dataset
    """
    results = {}
    
    for name, path in benchmark_datasets.items():
        print(f"\nEvaluating on {name}...")
        
        loader = create_benchmark_loader(
            data_root=path,
            batch_size=batch_size,
            image_size=image_size,
        )
        
        metrics = evaluate_on_benchmark(
            model=model,
            benchmark_loader=loader,
            device=device,
            use_text=False,  # No text for benchmark datasets
        )
        
        results[name] = metrics
        
        print(f"  Dice: {metrics['dice']['mean']:.4f} ± {metrics['dice']['std']:.4f}")
        print(f"  IoU:  {metrics['iou']['mean']:.4f} ± {metrics['iou']['std']:.4f}")
    
    return results


# ============ Testing ============

if __name__ == "__main__":
    print("Benchmark Dataset Loader")
    print("=" * 60)
    
    # Test with a sample path (won't work without actual data)
    print("\nTo use this loader:")
    print("1. Download Kvasir-SEG from: https://datasets.simula.no/kvasir-seg/")
    print("2. Download CVC-ClinicDB from: https://polyp.grand-challenge.org/CVCClinicDB/")
    print("\n3. Use the loaders:")
    print("   kvasir_loader = create_kvasir_loader('./data/kvasir_seg')")
    print("   cvc_loader = create_cvc_clinicdb_loader('./data/cvc_clinicdb')")
