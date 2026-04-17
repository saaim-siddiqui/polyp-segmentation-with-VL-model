"""
Dataset Module for SUN Colonoscopy Video Database - Improved Balancing.

Provides multiple strategies for handling the frame imbalance:
1. Frame limiting per case
2. Weighted sampling
3. All frames with case-weighted loss

Recommended: Use WeightedCaseSampler with all frames.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from pathlib import Path
from collections import defaultdict
import random

from src.models.text_encoder_sun import SUNCaptionGenerator

class SUNPolypDataset(Dataset):
    """
    Dataset for SUN Colonoscopy Video Database with flexible balancing.
    """
        
    def __init__(
        self,
        data_root: str,
        metadata_path: Optional[str] = None,
        split: str = "train",
        image_size: int = 256,
        transform: Optional[Any] = None,
        use_augmentation: bool = True,
        caption_attributes: Optional[List[str]] = None,
        return_metadata: bool = True,
        # Balancing options
        frames_per_case: Optional[int] = None,  # None = use all frames
        balance_strategy: str = "none",  # "none", "limit", "oversample"
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.return_metadata = return_metadata
        self.frames_per_case = frames_per_case
        self.balance_strategy = balance_strategy
        
        # Load metadata
        self.metadata = self._load_metadata(metadata_path)
        
        # Build sample list
        self.samples = self._build_sample_list(None)
        
        # Apply split
        self.samples = self._apply_split(split)
        
        # Apply balancing if specified
        if balance_strategy == "limit" and frames_per_case:
            self.samples = self._limit_frames_per_case(self.samples, frames_per_case)
        elif balance_strategy == "oversample":
            self.samples = self._oversample_minority_cases(self.samples)
        
        # Setup caption generator
        self.caption_generator = SUNCaptionGenerator(
            attributes_to_include=caption_attributes
        )
        
        # Setup transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms(
                use_augmentation and split == "train"
            )
        
        # Build case-to-indices mapping for weighted sampling
        self._build_case_index_map()
        
        print(f"[{split}] Loaded {len(self.samples)} samples from {len(self.case_to_indices)} cases")
    
    def _build_case_index_map(self):
        """Build mapping from case_id to sample indices."""
        self.case_to_indices = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            self.case_to_indices[sample['case_id']].append(idx)
        
        # Compute weights for balanced sampling
        self.sample_weights = []
        for sample in self.samples:
            case_id = sample['case_id']
            case_size = len(self.case_to_indices[case_id])
            # Weight inversely proportional to case size
            weight = 1.0 / case_size
            self.sample_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(self.sample_weights)
        self.sample_weights = [w / total_weight * len(self.samples) for w in self.sample_weights]
    
    def _limit_frames_per_case(self, samples: List[Dict], max_frames: int) -> List[Dict]:
        """Limit frames per case with uniform sampling."""
        case_samples = defaultdict(list)
        for sample in samples:
            case_samples[sample['case_id']].append(sample)
        
        balanced = []
        for case_id, case_list in case_samples.items():
            if len(case_list) > max_frames:
                # Uniform sampling
                indices = np.linspace(0, len(case_list)-1, max_frames, dtype=int)
                case_list = [case_list[i] for i in indices]
            balanced.extend(case_list)
        
        return balanced
    
    def _oversample_minority_cases(self, samples: List[Dict]) -> List[Dict]:
        """Oversample frames from cases with fewer frames."""
        case_samples = defaultdict(list)
        for sample in samples:
            case_samples[sample['case_id']].append(sample)
        
        # Find max case size
        max_size = max(len(v) for v in case_samples.values())
        
        # Target: bring all cases to at least 50% of max
        target_size = max(max_size // 2, 100)
        
        balanced = []
        for case_id, case_list in case_samples.items():
            balanced.extend(case_list)
            # Oversample if needed
            if len(case_list) < target_size:
                extra_needed = target_size - len(case_list)
                extra = random.choices(case_list, k=extra_needed)
                balanced.extend(extra)
        
        return balanced
    
    def get_sample_weights(self) -> List[float]:
        """Get sample weights for WeightedRandomSampler."""
        return self.sample_weights
    
    def get_case_weights(self) -> Dict[int, float]:
        """Get weight per case (for loss weighting)."""
        case_weights = {}
        total_samples = len(self.samples)
        num_cases = len(self.case_to_indices)
        
        for case_id, indices in self.case_to_indices.items():
            # Each case should contribute equally
            # Weight = (total_samples / num_cases) / case_size
            case_weights[case_id] = (total_samples / num_cases) / len(indices)
        
        return case_weights

    def _load_metadata(self, metadata_path: Optional[str] = None) -> pd.DataFrame:
        """Load SUN database metadata."""
        if metadata_path and Path(metadata_path).exists():
            df = pd.read_csv(metadata_path)
            column_mapping = {
                'ID': 'case_id', 'Number of frames': 'num_frames',
                'Shape': 'shape', 'Size': 'size',
                'Location': 'location', 'Pathological diagnosis': 'pathology',
            }
            df = df.rename(columns=column_mapping)
            if 'case_id' in df.columns:
                df['case_id'] = df['case_id'].astype(int)
                df = df.set_index('case_id')
            return df
        return self._create_default_metadata()
    
    def _create_default_metadata(self) -> pd.DataFrame:
        """Create metadata from SUN database specification (Table 2)."""
        data = [
            {"case_id": 1, "num_frames": 527, "shape": "Is", "size": "6mm", "location": "Cecum", "pathology": "Low-grade adenoma"},
            {"case_id": 2, "num_frames": 1313, "shape": "Is", "size": "18mm", "location": "Rectum", "pathology": "High-grade adenoma"},
            {"case_id": 3, "num_frames": 292, "shape": "IIa", "size": "3mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 4, "num_frames": 80, "shape": "Is", "size": "4mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 5, "num_frames": 930, "shape": "IIa", "size": "3mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 6, "num_frames": 491, "shape": "IIa", "size": "3mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 7, "num_frames": 315, "shape": "IIa", "size": "6mm", "location": "Descending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 8, "num_frames": 256, "shape": "Isp", "size": "12mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 9, "num_frames": 136, "shape": "Is", "size": "4mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 10, "num_frames": 436, "shape": "IIa", "size": "3mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 11, "num_frames": 113, "shape": "IIa", "size": "5mm", "location": "Descending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 12, "num_frames": 538, "shape": "Is", "size": "5mm", "location": "Rectum", "pathology": "Low-grade adenoma"},
            {"case_id": 13, "num_frames": 479, "shape": "Is", "size": "5mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 14, "num_frames": 1183, "shape": "IIa", "size": "3mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 15, "num_frames": 487, "shape": "Is", "size": "5mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 16, "num_frames": 199, "shape": "Is", "size": "4mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 17, "num_frames": 304, "shape": "Is", "size": "4mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 18, "num_frames": 243, "shape": "Is", "size": "2mm", "location": "Sigmoid colon", "pathology": "Hyperplastic polyp"},
            {"case_id": 19, "num_frames": 96, "shape": "IIa", "size": "3mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 20, "num_frames": 3159, "shape": "IIa", "size": "3mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 21, "num_frames": 100, "shape": "IIa", "size": "3mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 22, "num_frames": 314, "shape": "IIa", "size": "2mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 23, "num_frames": 182, "shape": "Ip", "size": "12mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 24, "num_frames": 973, "shape": "Ip", "size": "15mm-", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 25, "num_frames": 338, "shape": "Is", "size": "7mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 26, "num_frames": 370, "shape": "Is", "size": "5mm", "location": "Descending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 27, "num_frames": 249, "shape": "Is", "size": "5mm", "location": "Ascending colon", "pathology": "Hyperplastic polyp"},
            {"case_id": 28, "num_frames": 195, "shape": "Is", "size": "2mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 29, "num_frames": 377, "shape": "Isp", "size": "13mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 30, "num_frames": 224, "shape": "IIa", "size": "4mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 31, "num_frames": 183, "shape": "Ip", "size": "12mm", "location": "Descending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 32, "num_frames": 981, "shape": "Ip", "size": "15mm-", "location": "Ascending colon", "pathology": "Traditional serrated adenoma"},
            {"case_id": 33, "num_frames": 594, "shape": "Is", "size": "5mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 34, "num_frames": 245, "shape": "Is", "size": "3mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 35, "num_frames": 1212, "shape": "Ip", "size": "15mm-", "location": "Sigmoid colon", "pathology": "High-grade adenoma"},
            {"case_id": 36, "num_frames": 815, "shape": "IIa", "size": "7mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 37, "num_frames": 448, "shape": "Is", "size": "7mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 38, "num_frames": 509, "shape": "Is", "size": "5mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 39, "num_frames": 713, "shape": "IIa", "size": "13mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 40, "num_frames": 159, "shape": "IIa", "size": "5mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 41, "num_frames": 108, "shape": "IIa", "size": "3mm", "location": "Rectum", "pathology": "Low-grade adenoma"},
            {"case_id": 42, "num_frames": 268, "shape": "Is", "size": "7mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 43, "num_frames": 260, "shape": "Isp", "size": "10mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 44, "num_frames": 745, "shape": "IIa", "size": "5mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 45, "num_frames": 383, "shape": "Is", "size": "3mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 46, "num_frames": 170, "shape": "IIa", "size": "2mm", "location": "Transverse colon", "pathology": "Hyperplastic polyp"},
            {"case_id": 47, "num_frames": 705, "shape": "Is", "size": "5mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 48, "num_frames": 176, "shape": "Is", "size": "3mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 49, "num_frames": 181, "shape": "IIa", "size": "3mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 50, "num_frames": 740, "shape": "Ip", "size": "10mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 51, "num_frames": 1737, "shape": "IIa(LST-NG)", "size": "15mm-", "location": "Cecum", "pathology": "Low-grade adenoma"},
            {"case_id": 52, "num_frames": 207, "shape": "IIa", "size": "6mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 53, "num_frames": 245, "shape": "Is", "size": "4mm", "location": "Rectum", "pathology": "Hyperplastic polyp"},
            {"case_id": 54, "num_frames": 345, "shape": "Is", "size": "4mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 55, "num_frames": 700, "shape": "Is", "size": "3mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 56, "num_frames": 248, "shape": "Is", "size": "4mm", "location": "Sigmoid colon", "pathology": "Hyperplastic polyp"},
            {"case_id": 57, "num_frames": 326, "shape": "Is", "size": "5mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 58, "num_frames": 267, "shape": "IIa", "size": "6mm", "location": "Transverse colon", "pathology": "Sessile serrated lesion"},
            {"case_id": 59, "num_frames": 646, "shape": "Isp", "size": "8mm", "location": "Sigmoid colon", "pathology": "Traditional serrated adenoma"},
            {"case_id": 60, "num_frames": 146, "shape": "IIa", "size": "8mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 61, "num_frames": 679, "shape": "Isp", "size": "6mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 62, "num_frames": 351, "shape": "Is", "size": "7mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 63, "num_frames": 632, "shape": "Is", "size": "7mm", "location": "Rectum", "pathology": "Invasive cancer (T1b)"},
            {"case_id": 64, "num_frames": 81, "shape": "IIa", "size": "3mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 65, "num_frames": 222, "shape": "IIa", "size": "3mm", "location": "Cecum", "pathology": "Low-grade adenoma"},
            {"case_id": 66, "num_frames": 1685, "shape": "Is", "size": "6mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 67, "num_frames": 191, "shape": "IIa", "size": "5mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 68, "num_frames": 1319, "shape": "Is", "size": "15mm-", "location": "Rectum", "pathology": "High-grade adenoma"},
            {"case_id": 69, "num_frames": 130, "shape": "IIa", "size": "3mm", "location": "Descending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 70, "num_frames": 264, "shape": "Ip", "size": "15mm-", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 71, "num_frames": 1021, "shape": "Is", "size": "4mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 72, "num_frames": 774, "shape": "Is", "size": "5mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 73, "num_frames": 1285, "shape": "Is", "size": "3mm", "location": "Cecum", "pathology": "Low-grade adenoma"},
            {"case_id": 74, "num_frames": 276, "shape": "Isp", "size": "5mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 75, "num_frames": 343, "shape": "Is", "size": "3mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 76, "num_frames": 343, "shape": "Is", "size": "3mm", "location": "Cecum", "pathology": "Low-grade adenoma"},
            {"case_id": 77, "num_frames": 215, "shape": "Is", "size": "4mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 78, "num_frames": 267, "shape": "Isp", "size": "12mm", "location": "Sigmoid colon", "pathology": "High-grade adenoma"},
            {"case_id": 79, "num_frames": 76, "shape": "Is", "size": "4mm", "location": "Descending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 80, "num_frames": 1192, "shape": "Is", "size": "10mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 81, "num_frames": 427, "shape": "Is", "size": "6mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 82, "num_frames": 111, "shape": "IIa", "size": "3mm", "location": "Sigmoid colon", "pathology": "Sessile serrated lesion"},
            {"case_id": 83, "num_frames": 795, "shape": "Isp", "size": "13mm", "location": "Rectum", "pathology": "Low-grade adenoma"},
            {"case_id": 84, "num_frames": 218, "shape": "Is", "size": "5mm", "location": "Descending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 85, "num_frames": 1393, "shape": "IIa", "size": "8mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 86, "num_frames": 257, "shape": "IIa", "size": "4mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 87, "num_frames": 454, "shape": "Is", "size": "3mm", "location": "Cecum", "pathology": "Low-grade adenoma"},
            {"case_id": 88, "num_frames": 249, "shape": "Is", "size": "4mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 89, "num_frames": 149, "shape": "Ip", "size": "5mm", "location": "Descending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 90, "num_frames": 479, "shape": "Is", "size": "10mm", "location": "Ascending colon", "pathology": "Sessile serrated lesion"},
            {"case_id": 91, "num_frames": 1061, "shape": "IIa", "size": "13mm", "location": "Ascending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 92, "num_frames": 391, "shape": "Is", "size": "7mm", "location": "Descending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 93, "num_frames": 452, "shape": "Is", "size": "7mm", "location": "Descending colon", "pathology": "Low-grade adenoma"},
            {"case_id": 94, "num_frames": 136, "shape": "Is", "size": "6mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 95, "num_frames": 606, "shape": "Isp", "size": "8mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 96, "num_frames": 301, "shape": "Is", "size": "5mm", "location": "Sigmoid colon", "pathology": "Hyperplastic polyp"},
            {"case_id": 97, "num_frames": 431, "shape": "IIa", "size": "15mm-", "location": "Cecum", "pathology": "Sessile serrated lesion"},
            {"case_id": 98, "num_frames": 170, "shape": "IIa", "size": "4mm", "location": "Transverse colon", "pathology": "Low-grade adenoma"},
            {"case_id": 99, "num_frames": 161, "shape": "Is", "size": "5mm", "location": "Sigmoid colon", "pathology": "Low-grade adenoma"},
            {"case_id": 100, "num_frames": 188, "shape": "IIa", "size": "3mm", "location": "Rectum", "pathology": "Hyperplastic polyp"},
        ]
        df = pd.DataFrame(data)
        return df.set_index('case_id')
    
    def _build_sample_list(self, case_ids: Optional[List[int]]) -> List[Dict]:
        """Build sample list - tries hierarchical then flat structure."""
        cases_to_process = case_ids if case_ids else list(self.metadata.index)
        
        positive_dir = self.data_root / "positive"
        if positive_dir.exists():
            return self._build_from_hierarchical(positive_dir, cases_to_process)
        else:
            return self._build_from_flat(cases_to_process)
    
    def _build_from_hierarchical(self, positive_dir: Path, cases: List[int]) -> List[Dict]:
        """Build from hierarchical directory structure."""
        samples = []
        for case_id in cases:
            for pattern in [f"case{case_id}", f"case_{case_id}", str(case_id)]:
                case_dir = positive_dir / pattern
                if case_dir.exists():
                    break
            else:
                continue

            images_dir = case_dir / "frames" if (case_dir / "frames").exists() else case_dir
            masks_dir = case_dir / "masks" if (case_dir / "masks").exists() else case_dir / "mask"
            
            if not masks_dir.exists():
                continue
            
            for img_path in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")):
                for ext in ['.png', '.jpg']:
                    mask_path = masks_dir / f"{img_path.stem}{ext}"
                    if mask_path.exists():
                        samples.append({
                            'image_path': str(img_path),
                            'mask_path': str(mask_path),
                            'case_id': case_id,
                            'frame_id': img_path.stem,
                        })
                        break
        return samples
    
    def _build_from_flat(self, cases: List[int]) -> List[Dict]:
        """Build from flat directory structure."""
        import re
        samples = []
        images_dir = self.data_root / "frames" if (self.data_root / "frames").exists() else self.data_root
        masks_dir = self.data_root / "masks" if (self.data_root / "masks").exists() else self.data_root
        
        for img_path in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")):
            # Extract case_id from filename
            match = re.search(r'case[_]?(\d+)', img_path.stem.lower())
            if not match:
                match = re.search(r'^(\d+)[_-]', img_path.stem)
            if not match:
                continue
            
            case_id = int(match.group(1))
            if case_id not in cases:
                continue
            
            for ext in ['.png', '.jpg']:
                mask_path = masks_dir / f"{img_path.stem}{ext}"
                if mask_path.exists():
                    samples.append({
                        'image_path': str(img_path),
                        'mask_path': str(mask_path),
                        'case_id': case_id,
                        'frame_id': img_path.stem,
                    })
                    break
        return samples
    
    def _apply_split(self, split: str) -> List[Dict]:
        """Apply case-level train/val/test split."""
        case_ids = sorted(set(s['case_id'] for s in self.samples))
        np.random.seed(42)
        np.random.shuffle(case_ids)
        
        n = len(case_ids)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        if split == "train":
            split_cases = set(case_ids[:train_end])
        elif split == "val":
            split_cases = set(case_ids[train_end:val_end])
        else:
            split_cases = set(case_ids[val_end:])
        
        return [s for s in self.samples if s['case_id'] in split_cases]
    
    def _get_default_transforms(self, use_augmentation: bool) -> A.Compose:
        """Get augmentation transforms."""
        if use_augmentation:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50), p=1),
                    A.GaussianBlur(blur_limit=(3, 5), p=1),
                ], p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        image = np.array(Image.open(sample['image_path']).convert('RGB'))
        mask = np.array(Image.open(sample['mask_path']).convert('L'))
        mask = (mask > 127).astype(np.float32)
        
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        case_id = sample['case_id']
        if case_id in self.metadata.index:
            meta_row = self.metadata.loc[case_id]
            metadata = {
                'shape': meta_row.get('shape'),
                'size': meta_row.get('size'),
                'location': meta_row.get('location'),
                'pathology': meta_row.get('pathology'),
            }
        else:
            metadata = {'shape': None, 'size': None, 'location': None, 'pathology': None}
        
        caption = self.caption_generator.generate_from_metadata(metadata)
        
        return {
            'image': image,
            'mask': mask.float(),
            'caption': caption,
            'metadata': metadata,
            'case_id': case_id,
            'frame_id': sample['frame_id'],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate function."""
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'mask': torch.stack([item['mask'] for item in batch]),
        'caption': [item['caption'] for item in batch],
        'metadata': [item['metadata'] for item in batch],
        'case_id': [item['case_id'] for item in batch],
        'frame_id': [item['frame_id'] for item in batch],
        'image_id': [item['frame_id'] for item in batch],
    }


def create_sun_dataloaders(
    data_root: str,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4,
    caption_attributes: Optional[List[str]] = None,
    balance_strategy: str = "weighted",  # "none", "limit", "weighted"
    frames_per_case: Optional[int] = None,  # Only for "limit" strategy
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders with proper balancing.
    
    Args:
        data_root: Root data directory
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of workers
        caption_attributes: Attributes for captions
        balance_strategy: 
            - "none": Use all frames, no balancing (not recommended)
            - "limit": Limit to frames_per_case per case
            - "weighted": Use all frames with WeightedRandomSampler (RECOMMENDED)
        frames_per_case: Max frames per case (only for "limit" strategy)
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    
    # Training dataset
    train_dataset = SUNPolypDataset(
        data_root=data_root,
        split="train",
        image_size=image_size,
        use_augmentation=True,
        caption_attributes=caption_attributes,
        frames_per_case=frames_per_case if balance_strategy == "limit" else None,
        balance_strategy="limit" if balance_strategy == "limit" else "none",
    )
    
    # Validation and test datasets (no balancing needed)
    val_dataset = SUNPolypDataset(
        data_root=data_root,
        split="val",
        image_size=image_size,
        use_augmentation=False,
        caption_attributes=caption_attributes,
    )
    
    test_dataset = SUNPolypDataset(
        data_root=data_root,
        split="test",
        image_size=image_size,
        use_augmentation=False,
        caption_attributes=caption_attributes,
    )
    
    # Create sampler for training
    if balance_strategy == "weighted":
        sampler = WeightedRandomSampler(
            weights=train_dataset.get_sample_weights(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"\nDataloader Summary:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_dataset.case_to_indices)} cases")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_dataset.case_to_indices)} cases")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_dataset.case_to_indices)} cases")
    print(f"  Balance strategy: {balance_strategy}")
    
    return train_loader, val_loader, test_loader