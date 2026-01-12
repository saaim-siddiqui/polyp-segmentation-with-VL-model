#!/usr/bin/env python
"""
Quick Start Script for VL Polyp Segmentation.

This script helps verify your installation and data setup before running full experiments.

Usage:
    python quick_start.py --data_root ./data/sun --check_only
    python quick_start.py --data_root ./data/sun --train_mini
"""

import argparse
import sys
from pathlib import Path


def check_installation():
    """Check if all required packages are installed."""
    print("Checking installation...")
    
    required = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "HuggingFace Transformers"),
        ("albumentations", "Albumentations"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("tqdm", "tqdm"),
        ("cv2", "OpenCV"),
    ]
    
    all_good = True
    for module, name in required:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            all_good = False
    
    return all_good


def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU...")
    
    import torch
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ GPU Available: {device_name}")
        print(f"  ✓ GPU Memory: {memory:.1f} GB")
        return True
    else:
        print("  ⚠ No GPU available. Training will be slow on CPU.")
        return False


def check_models():
    """Check if models can be loaded."""
    print("\nChecking model imports...")
    
    try:
        from src.models import VLSegmentationModel, VisionOnlyModel, create_model
        from src.config import get_config
        print("  ✓ Model imports successful")
        
        # Try creating a model
        config = get_config("vision_only")
        model = create_model(config)
        print(f"  ✓ Vision model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_data(data_root: str):
    """Check if data is properly organized."""
    print(f"\nChecking data at {data_root}...")
    
    data_path = Path(data_root)
    
    if not data_path.exists():
        print(f"  ✗ Data directory not found: {data_root}")
        print("    Please download SUN database and extract to this location.")
        return False
    
    # Check for positive cases directory
    positive_dir = data_path / "positive"
    if not positive_dir.exists():
        # Try flat structure
        images_dir = data_path / "frames"
        if images_dir.exists():
            print(f"  ✓ Found flat structure with frames/ directory")
        else:
            print(f"  ⚠ Expected structure not found")
            print(f"    Expected: {data_root}/positive/case1/frames/ or {data_root}/frames/")
            return False
    else:
        # Count cases
        cases = list(positive_dir.glob("case*"))
        print(f"  ✓ Found {len(cases)} case directories")
        
        if len(cases) > 0:
            # Check first case structure
            first_case = cases[0]
            images = list((first_case / "frames").glob("*")) if (first_case / "frames").exists() else []
            masks = list((first_case / "masks").glob("*")) if (first_case / "masks").exists() else []
            print(f"  ✓ First case has {len(images)} frames and {len(masks)} masks")
    
    return True


def run_mini_training(data_root: str):
    """Run a minimal training loop to verify everything works."""
    print("\nRunning mini training test (1 epoch, 10 steps)...")
    
    import torch
    from src.config import get_config
    from src.models import create_model
    from src.data import create_sun_dataloaders
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create small config
    config = get_config("vision_only")
    config.training.epochs = 1
    config.training.batch_size = 2
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Try to load data
    try:
        train_loader, val_loader, _ = create_sun_dataloaders(
            data_root=data_root,
            batch_size=2,
            image_size=256,
            num_workers=0,  # For testing
            balance_strategy="limit",
            frames_per_case=10,
        )
        print(f"  ✓ Data loaded: {len(train_loader.dataset)} training samples")
    except Exception as e:
        print(f"  ✗ Failed to load data: {e}")
        return False
    
    # Run a few training steps
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    model.train()
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break
        
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs['logits'], masks)
        loss.backward()
        optimizer.step()
        
        if i % 5 == 0:
            print(f"    Step {i}: Loss = {loss.item():.4f}")
    
    print("  ✓ Mini training completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Quick start verification")
    parser.add_argument("--data_root", type=str, default="./data/sun",
                        help="Path to SUN database")
    parser.add_argument("--check_only", action="store_true",
                        help="Only check installation, don't run training")
    parser.add_argument("--train_mini", action="store_true",
                        help="Run a mini training loop")
    
    args = parser.parse_args()
    
    print("="*60)
    print("VL Polyp Segmentation - Quick Start Verification")
    print("="*60)
    
    # Check installation
    if not check_installation():
        print("\n✗ Installation check failed. Please install missing packages:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Check GPU
    check_gpu()
    
    # Check models
    if not check_models():
        print("\n✗ Model check failed. Check your installation.")
        sys.exit(1)
    
    # Check data
    data_ok = check_data(args.data_root)
    
    if args.check_only:
        print("\n" + "="*60)
        if data_ok:
            print("✓ All checks passed! You're ready to train.")
        else:
            print("⚠ Setup your data directory before training.")
        print("="*60)
        return
    
    if args.train_mini:
        if not data_ok:
            print("\n✗ Cannot run training without data. Setup data first.")
            sys.exit(1)
        
        if run_mini_training(args.data_root):
            print("\n" + "="*60)
            print("✓ Everything is working! You're ready for full training.")
            print("  Run: bash scripts/run_all_experiments.sh")
            print("="*60)
        else:
            print("\n✗ Mini training failed. Check errors above.")
            sys.exit(1)


if __name__ == "__main__":
    main()
