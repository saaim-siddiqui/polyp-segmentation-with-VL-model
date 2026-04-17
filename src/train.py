"""
Training Script for Vision-Language Segmentation.

Handles training, validation, and checkpointing.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Optional, Tuple

from .config import Config, get_config
from .models import create_model, VLSegmentationModel, VisionOnlyModel
from .data import create_sun_dataloaders
from .metrics import UncertaintyMetricsCalculator


# ============ Loss Functions ============

class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            predictions: Predicted probabilities [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W]
            
        Returns:
            Dice loss (1 - Dice score)
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss."""
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss."""
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(torch.sigmoid(logits), targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss."""
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce
        
        return focal_loss.mean()


def get_loss_function(loss_type: str, config: Config) -> nn.Module:
    """Get loss function based on config."""
    if loss_type == "dice":
        return DiceLoss()
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "dice_bce":
        return BCEDiceLoss(
            bce_weight=config.training.bce_weight,
            dice_weight=config.training.dice_weight,
        )
    elif loss_type == "focal":
        return FocalLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ============ Metrics ============

def compute_dice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute Dice score."""
    preds = (predictions > threshold).float()
    targets = targets.float()
    
    intersection = (preds * targets).sum()
    dice = (2. * intersection) / (preds.sum() + targets.sum() + 1e-6)
    
    return dice.item()


def compute_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute IoU (Jaccard index)."""
    preds = (predictions > threshold).float()
    targets = targets.float()
    
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = intersection / (union + 1e-6)
    
    return iou.item()


# ============ Training Functions ============

class Trainer:
    """Trainer class for VL Segmentation models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader,
        val_loader,
        device: str = "cuda",
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Configuration object
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        self.criterion = get_loss_function(
            config.training.loss_type, config
        )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        
        # Scheduler
        if config.training.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.training.num_epochs,
                eta_min=config.training.learning_rate * 0.01,
            )
        elif config.training.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
            )
        else:
            self.scheduler = None
        
        # Mixed precision
        self.scaler = GradScaler()
        self.use_amp = torch.cuda.is_available()
        
        # Tracking
        self.best_dice = 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'lr': [],
        }
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            captions = batch['caption']
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                # Forward pass
                if isinstance(self.model, VLSegmentationModel) and self.config.experiment.use_text:
                    outputs = self.model(images, texts=captions)
                else:
                    outputs = self.model(images)
                
                loss = self.criterion(outputs['logits'], masks)
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_samples = 0
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            captions = batch['caption']
            
            # Forward pass
            if isinstance(self.model, VLSegmentationModel) and self.config.experiment.use_text:
                outputs = self.model(images, texts=captions)
            else:
                outputs = self.model(images)
            
            loss = self.criterion(outputs['logits'], masks)
            total_loss += loss.item()
            
            # Compute metrics
            probs = outputs['probs']
            batch_dice = compute_dice_score(probs, masks)
            batch_iou = compute_iou(probs, masks)
            
            total_dice += batch_dice * images.size(0)
            total_iou += batch_iou * images.size(0)
            num_samples += images.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_dice = total_dice / num_samples
        avg_iou = total_iou / num_samples
        
        return avg_loss, avg_dice, avg_iou
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_dice': self.best_dice,
            'patience_counter': self.patience_counter,
            'config': self.config.__dict__,
            'history': self.history,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model with Dice: {self.best_dice:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_dice = checkpoint.get('best_dice', 0.0)
        self.patience_counter = checkpoint.get('patience_counter', 0)
        self.history = checkpoint.get('history', self.history)
        return checkpoint.get('epoch', 0) + 1
    
    def train(self, start_epoch: int = 0):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Training: {self.config.experiment.name}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(start_epoch, self.config.training.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_dice, val_iou = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_dice)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_dice)
            self.history['val_iou'].append(val_iou)
            self.history['lr'].append(current_lr)
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{self.config.training.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Dice: {val_dice:.4f}")
            print(f"  Val IoU: {val_iou:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Check for best model
            is_best = val_dice > self.best_dice
            if is_best:
                self.best_dice = val_dice
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.training.patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
        
        print(f"\nTraining complete. Best Dice: {self.best_dice:.4f}")
        
        # Save training history
        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


def train_model(
    config: Config,
    data_root: str,
    resume_from: Optional[str] = None,
) -> Dict:
    """
    Train a model with the given configuration.
    
    Args:
        config: Configuration object
        data_root: Path to data directory
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Training history dictionary
    """
    # Set seed
    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    
    # Set device
    device = config.experiment.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Create dataloaders
    train_loader, val_loader, _ = create_sun_dataloaders(
        data_root=data_root,
        batch_size=config.training.batch_size,
        image_size=config.data.image_size,
        num_workers=config.data.num_workers,
        caption_attributes=config.experiment.text_attributes if config.experiment.use_text else None,
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    model = create_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    
    # Resume if specified
    start_epoch = 0
    if resume_from is not None:
        start_epoch = trainer.load_checkpoint(resume_from)
        print(f"Resuming from epoch {start_epoch}")
    
    # Train
    history = trainer.train(start_epoch=start_epoch)
    
    return history


# ============ Main ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train VL Segmentation Model")
    parser.add_argument("--data_root", type=str, required=True, help="Path to data directory")
    parser.add_argument("--experiment", type=str, default="full", 
                       choices=["vision_only", "full", "text_shape_only", 
                               "text_size_only", "text_boundary_only", "text_location_only",
                               "text_pathology_only"],
                       help="Experiment configuration")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save checkpoints (overrides timestamped default)")
    
    args = parser.parse_args()
    
    # Get config
    config = get_config(args.experiment)
    
    # Override config with command line args
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    
    # Set checkpoint directory
    if args.output_dir is not None:
        config.training.checkpoint_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.training.checkpoint_dir = f"./checkpoints/{config.experiment.name}_{timestamp}"
    
    # Train
    history = train_model(config, args.data_root, args.resume)
    
    print("\nTraining finished!")