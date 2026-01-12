# Vision-Language Polyp Segmentation with Uncertainty Estimation

A research project investigating how semantic language cues improve segmentation quality and uncertainty interpretability in medical image segmentation.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Data Preparation](#data-preparation)
5. [Running Experiments](#running-experiments)
6. [Expected Results](#expected-results)
7. [Citation](#citation)

---

## Project Overview

### Research Claim

> **Semantic language cues improve both segmentation quality and interpretability of uncertainty in medical image segmentation.**

### Key Contributions

1. **Uncertainty Reduction Score (URS)**: Quantifies how text conditioning reduces predictive uncertainty
2. **Semantic Alignment Score (SAS)**: Measures whether uncertainty patterns align with text-described ambiguity  
3. **Attribute-Conditioned ECE**: Calibration analysis conditioned on lesion attributes

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Vision-Language Segmentation                    │
└─────────────────────────────────────────────────────────────────┘

    Image ────► Vision Encoder ────► Cross-Attention ────► Decoder ────► Segmentation
                  (ResNet34)             Fusion            (U-Net)      + Uncertainty
                                          ▲                   │
                                          │                   ▼
    Caption ──► Text Encoder ─────────────┘            MC Dropout
                (PubMedBERT)                           (T=10 samples)
```

---

## Installation

### Step 1: Clone and Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd vl_polyp_segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Optional: For CUDA acceleration
# torch with CUDA - install separately based on your CUDA version
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python -c "from src.models import VLSegmentationModel; print('✓ Installation successful!')"
```
For Complete Check

```bash
python quick_start.py --data_root ./data/sun --check_only    
```

---

## 📁 Project Structure

```
vl_polyp_segmentation/
│
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
│
├── configs/                       # Configuration files (Currently all configurations exist in src/config.py Apply this approach later)
│   ├── default.yaml               # Default hyperparameters
│   ├── vision_only.yaml           # Vision-only baseline
│   ├── full_vl.yaml               # Full VL model
│   └── ablations/                 # Ablation configs
│       ├── shape_only.yaml
│       ├── size_only.yaml
│       ├── location_only.yaml
│       └── pathology_only.yaml
│
├── data/                           # Data directory (create this)
│   ├── sun/                       # SUN Database (primary)
│   │   ├── positive/
│   │   │   ├── case1/
│   │   │   │   ├── frames/
│   │   │   │   └── masks/
│   │   │   ├── case2/
│   │   │   └── ...
│   │   └── metadata.csv           # Optional (built-in if missing)
│   │
│   ├── kvasir_seg/                # Kvasir-SEG (benchmark)
│   │   ├── frames/
│   │   └── masks/
│   │
│   └── cvc_clinicdb/              # CVC-ClinicDB (benchmark)
│       ├── Original/
│       └── Ground Truth/
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── config.py                  # Configuration management
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   │
│   ├── models/                    # Model implementations
│   │   ├── __init__.py
│   │   ├── vision_encoder.py     # ResNet backbone
│   │   ├── text_encoder.py       # PubMedBERT + caption generator
│   │   ├── fusion.py             # Cross-attention fusion
│   │   ├── decoder.py            # U-Net decoder with MC Dropout
│   │   └── vl_segmentation.py    # Main VL model
│   │
│   ├── data/                      # Data loading
│   │   ├── __init__.py
│   │   ├── dataset_sun.py        # SUN database loader
│   │   └── benchmark_datasets.py # Kvasir/CVC loaders
│   │
│   └── metrics/                   # Evaluation metrics
│       ├── __init__.py
│       └── uncertainty_metrics.py # URS, SAS, ECE
│
├── scripts/                       # Utility scripts
│   ├── download_data.sh           # Download benchmark datasets
│   ├── run_all_experiments.sh     # Run all experiments
│   └── generate_figures.py        # Generate paper figures
│
├── checkpoints/                    # Saved models (created during training)
│   ├── vision_only/
│   ├── full_vl/
│   └── ablations/
│
├── results/                        # Experiment results (created during eval)
│   ├── tables/
│   ├── figures/
│   └── logs/
│
└── notebooks/                      # Jupyter notebooks (optional)
    ├── 01_data_exploration.ipynb
    ├── 02_model_analysis.ipynb
    └── 03_results_visualization.ipynb
```

---

## Data Preparation

### Primary Dataset: SUN Database

1. **Request access** from: http://amed8k.sundatabase.org/
2. **Download** and extract to `data/sun/`
3. **Organize** in hierarchical structure:

```
data/sun/
├── positive/
│   ├── case1/
│   │   ├── frames/
│   │   │   ├── frame_0001.jpg
│   │   │   ├── frame_0002.jpg
│   │   │   └── ...
│   │   └── masks/
│   │       ├── frame_0001.png
│   │       └── ...
│   ├── case2/
│   └── ... (100 cases total)
```

**Note:** Metadata is built-in from Table 2 of the SUN paper. No CSV required.

### Benchmark Datasets

#### Kvasir-SEG
```bash
# Download from: https://datasets.simula.no/kvasir-seg/
# Extract to data/kvasir_seg/

data/kvasir_seg/
├── images/
│   ├── cju0qkwl35piu0993l0dewei2.jpg
│   └── ... (1000 images)
└── masks/
    ├── cju0qkwl35piu0993l0dewei2.jpg
    └── ...
```

#### CVC-ClinicDB
```bash
# Download from: https://polyp.grand-challenge.org/CVCClinicDB/
# Extract to data/cvc_clinicdb/

data/cvc_clinicdb/
├── Original/
│   ├── 1.tif
│   └── ... (612 images)
└── Ground Truth/
    ├── 1.tif
    └── ...
```

---

## Running Experiments

### Quick Start (Single Command)

```bash
# Run everything
bash scripts/run_all_experiments.sh
```

### Step-by-Step Execution

#### Step 1: Train Vision-Only Baseline

```bash
python -m src.train \
    --data_root ./data/sun \
    --experiment vision_only \
    --epochs 100 \
    --batch_size 8 \
    --output_dir ./checkpoints/vision_only
```

**Expected time:** ~5-6 hours on single GPU

#### Step 2: Train Full VL Model

```bash
python -m src.train \
    --data_root ./data/sun \
    --experiment full \
    --epochs 100 \
    --batch_size 8 \
    --output_dir ./checkpoints/full_vl
```

**Expected time:** ~3-5 hours on single GPU

#### Step 3: Run Ablation Experiments

```bash
# Shape only
python -m src.train \
    --data_root ./data/sun \
    --experiment text_shape_only \
    --epochs 100 \
    --output_dir ./checkpoints/ablations/shape_only

# Size only
python -m src.train \
    --data_root ./data/sun \
    --experiment text_size_only \
    --epochs 100 \
    --output_dir ./checkpoints/ablations/size_only

# Location only
python -m src.train \
    --data_root ./data/sun \
    --experiment text_location_only \
    --epochs 100 \
    --output_dir ./checkpoints/ablations/location_only

# Pathology only
python -m src.train \
    --data_root ./data/sun \
    --experiment text_pathology_only \
    --epochs 100 \
    --output_dir ./checkpoints/ablations/pathology_only
```

#### Step 4: Evaluate on SUN Database (with Uncertainty Analysis)

```bash
python -m src.evaluate \
    --checkpoint ./checkpoints/full_vl/best.pt \
    --vision_checkpoint ./checkpoints/vision_only/best.pt \
    --data_root ./data/sun \
    --output_dir ./results/sun_evaluation
```

**This generates:**
- Segmentation metrics (Dice, IoU)
- Uncertainty metrics (URS, SAS, ECE)
- Visualization plots

#### Step 5: Evaluate on Benchmark Datasets

```bash
python -m src.evaluate_benchmarks \
    --checkpoint ./checkpoints/full_vl/best.pt \
    --kvasir_path ./data/kvasir_seg \
    --cvc_path ./data/cvc_clinicdb \
    --output_dir ./results/benchmarks
```

---

## Expected Results

### Table 1: Main Results on SUN Database

| Model | Dice ↑ | IoU ↑ | URS ↑ | SAS | ECE ↓ |
|-------|--------|-------|-------|-----|-------|
<!-- | Vision-Only | ~0.84 | ~0.74 | - | ~0.31 | ~0.12 |
| VL (Shape) | ~0.85 | ~0.75 | ~0.08 | ~0.34 | ~0.11 |
| VL (Size) | ~0.85 | ~0.75 | ~0.06 | ~0.33 | ~0.11 |
| VL (Location) | ~0.84 | ~0.74 | ~0.03 | ~0.32 | ~0.11 |
| VL (Pathology) | ~0.85 | ~0.75 | ~0.05 | ~0.33 | ~0.10 |
| **VL (Full)** | **~0.87** | **~0.78** | **~0.15** | **~0.42** | **~0.08** | -->

### Table 2: Cross-Dataset Generalization

| Train → Test | Dice | IoU |
|--------------|------|-----|
<!-- | SUN → SUN | ~0.87 | ~0.78 |
| SUN → Kvasir-SEG | ~0.80 | ~0.70 |
| SUN → CVC-ClinicDB | ~0.78 | ~0.68 | -->

### Generated Figures

After evaluation, find these in `results/`:

1. `attribute_uncertainty_correlation.png` - Box plots showing uncertainty by attribute
2. `urs_heatmap.png` - URS values for each attribute
3. `sas_boundary.png` - SAS scores by boundary type
4. `model_comparison.png` - Vision vs VL performance
5. `uncertainty_maps/` - Sample uncertainty visualizations

---

## Experiment Configurations

### Available Experiments

| Experiment Name | Text Attributes | Command |
|-----------------|-----------------|---------|
| `vision_only` | None | `--experiment vision_only` |
| `full` | shape, size, location, pathology | `--experiment full` |
| `text_shape_only` | shape | `--experiment text_shape_only` |
| `text_size_only` | size | `--experiment text_size_only` |
| `text_location_only` | location | `--experiment text_location_only` |
| `text_pathology_only` | pathology | `--experiment text_pathology_only` |

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--image_size` | 256 | Input image size |
| `--mc_samples` | 10 | MC Dropout samples for uncertainty |
| `--balance_strategy` | weighted | Dataset balancing (none/limit/weighted) |

---



<!-- ## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_name_2024,
  title={Semantic Language Cues Improve Segmentation Quality and 
         Uncertainty Interpretability in Medical Image Segmentation},
  author={Saaim Siddiqui},
  journal={Conference/Journal Name},
  year={2026}
}
``` -->

<!-- Also cite the SUN Database:
```bibtex
@article{misawa2021sun,
  title={Development of a computer-aided detection system for colonoscopy 
         and a publicly accessible large colonoscopy video database},
  author={Misawa, Masashi and others},
  journal={Gastrointestinal Endoscopy},
  year={2021}
} -->
```

---

## Contact

For questions about this implementation, please open an issue or contact saaimsiddiqui234@gmail.com.

---

## License

This project is for academic research purposes only.