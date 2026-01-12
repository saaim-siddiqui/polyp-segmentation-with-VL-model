"""
Data loading utilities for VL Polyp Segmentation.

Datasets:
- SUNPolypDataset: SUN Colonoscopy Database (primary, with metadata)
- BenchmarkPolypDataset: Kvasir-SEG, CVC-ClinicDB (benchmarks, no metadata)
"""

from .dataset_sun import (
    SUNPolypDataset,
    create_sun_dataloaders,
    collate_fn,
)

from .benchmark_datasets import (
    BenchmarkPolypDataset,
    create_benchmark_loader,
    create_kvasir_loader,
    create_cvc_clinicdb_loader,
    evaluate_on_benchmark,
    run_benchmark_evaluation,
)

__all__ = [
    # SUN Database (primary)
    'SUNPolypDataset',
    'create_sun_dataloaders',
    'collate_fn',
    # Benchmark Datasets
    'BenchmarkPolypDataset',
    'create_benchmark_loader',
    'create_kvasir_loader',
    'create_cvc_clinicdb_loader',
    'evaluate_on_benchmark',
    'run_benchmark_evaluation',
]