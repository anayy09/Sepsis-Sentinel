"""
PyTorch Lightning DataModule for Sepsis Sentinel
Handles data loading, preprocessing, and batching for training.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Tensor

logger = logging.getLogger(__name__)


class SepsisDataset(Dataset):
    """Dataset class for Sepsis prediction with TFT and GNN data."""
    
    def __init__(self,
                 data_path: str,
                 split: str = 'train',
                 seq_len: int = 72,
                 transform: Optional[callable] = None):
        """
        Initialize Sepsis dataset.
        
        Args:
            data_path: Path to processed data directory
            split: Data split ('train', 'val', 'test')
            seq_len: Sequence length for temporal data
            transform: Optional data transformation function
        """
        self.data_path = Path(data_path)
        self.split = split
        self.seq_len = seq_len
        self.transform = transform
        
        # Load data
        self.samples = self._load_samples()
        
        # Load normalization statistics
        stats_path = self.data_path / 'stats.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.normalization_stats = json.load(f)
        else:
            self.normalization_stats = {}
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Dict]:
        """Load and prepare data samples."""
        # Mock data loading - in practice would load from Delta Lake or other format
        samples = []
        
        # Create synthetic data for demonstration
        for i in range(1000 if self.split == 'train' else 200):
            sample = self._create_synthetic_sample(i)
            samples.append(sample)
        
        return samples
    
    def _create_synthetic_sample(self, idx: int) -> Dict:
        """Create a synthetic data sample for testing."""
        # TFT static features
        tft_static_demographics = torch.randn(10)  # age, gender, race, etc.
        tft_static_admission = torch.randn(5)      # admission type, location, etc.
        
        # TFT temporal features
        tft_temporal_vitals = torch.randn(self.seq_len, 15)      # HR, BP, etc.
        tft_temporal_labs = torch.randn(self.seq_len, 25)        # Lab values
        tft_temporal_waveforms = torch.randn(self.seq_len, 20)   # Waveform features
        
        # GNN node features
        gnn_x_patient = torch.randn(20)    # Patient-level features
        gnn_x_stay = torch.randn(30)       # Stay-level features
        gnn_x_day = torch.randn(50)        # Day-level features
        
        # GNN edge indices (simplified)
        gnn_edge_patient_to_stay = torch.tensor([[0], [0]], dtype=torch.long)
        gnn_edge_stay_to_day = torch.tensor([[0], [0]], dtype=torch.long)
        gnn_edge_has_lab = torch.tensor([[0], [0]], dtype=torch.long)
        gnn_edge_has_vital = torch.tensor([[0], [0]], dtype=torch.long)
        gnn_edge_rev_patient_to_stay = torch.tensor([[0], [0]], dtype=torch.long)
        gnn_edge_rev_stay_to_day = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Target label (binary sepsis prediction)
        # Bias towards positive samples in certain ranges for realism
        prob_positive = 0.15 if idx % 10 < 8 else 0.7  # ~15% positive rate normally, 70% for some samples
        target = torch.tensor(1 if np.random.random() < prob_positive else 0, dtype=torch.long)
        
        return {
            # TFT inputs
            'tft_static_demographics': tft_static_demographics,
            'tft_static_admission': tft_static_admission,
            'tft_temporal_vitals': tft_temporal_vitals,
            'tft_temporal_labs': tft_temporal_labs,
            'tft_temporal_waveforms': tft_temporal_waveforms,
            
            # GNN inputs
            'gnn_x_patient': gnn_x_patient,
            'gnn_x_stay': gnn_x_stay,
            'gnn_x_day': gnn_x_day,
            'gnn_edge_patient_to_stay': gnn_edge_patient_to_stay,
            'gnn_edge_stay_to_day': gnn_edge_stay_to_day,
            'gnn_edge_has_lab': gnn_edge_has_lab,
            'gnn_edge_has_vital': gnn_edge_has_vital,
            'gnn_edge_rev_patient_to_stay': gnn_edge_rev_patient_to_stay,
            'gnn_edge_rev_stay_to_day': gnn_edge_rev_stay_to_day,
            
            # Target
            'targets': target,
            
            # Metadata
            'patient_id': f'PATIENT_{idx:06d}',
            'sample_idx': torch.tensor(idx, dtype=torch.long)
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = self.samples[idx].copy()
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class SepsisDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Sepsis prediction."""
    
    def __init__(self,
                 data_path: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 seq_len: int = 72,
                 train_split: float = 0.7,
                 val_split: float = 0.15,
                 test_split: float = 0.15,
                 **kwargs):
        """
        Initialize Sepsis DataModule.
        
        Args:
            data_path: Path to processed data
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for GPU transfer
            persistent_workers: Whether to keep workers alive between epochs
            seq_len: Sequence length for temporal data
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seq_len = seq_len
        
        # Data splits
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Datasets will be initialized in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Collate function for batching
        self.collate_fn = self._collate_batch
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing."""
        
        if stage == "fit" or stage is None:
            # Create full dataset
            full_dataset = SepsisDataset(
                data_path=self.data_path,
                split='full',
                seq_len=self.seq_len
            )
            
            # Calculate split sizes
            total_size = len(full_dataset)
            train_size = int(self.train_split * total_size)
            val_size = int(self.val_split * total_size)
            test_size = total_size - train_size - val_size
            
            # Split dataset
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset, 
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            logger.info(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        if stage == "test" or stage is None:
            if self.test_dataset is None:
                # Create test dataset separately if not already created
                full_dataset = SepsisDataset(
                    data_path=self.data_path,
                    split='test',
                    seq_len=self.seq_len
                )
                self.test_dataset = full_dataset
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=self.collate_fn,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=self.collate_fn,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=self.collate_fn,
            drop_last=False
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Return prediction dataloader (same as test)."""
        return self.test_dataloader()
    
    def _collate_batch(self, batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """
        Collate function to batch samples together.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Batched dictionary
        """
        collated = {}
        
        # Get keys from first sample
        keys = batch[0].keys()
        
        for key in keys:
            if key in ['patient_id']:
                # String fields - keep as list
                collated[key] = [sample[key] for sample in batch]
            elif key.startswith('gnn_edge_'):
                # Edge indices - handle specially for batched graphs
                edges = [sample[key] for sample in batch]
                # Simple concatenation for demo - in practice would handle graph batching properly
                collated[key] = torch.cat(edges, dim=1) if edges[0].numel() > 0 else torch.empty(2, 0, dtype=torch.long)
            else:
                # Tensor fields - stack into batch
                tensors = [sample[key] for sample in batch]
                collated[key] = torch.stack(tensors, dim=0)
        
        return collated
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the dataset."""
        if self.train_dataset is None:
            self.setup()
        
        # Calculate class distribution
        train_targets = [self.train_dataset[i]['targets'].item() for i in range(len(self.train_dataset))]
        val_targets = [self.val_dataset[i]['targets'].item() for i in range(len(self.val_dataset))]
        
        stats = {
            'total_samples': len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset),
            'train_samples': len(self.train_dataset),
            'val_samples': len(self.val_dataset),
            'test_samples': len(self.test_dataset),
            'train_positive_rate': np.mean(train_targets),
            'val_positive_rate': np.mean(val_targets),
            'sequence_length': self.seq_len,
            'batch_size': self.batch_size
        }
        
        return stats


if __name__ == "__main__":
    # Test the data module
    data_module = SepsisDataModule(
        data_path="/tmp/test_data",
        batch_size=4,
        num_workers=0  # For testing
    )
    
    # Setup and test
    data_module.setup()
    
    print("Dataset statistics:")
    stats = data_module.get_dataset_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test data loading
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"Batch size: {batch['targets'].shape[0]}")
    print(f"TFT temporal vitals shape: {batch['tft_temporal_vitals'].shape}")
    print(f"Target distribution: {batch['targets'].float().mean().item():.3f}")