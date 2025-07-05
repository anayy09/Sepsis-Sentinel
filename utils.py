"""
Utility functions for Sepsis Sentinel.
"""

import json
import logging
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import torch
import yaml


logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seeds set to {seed}")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """Save configuration to YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration saved to {config_path}")


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load data from JSON file."""
    file_path = Path(file_path)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2):
    """Save data to JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_pickle(file_path: Union[str, Path]) -> Any:
    """Load data from pickle file."""
    file_path = Path(file_path)
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def save_pickle(data: Any, file_path: Union[str, Path]):
    """Save data to pickle file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_device(device: Optional[str] = None) -> torch.device:
    """Get appropriate torch device."""
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    torch_device = torch.device(device)
    logger.info(f"Using device: {torch_device}")
    return torch_device


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def format_number(number: Union[int, float], precision: int = 2) -> str:
    """Format large numbers with appropriate suffixes."""
    if number >= 1e9:
        return f"{number / 1e9:.{precision}f}B"
    elif number >= 1e6:
        return f"{number / 1e6:.{precision}f}M"
    elif number >= 1e3:
        return f"{number / 1e3:.{precision}f}K"
    else:
        return f"{number:.{precision}f}"


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Calculate model size in MB."""
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        total_size += buffer.numel() * buffer.element_size()
    
    return total_size / (1024 ** 2)  # Convert to MB


def timing_decorator(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    return wrapper


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        logger.info(f"{self.name}: {elapsed:.4f} seconds")
    
    @property
    def elapsed(self) -> Optional[float]:
        """Get elapsed time if timer has finished."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


def validate_input_shapes(inputs: Dict[str, torch.Tensor], 
                         expected_shapes: Dict[str, Tuple[int, ...]]) -> bool:
    """Validate input tensor shapes match expected shapes."""
    for key, expected_shape in expected_shapes.items():
        if key not in inputs:
            logger.error(f"Missing input: {key}")
            return False
        
        actual_shape = inputs[key].shape
        
        # Check that dimensions match (ignoring batch dimension)
        if len(actual_shape) != len(expected_shape):
            logger.error(f"Shape mismatch for {key}: expected {len(expected_shape)} dims, got {len(actual_shape)}")
            return False
        
        # Check each dimension (skip batch dimension at index 0)
        for i, (actual, expected) in enumerate(zip(actual_shape[1:], expected_shape[1:]), 1):
            if expected != -1 and actual != expected:  # -1 means any size is allowed
                logger.error(f"Shape mismatch for {key} at dim {i}: expected {expected}, got {actual}")
                return False
    
    return True


def calculate_class_weights(labels: List[int]) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets."""
    from collections import Counter
    
    label_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(label_counts)
    
    # Calculate inverse frequency weights
    weights = []
    for class_idx in range(num_classes):
        class_count = label_counts.get(class_idx, 1)  # Avoid division by zero
        weight = total_samples / (num_classes * class_count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def normalize_features(features: torch.Tensor, 
                      method: str = "min_max",
                      dim: int = 0,
                      eps: float = 1e-8) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Normalize features using specified method.
    
    Args:
        features: Input features
        method: Normalization method ('min_max', 'z_score', 'robust')
        dim: Dimension along which to compute statistics
        eps: Small value to avoid division by zero
    
    Returns:
        Tuple of (normalized_features, normalization_stats)
    """
    if method == "min_max":
        min_vals = features.min(dim=dim, keepdim=True)[0]
        max_vals = features.max(dim=dim, keepdim=True)[0]
        range_vals = max_vals - min_vals + eps
        
        normalized = (features - min_vals) / range_vals
        stats = {'min': min_vals, 'max': max_vals, 'range': range_vals}
        
    elif method == "z_score":
        mean_vals = features.mean(dim=dim, keepdim=True)
        std_vals = features.std(dim=dim, keepdim=True) + eps
        
        normalized = (features - mean_vals) / std_vals
        stats = {'mean': mean_vals, 'std': std_vals}
        
    elif method == "robust":
        median_vals = features.median(dim=dim, keepdim=True)[0]
        mad_vals = torch.median(torch.abs(features - median_vals), dim=dim, keepdim=True)[0] + eps
        
        normalized = (features - median_vals) / mad_vals
        stats = {'median': median_vals, 'mad': mad_vals}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, stats


def apply_normalization(features: torch.Tensor,
                       stats: Dict[str, torch.Tensor],
                       method: str = "min_max") -> torch.Tensor:
    """Apply pre-computed normalization statistics to features."""
    if method == "min_max":
        normalized = (features - stats['min']) / stats['range']
    elif method == "z_score":
        normalized = (features - stats['mean']) / stats['std']
    elif method == "robust":
        normalized = (features - stats['median']) / stats['mad']
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def setup_logging(log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 log_format: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration."""
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    
    return logging.getLogger(__name__)


def memory_efficient_batch_process(data: List[Any],
                                  process_fn: callable,
                                  batch_size: int = 32,
                                  desc: str = "Processing") -> List[Any]:
    """Process data in memory-efficient batches."""
    try:
        from tqdm import tqdm
        progress_bar = True
    except ImportError:
        progress_bar = False
    
    results = []
    
    if progress_bar:
        iterator = tqdm(range(0, len(data), batch_size), desc=desc)
    else:
        iterator = range(0, len(data), batch_size)
    
    for i in iterator:
        batch = data[i:i + batch_size]
        batch_results = process_fn(batch)
        results.extend(batch_results)
    
    return results


def create_checkpoint_callback(monitor: str = "val_auroc",
                              dirpath: str = "./checkpoints",
                              filename: str = "model-{epoch:02d}-{val_auroc:.3f}",
                              save_top_k: int = 3,
                              mode: str = "max") -> 'ModelCheckpoint':
    """Create PyTorch Lightning ModelCheckpoint callback."""
    try:
        from pytorch_lightning.callbacks import ModelCheckpoint
        
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            dirpath=dirpath,
            filename=filename,
            save_top_k=save_top_k,
            mode=mode,
            save_last=True,
            auto_insert_metric_name=False,
            verbose=True
        )
        
        return checkpoint_callback
        
    except ImportError:
        logger.warning("PyTorch Lightning not available. Checkpoint callback not created.")
        return None


def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        import subprocess
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        return git_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def create_experiment_name(prefix: str = "sepsis",
                          timestamp: bool = True,
                          git_hash: bool = True) -> str:
    """Create a unique experiment name."""
    name_parts = [prefix]
    
    if timestamp:
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_parts.append(timestamp_str)
    
    if git_hash:
        git_hash_str = get_git_hash()[:8]  # Short hash
        if git_hash_str != "unknown":
            name_parts.append(git_hash_str)
    
    return "_".join(name_parts)


def check_data_leakage(train_ids: List[str], 
                      val_ids: List[str], 
                      test_ids: List[str]) -> bool:
    """Check for data leakage between train/val/test splits."""
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    
    # Check for overlaps
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    if train_val_overlap:
        logger.error(f"Data leakage: {len(train_val_overlap)} samples overlap between train and val")
        return False
    
    if train_test_overlap:
        logger.error(f"Data leakage: {len(train_test_overlap)} samples overlap between train and test")
        return False
    
    if val_test_overlap:
        logger.error(f"Data leakage: {len(val_test_overlap)} samples overlap between val and test")
        return False
    
    logger.info("No data leakage detected between splits")
    return True