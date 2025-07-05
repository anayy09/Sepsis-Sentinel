"""
Sepsis Sentinel - Early sepsis prediction using multimodal AI.
"""

__version__ = "1.0.0"
__author__ = "Sepsis Sentinel Team"
__email__ = "contact@sepsis-sentinel.org"

# Core components
from .models import TFTEncoder, HeteroGNN, FusionHead, FocalLoss, AttentionFusion
from .training import SepsisSentinelLightning
from .data_pipeline import SepsisDataModule, SepsisDataset
from .utils import (
    set_random_seeds, load_config, save_config, get_device, 
    count_parameters, format_number, Timer
)
from .evaluate import ModelEvaluator

__all__ = [
    # Version info
    '__version__', '__author__', '__email__',
    
    # Models
    'TFTEncoder', 'HeteroGNN', 'FusionHead', 'FocalLoss', 'AttentionFusion',
    
    # Training
    'SepsisSentinelLightning',
    
    # Data
    'SepsisDataModule', 'SepsisDataset',
    
    # Utilities
    'set_random_seeds', 'load_config', 'save_config', 'get_device',
    'count_parameters', 'format_number', 'Timer',
    
    # Evaluation
    'ModelEvaluator',
]