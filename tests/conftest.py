"""
Test configuration and fixtures for Sepsis Sentinel tests.
"""

import os
import tempfile
import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Set test environment
os.environ["TESTING"] = "true"


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return {
        "data": {
            "mimic_root": "/tmp/test_mimic",
            "output_path": "/tmp/test_output",
            "sequence_length": 72,
            "prediction_horizon": 6,
            "sampling_rate": 125
        },
        "model": {
            "tft": {
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_layers": 2,
                "dropout": 0.1
            },
            "gnn": {
                "hidden_channels": 32,
                "num_layers": 2,
                "heads": 2,
                "dropout": 0.1
            },
            "fusion": {
                "hidden_size": 128,
                "num_classes": 2,
                "dropout": 0.2
            }
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 1e-4,
            "max_epochs": 2,
            "patience": 5,
            "gradient_clip_val": 1.0
        }
    }


@pytest.fixture
def mock_patient_data():
    """Mock patient data for testing."""
    return {
        "patient_id": "TEST_001",
        "demographics": {
            "age": 65,
            "gender": "M",
            "race": "WHITE",
            "weight": 80.0,
            "height": 175.0
        },
        "admission_info": {
            "admission_type": "EMERGENCY",
            "admission_location": "ER",
            "insurance": "MEDICARE"
        },
        "vitals_sequence": np.random.randn(72, 7).astype(np.float32),
        "labs_sequence": np.random.randn(72, 15).astype(np.float32),
        "waveform_features": np.random.randn(72, 10).astype(np.float32),
        "sepsis_label": 0,
        "time_to_sepsis": None
    }


@pytest.fixture
def mock_batch_data():
    """Mock batch data for model testing."""
    batch_size = 4
    sequence_length = 72
    
    return {
        "tft_features": torch.randn(batch_size, sequence_length, 32),
        "static_features": torch.randn(batch_size, 8),
        "gnn_node_features": torch.randn(batch_size * 10, 16),
        "gnn_edge_index": torch.randint(0, batch_size * 10, (2, batch_size * 20)),
        "gnn_edge_attr": torch.randn(batch_size * 20, 4),
        "gnn_batch": torch.repeat_interleave(torch.arange(batch_size), 10),
        "labels": torch.randint(0, 2, (batch_size,)),
        "patient_ids": [f"TEST_{i:03d}" for i in range(batch_size)]
    }


@pytest.fixture
def mock_mimic_data():
    """Mock MIMIC-IV data for ETL testing."""
    # Mock waveform data
    waveform_data = pd.DataFrame({
        'subject_id': [1001, 1001, 1002, 1002] * 100,
        'hadm_id': [2001, 2001, 2002, 2002] * 100,
        'stay_id': [3001, 3001, 3002, 3002] * 100,
        'charttime': pd.date_range('2020-01-01', periods=400, freq='30min'),
        'itemid': [220045, 220050, 220045, 220050] * 100,  # HR, SBP
        'value': np.random.normal(80, 15, 400),
        'valueuom': ['bpm'] * 200 + ['mmHg'] * 200
    })
    
    # Mock hospital data
    hospital_data = pd.DataFrame({
        'subject_id': [1001, 1002, 1003, 1004],
        'hadm_id': [2001, 2002, 2003, 2004],
        'stay_id': [3001, 3002, 3003, 3004],
        'admittime': pd.date_range('2020-01-01', periods=4, freq='1D'),
        'dischtime': pd.date_range('2020-01-05', periods=4, freq='1D'),
        'deathtime': [None, None, '2020-01-08', None],
        'anchor_age': [65, 45, 78, 55],
        'gender': ['M', 'F', 'M', 'F']
    })
    
    # Mock lab data
    lab_data = pd.DataFrame({
        'subject_id': [1001, 1001, 1002, 1002] * 50,
        'hadm_id': [2001, 2001, 2002, 2002] * 50,
        'charttime': pd.date_range('2020-01-01', periods=200, freq='6H'),
        'itemid': [51006, 51248, 51006, 51248] * 50,  # BUN, MCH
        'value': np.random.normal(20, 5, 200),
        'valueuom': ['mg/dL'] * 100 + ['pg'] * 100
    })
    
    return {
        'waveforms': waveform_data,
        'hospital': hospital_data,
        'labs': lab_data
    }


@pytest.fixture
def temp_directory():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_model_artifacts(temp_directory):
    """Mock model artifacts for testing."""
    # Create mock ONNX model file
    onnx_path = temp_directory / "sepsis_sentinel.onnx"
    onnx_path.touch()
    
    # Create mock metadata
    metadata_path = temp_directory / "model_metadata.json"
    metadata = {
        "model_name": "sepsis_sentinel",
        "version": "1.0.0",
        "input_shapes": {
            "tft_features": [1, 256],
            "gnn_features": [1, 64]
        },
        "output_shapes": {
            "probabilities": [1, 2]
        },
        "created_at": "2024-01-01T00:00:00"
    }
    
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    return {
        "onnx_path": onnx_path,
        "metadata_path": metadata_path,
        "metadata": metadata
    }


@pytest.fixture
def mock_triton_client():
    """Mock Triton client for API testing."""
    client = Mock()
    client.is_server_live.return_value = True
    client.is_server_ready.return_value = True
    client.get_model_metadata.return_value = Mock()
    
    # Mock inference response
    response = Mock()
    response.as_numpy.return_value = np.array([[0.1, 0.9]])
    client.infer.return_value = response
    
    return client


@pytest.fixture
def mock_prediction_request():
    """Mock prediction request for API testing."""
    from datetime import datetime
    
    return {
        "patient_id": "TEST_001",
        "timestamp": datetime.now().isoformat(),
        "demographics": {
            "age": 65,
            "gender": "M",
            "race": "WHITE",
            "weight": 80.0,
            "height": 175.0
        },
        "admission_info": {
            "admission_type": "EMERGENCY",
            "admission_location": "ER",
            "insurance": "MEDICARE"
        },
        "vitals": [
            {
                "heart_rate": 85.0,
                "systolic_bp": 130.0,
                "diastolic_bp": 80.0,
                "temperature": 98.6,
                "respiratory_rate": 16.0,
                "spo2": 98.0,
                "gcs_total": 15
            }
        ] * 24,  # 24 time points minimum
        "labs": [
            {
                "wbc": 8.0,
                "hemoglobin": 12.0,
                "platelets": 250.0,
                "sodium": 140.0,
                "potassium": 4.0,
                "creatinine": 1.0,
                "glucose": 100.0,
                "lactate": 2.0
            }
        ] * 24,
        "waveforms": [
            {
                "ecg_hr_mean": 85.0,
                "abp_systolic_mean": 130.0,
                "resp_rate_mean": 16.0
            }
        ] * 24,
        "return_explanations": True,
        "return_confidence": True
    }


@pytest.fixture
def mock_shap_explainer():
    """Mock SHAP explainer for testing."""
    explainer = Mock()
    
    # Mock SHAP values
    shap_values = np.random.normal(0, 0.1, (1, 20))
    explainer.shap_values.return_value = shap_values
    explainer.expected_value = 0.15
    
    return explainer


@pytest.fixture
def mock_wandb():
    """Mock Weights & Biases for testing."""
    with patch('wandb.init') as mock_init, \
         patch('wandb.log') as mock_log, \
         patch('wandb.finish') as mock_finish:
        
        mock_run = Mock()
        mock_init.return_value = mock_run
        
        yield {
            'init': mock_init,
            'log': mock_log,
            'finish': mock_finish,
            'run': mock_run
        }


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing."""
    with patch('mlflow.start_run') as mock_start, \
         patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.log_param') as mock_log_param, \
         patch('mlflow.log_artifacts') as mock_log_artifacts, \
         patch('mlflow.end_run') as mock_end:
        
        yield {
            'start_run': mock_start,
            'log_metric': mock_log_metric,
            'log_param': mock_log_param,
            'log_artifacts': mock_log_artifacts,
            'end_run': mock_end
        }


# Test utilities
def assert_model_output_shape(output: torch.Tensor, expected_batch_size: int, expected_classes: int = 2):
    """Assert model output has correct shape."""
    assert isinstance(output, torch.Tensor)
    assert output.shape == (expected_batch_size, expected_classes)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def assert_valid_probability(prob: float):
    """Assert probability is valid."""
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    assert not np.isnan(prob)
    assert not np.isinf(prob)


def create_mock_checkpoint(path: Path, epoch: int = 10, val_auroc: float = 0.85):
    """Create a mock checkpoint file."""
    checkpoint = {
        'epoch': epoch,
        'state_dict': {},
        'optimizer': {},
        'val_auroc': val_auroc,
        'val_loss': 0.3,
        'hyper_parameters': {}
    }
    
    torch.save(checkpoint, path)
    return checkpoint


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Configure torch for testing
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers for slow tests
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if "e2e" in item.nodeid:
            item.add_marker(pytest.mark.slow)
