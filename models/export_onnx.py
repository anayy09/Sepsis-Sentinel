"""
ONNX Export Utility for Sepsis Sentinel Models
Converts the fusion model to ONNX format for deployment.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torch import Tensor

from .fusion_head import SepsisClassificationHead
from .tft_encoder import TFTEncoder
from .hetero_gnn import HeteroGNN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SepsisModelONNXExporter:
    """Handles ONNX export for the complete Sepsis Sentinel model."""
    
    def __init__(self,
                 tft_model: TFTEncoder,
                 gnn_model: HeteroGNN,
                 fusion_head: SepsisClassificationHead,
                 device: torch.device):
        self.tft_model = tft_model
        self.gnn_model = gnn_model
        self.fusion_head = fusion_head
        self.device = device
        
        # Set models to eval mode
        self.tft_model.eval()
        self.gnn_model.eval()
        self.fusion_head.eval()
    
    def create_complete_model(self) -> nn.Module:
        """Create a complete model that combines TFT, GNN, and Fusion components."""
        
        class CompleteSepsisModel(nn.Module):
            def __init__(self, tft_model, gnn_model, fusion_head):
                super().__init__()
                self.tft_model = tft_model
                self.gnn_model = gnn_model
                self.fusion_head = fusion_head
            
            def forward(self,
                       # TFT inputs
                       static_demographics: Tensor,
                       static_admission: Tensor,
                       temporal_vitals: Tensor,
                       temporal_labs: Tensor,
                       temporal_waveforms: Tensor,
                       # GNN inputs
                       patient_features: Tensor,
                       stay_features: Tensor,
                       day_features: Tensor,
                       patient_to_stay_edges: Tensor,
                       stay_to_day_edges: Tensor,
                       has_lab_edges: Tensor,
                       has_vital_edges: Tensor) -> Tensor:
                
                # Prepare TFT inputs
                static_inputs = {
                    'demographics': static_demographics,
                    'admission': static_admission
                }
                
                temporal_inputs = {
                    'vitals': temporal_vitals,
                    'labs': temporal_labs,
                    'waveforms': temporal_waveforms
                }
                
                # TFT forward pass
                tft_outputs = self.tft_model(static_inputs, temporal_inputs)
                tft_encoded = tft_outputs['encoded']  # [batch, seq_len, hidden]
                
                # Prepare GNN inputs
                x_dict = {
                    'patient': patient_features,
                    'stay': stay_features,
                    'day': day_features
                }
                
                edge_index_dict = {
                    'patient_to_stay': patient_to_stay_edges,
                    'stay_to_day': stay_to_day_edges,
                    'has_lab': has_lab_edges,
                    'has_vital': has_vital_edges,
                    'rev_patient_to_stay': patient_to_stay_edges[[1, 0]],
                    'rev_stay_to_day': stay_to_day_edges[[1, 0]]
                }
                
                # GNN forward pass
                gnn_outputs = self.gnn_model(x_dict, edge_index_dict)
                
                # Use patient-level embeddings from GNN
                gnn_features = gnn_outputs['graph_embeddings']['patient']
                
                # Fusion forward pass
                fusion_results = self.fusion_head(tft_encoded, gnn_features)
                
                return fusion_results['probabilities']
        
        return CompleteSepsisModel(self.tft_model, self.gnn_model, self.fusion_head)
    
    def create_simplified_model(self) -> nn.Module:
        """Create a simplified model that takes preprocessed features."""
        
        class SimplifiedSepsisModel(nn.Module):
            def __init__(self, fusion_head):
                super().__init__()
                self.fusion_head = fusion_head
            
            def forward(self, tft_features: Tensor, gnn_features: Tensor) -> Tensor:
                """
                Forward pass with preprocessed features.
                
                Args:
                    tft_features: Preprocessed TFT features [batch, tft_dim]
                    gnn_features: Preprocessed GNN features [batch, gnn_dim]
                
                Returns:
                    Sepsis probabilities [batch]
                """
                results = self.fusion_head(tft_features, gnn_features)
                return results['probabilities']
        
        return SimplifiedSepsisModel(self.fusion_head)
    
    def export_to_onnx(self,
                      output_path: str,
                      model_type: str = "simplified",
                      opset_version: int = 17,
                      dynamic_axes: Optional[Dict] = None,
                      input_shapes: Optional[Dict] = None) -> str:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            model_type: Type of model to export ("complete" or "simplified")
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification
            input_shapes: Input shapes for dummy data generation
        
        Returns:
            Path to the exported ONNX model
        """
        logger.info(f"Exporting {model_type} model to ONNX...")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if model_type == "complete":
            return self._export_complete_model(output_path, opset_version, dynamic_axes, input_shapes)
        elif model_type == "simplified":
            return self._export_simplified_model(output_path, opset_version, dynamic_axes, input_shapes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _export_simplified_model(self,
                                output_path: str,
                                opset_version: int,
                                dynamic_axes: Optional[Dict],
                                input_shapes: Optional[Dict]) -> str:
        """Export simplified model that takes preprocessed features."""
        
        model = self.create_simplified_model()
        
        # Default input shapes
        if input_shapes is None:
            input_shapes = {
                'batch_size': 1,
                'tft_dim': 256,
                'gnn_dim': 64
            }
        
        # Create dummy inputs
        dummy_tft = torch.randn(
            input_shapes['batch_size'], 
            input_shapes['tft_dim']
        ).to(self.device)
        
        dummy_gnn = torch.randn(
            input_shapes['batch_size'], 
            input_shapes['gnn_dim']
        ).to(self.device)
        
        # Default dynamic axes
        if dynamic_axes is None:
            dynamic_axes = {
                'tft_features': {0: 'batch_size'},
                'gnn_features': {0: 'batch_size'},
                'probabilities': {0: 'batch_size'}
            }
        
        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_tft, dummy_gnn),
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['tft_features', 'gnn_features'],
                output_names=['probabilities'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
        
        logger.info(f"Simplified model exported to: {output_path}")
        return output_path
    
    def _export_complete_model(self,
                              output_path: str,
                              opset_version: int,
                              dynamic_axes: Optional[Dict],
                              input_shapes: Optional[Dict]) -> str:
        """Export complete model with all inputs."""
        
        model = self.create_complete_model()
        
        # Default input shapes
        if input_shapes is None:
            input_shapes = {
                'batch_size': 1,
                'seq_len': 72,
                'demographics_dim': 10,
                'admission_dim': 5,
                'vitals_dim': 15,
                'labs_dim': 25,
                'waveforms_dim': 20,
                'num_patients': 100,
                'num_stays': 150,
                'num_days': 500,
                'patient_features_dim': 20,
                'stay_features_dim': 30,
                'day_features_dim': 50
            }
        
        # Create dummy inputs
        batch_size = input_shapes['batch_size']
        seq_len = input_shapes['seq_len']
        
        dummy_inputs = [
            # TFT static inputs
            torch.randn(batch_size, input_shapes['demographics_dim']).to(self.device),
            torch.randn(batch_size, input_shapes['admission_dim']).to(self.device),
            # TFT temporal inputs
            torch.randn(batch_size, seq_len, input_shapes['vitals_dim']).to(self.device),
            torch.randn(batch_size, seq_len, input_shapes['labs_dim']).to(self.device),
            torch.randn(batch_size, seq_len, input_shapes['waveforms_dim']).to(self.device),
            # GNN node features
            torch.randn(input_shapes['num_patients'], input_shapes['patient_features_dim']).to(self.device),
            torch.randn(input_shapes['num_stays'], input_shapes['stay_features_dim']).to(self.device),
            torch.randn(input_shapes['num_days'], input_shapes['day_features_dim']).to(self.device),
            # GNN edge indices
            torch.randint(0, 100, (2, 200)).to(self.device),  # patient_to_stay
            torch.randint(0, 300, (2, 400)).to(self.device),  # stay_to_day
            torch.stack([torch.arange(500), torch.arange(500)]).to(self.device),  # has_lab
            torch.stack([torch.arange(500), torch.arange(500)]).to(self.device),  # has_vital
        ]
        
        input_names = [
            'static_demographics', 'static_admission',
            'temporal_vitals', 'temporal_labs', 'temporal_waveforms',
            'patient_features', 'stay_features', 'day_features',
            'patient_to_stay_edges', 'stay_to_day_edges',
            'has_lab_edges', 'has_vital_edges'
        ]
        
        # Default dynamic axes for complete model
        if dynamic_axes is None:
            dynamic_axes = {
                'static_demographics': {0: 'batch_size'},
                'static_admission': {0: 'batch_size'},
                'temporal_vitals': {0: 'batch_size', 1: 'seq_len'},
                'temporal_labs': {0: 'batch_size', 1: 'seq_len'},
                'temporal_waveforms': {0: 'batch_size', 1: 'seq_len'},
                'probabilities': {0: 'batch_size'}
            }
        
        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                model,
                tuple(dummy_inputs),
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=['probabilities'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
        
        logger.info(f"Complete model exported to: {output_path}")
        return output_path
    
    def validate_onnx_model(self, onnx_path: str, test_data: Optional[Dict] = None) -> bool:
        """
        Validate the exported ONNX model.
        
        Args:
            onnx_path: Path to the ONNX model
            test_data: Optional test data for validation
        
        Returns:
            True if validation passes
        """
        logger.info(f"Validating ONNX model: {onnx_path}")
        
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model structure validation passed")
            
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(onnx_path)
            
            # Print model info
            logger.info("Model inputs:")
            for input_meta in ort_session.get_inputs():
                logger.info(f"  {input_meta.name}: {input_meta.shape} ({input_meta.type})")
            
            logger.info("Model outputs:")
            for output_meta in ort_session.get_outputs():
                logger.info(f"  {output_meta.name}: {output_meta.shape} ({output_meta.type})")
            
            # Test inference if test data provided
            if test_data is not None:
                logger.info("Testing ONNX model inference...")
                
                # Prepare inputs for ONNX Runtime
                ort_inputs = {}
                for input_meta in ort_session.get_inputs():
                    input_name = input_meta.name
                    if input_name in test_data:
                        ort_inputs[input_name] = test_data[input_name].cpu().numpy()
                
                # Run inference
                ort_outputs = ort_session.run(None, ort_inputs)
                logger.info(f"ONNX inference successful. Output shape: {ort_outputs[0].shape}")
                
                # Compare with PyTorch model if possible
                if hasattr(self, '_compare_with_pytorch'):
                    self._compare_with_pytorch(ort_outputs, test_data)
            
            logger.info("ONNX model validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"ONNX model validation failed: {str(e)}")
            return False
    
    def optimize_onnx_model(self, input_path: str, output_path: str) -> str:
        """
        Optimize ONNX model for better performance.
        
        Args:
            input_path: Path to input ONNX model
            output_path: Path to save optimized model
        
        Returns:
            Path to optimized model
        """
        try:
            import onnxoptimizer
            
            logger.info("Optimizing ONNX model...")
            
            # Load model
            model = onnx.load(input_path)
            
            # Apply optimizations
            optimized_model = onnxoptimizer.optimize(
                model,
                passes=[
                    'eliminate_identity',
                    'eliminate_nop_dropout',
                    'eliminate_nop_transpose',
                    'fuse_add_bias_into_conv',
                    'fuse_consecutive_transposes',
                    'fuse_transpose_into_gemm',
                    'lift_lexical_references'
                ]
            )
            
            # Save optimized model
            onnx.save(optimized_model, output_path)
            logger.info(f"Optimized model saved to: {output_path}")
            
            return output_path
            
        except ImportError:
            logger.warning("onnxoptimizer not available. Skipping optimization.")
            return input_path
        except Exception as e:
            logger.error(f"Model optimization failed: {str(e)}")
            return input_path


def create_triton_config(model_name: str,
                        input_specs: List[Dict],
                        output_specs: List[Dict],
                        max_batch_size: int = 128,
                        instance_group_count: int = 1) -> str:
    """
    Create Triton Inference Server configuration.
    
    Args:
        model_name: Name of the model
        input_specs: List of input specifications
        output_specs: List of output specifications
        max_batch_size: Maximum batch size
        instance_group_count: Number of model instances
    
    Returns:
        Configuration string
    """
    config_lines = [
        f'name: "{model_name}"',
        'platform: "onnxruntime_onnx"',
        f'max_batch_size: {max_batch_size}',
        '',
        'sequence_batching {',
        '  max_sequence_idle_microseconds: 5000000',
        '  control_input [',
        '    {',
        '      name: "START"',
        '      control [',
        '        {',
        '          kind: CONTROL_SEQUENCE_START',
        '          fp32_false_true: [ 0, 1 ]',
        '        }',
        '      ]',
        '    },',
        '    {',
        '      name: "END"',
        '      control [',
        '        {',
        '          kind: CONTROL_SEQUENCE_END',
        '          fp32_false_true: [ 0, 1 ]',
        '        }',
        '      ]',
        '    }',
        '  ]',
        '}',
        ''
    ]
    
    # Add input specifications
    for i, input_spec in enumerate(input_specs):
        config_lines.extend([
            'input [',
            '  {',
            f'    name: "{input_spec["name"]}"',
            f'    data_type: {input_spec["data_type"]}',
            f'    dims: {input_spec["dims"]}',
            '  }',
            ']' if i == len(input_specs) - 1 else ','
        ])
    
    # Add output specifications
    for i, output_spec in enumerate(output_specs):
        config_lines.extend([
            'output [',
            '  {',
            f'    name: "{output_spec["name"]}"',
            f'    data_type: {output_spec["data_type"]}',
            f'    dims: {output_spec["dims"]}',
            '  }',
            ']' if i == len(output_specs) - 1 else ','
        ])
    
    # Add instance group
    config_lines.extend([
        '',
        'instance_group [',
        '  {',
        f'    count: {instance_group_count}',
        '    kind: KIND_CPU',
        '  }',
        ']',
        '',
        'optimization {',
        '  execution_accelerators {',
        '    cpu_execution_accelerator : [ {',
        '      name : "openvino"',
        '    } ]',
        '  }',
        '}'
    ])
    
    return '\n'.join(config_lines)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Export Sepsis Sentinel model to ONNX")
    parser.add_argument("--model-path", required=True, help="Path to trained PyTorch model")
    parser.add_argument("--output-path", required=True, help="Output path for ONNX model")
    parser.add_argument("--model-type", choices=["complete", "simplified"], 
                       default="simplified", help="Type of model to export")
    parser.add_argument("--validate", action="store_true", help="Validate exported model")
    parser.add_argument("--optimize", action="store_true", help="Optimize exported model")
    
    args = parser.parse_args()
    
    # This would need to be implemented with proper model loading
    logger.info("ONNX export utility ready. Implement model loading for actual export.")
