"""
Unit tests for Heterogeneous GNN model.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the actual model
from models.hetero_gnn import HeteroGNN, MedicalHeteroGATConv


class TestHeteroGNN:
    """Test cases for Heterogeneous GNN."""
    
    def test_medical_hetero_gat_conv(self):
        """Test MedicalHeteroGATConv layer."""
        batch_size = 8
        in_channels = 32
        out_channels = 16
        heads = 2
        
        conv = MedicalHeteroGATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=True
        )
        
        # Test homogeneous edge
        x = torch.randn(batch_size, in_channels)
        edge_index = torch.randint(0, batch_size, (2, 16))
        
        output = conv(x, edge_index)
        
        expected_out_channels = heads * out_channels  # concat=True
        assert output.shape == (batch_size, expected_out_channels)
        assert not torch.isnan(output).any()
        
        # Test heterogeneous edge
        x_src = torch.randn(batch_size, in_channels)
        x_dst = torch.randn(batch_size, in_channels)
        
        conv_hetero = MedicalHeteroGATConv(
            in_channels=(in_channels, in_channels),
            out_channels=out_channels,
            heads=heads,
            concat=False  # Test concat=False
        )
        
        output_hetero = conv_hetero((x_src, x_dst), edge_index)
        assert output_hetero.shape == (batch_size, out_channels)  # concat=False
        assert not torch.isnan(output_hetero).any()
    
    def test_hetero_gnn_initialization(self):
        """Test HeteroGNN initialization."""
        node_channels = {
            'patient': 20,
            'stay': 30,
            'day': 50,
        }
        
        model = HeteroGNN(
            node_channels=node_channels,
            hidden_channels=64,
            out_channels=64,
            num_layers=2,
            heads=4,
            dropout=0.1
        )
        
        # Check basic properties
        assert model.node_channels == node_channels
        assert model.hidden_channels == 64
        assert model.out_channels == 64
        assert model.num_layers == 2
        assert model.heads == 4
        
        # Check components
        assert len(model.input_projections) == len(node_channels)
        assert len(model.convs) == 2
        assert len(model.output_projections) == len(node_channels)
        assert len(model.global_pools) == len(node_channels)
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
    
    def test_hetero_gnn_forward_pass(self):
        """Test HeteroGNN forward pass."""
        node_channels = {
            'patient': 20,
            'stay': 30,
            'day': 50,
        }
        
        model = HeteroGNN(
            node_channels=node_channels,
            hidden_channels=32,
            out_channels=32,
            num_layers=1,  # Single layer for faster testing
            heads=2,
            dropout=0.1
        )
        
        # Create node features
        num_patients = 5
        num_stays = 8
        num_days = 15
        
        x_dict = {
            'patient': torch.randn(num_patients, 20),
            'stay': torch.randn(num_stays, 30),
            'day': torch.randn(num_days, 50),
        }
        
        # Create edge indices (simplified for testing)
        edge_index_dict = {
            'patient_to_stay': torch.randint(0, min(num_patients, num_stays), (2, 10)),
            'stay_to_day': torch.randint(0, min(num_stays, num_days), (2, 12)),
            'has_lab': torch.stack([torch.arange(num_days), torch.arange(num_days)]),
            'has_vital': torch.stack([torch.arange(num_days), torch.arange(num_days)]),
            'rev_patient_to_stay': torch.randint(0, min(num_stays, num_patients), (2, 10)),
            'rev_stay_to_day': torch.randint(0, min(num_days, num_stays), (2, 12)),
        }
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(x_dict, edge_index_dict)
        
        # Check outputs
        assert 'node_embeddings' in outputs
        assert 'graph_embeddings' in outputs
        assert 'attention_weights' in outputs
        
        node_embeddings = outputs['node_embeddings']
        graph_embeddings = outputs['graph_embeddings']
        
        # Check node embeddings
        for node_type in node_channels.keys():
            assert node_type in node_embeddings
            if node_type == 'patient':
                assert node_embeddings[node_type].shape == (num_patients, 32)
            elif node_type == 'stay':
                assert node_embeddings[node_type].shape == (num_stays, 32)
            elif node_type == 'day':
                assert node_embeddings[node_type].shape == (num_days, 32)
            
            assert not torch.isnan(node_embeddings[node_type]).any()
        
        # Check graph embeddings
        for node_type in node_channels.keys():
            assert node_type in graph_embeddings
            assert graph_embeddings[node_type].shape[1] == 8  # out_channels // 4
            assert not torch.isnan(graph_embeddings[node_type]).any()
    
    def test_hetero_gnn_different_architectures(self):
        """Test HeteroGNN with different architectures."""
        node_channels = {
            'patient': 10,
            'stay': 15,
        }
        
        # Test different configurations
        configs = [
            {'hidden_channels': 16, 'out_channels': 16, 'num_layers': 1, 'heads': 1},
            {'hidden_channels': 32, 'out_channels': 64, 'num_layers': 2, 'heads': 2},
            {'hidden_channels': 64, 'out_channels': 32, 'num_layers': 3, 'heads': 4},
        ]
        
        for config in configs:
            model = HeteroGNN(
                node_channels=node_channels,
                **config,
                dropout=0.1
            )
            
            # Create simple inputs
            x_dict = {
                'patient': torch.randn(3, 10),
                'stay': torch.randn(5, 15),
            }
            
            edge_index_dict = {
                'patient_to_stay': torch.tensor([[0, 1, 2], [0, 1, 2]]),
                'stay_to_day': torch.empty((2, 0), dtype=torch.long),
                'has_lab': torch.empty((2, 0), dtype=torch.long),
                'has_vital': torch.empty((2, 0), dtype=torch.long),
                'rev_patient_to_stay': torch.tensor([[0, 1, 2], [0, 1, 2]]),
                'rev_stay_to_day': torch.empty((2, 0), dtype=torch.long),
            }
            
            with torch.no_grad():
                outputs = model(x_dict, edge_index_dict)
            
            # Check that forward pass works
            assert 'node_embeddings' in outputs
            assert 'graph_embeddings' in outputs
            
            # Check output dimensions
            for node_type in node_channels.keys():
                assert outputs['node_embeddings'][node_type].shape[1] == config['out_channels']
    
    def test_hetero_gnn_gradients(self):
        """Test that gradients flow properly through HeteroGNN."""
        node_channels = {
            'patient': 10,
            'stay': 15,
        }
        
        model = HeteroGNN(
            node_channels=node_channels,
            hidden_channels=16,
            out_channels=16,
            num_layers=1,
            heads=1,
            dropout=0.1
        )
        
        # Create inputs with gradient tracking
        x_dict = {
            'patient': torch.randn(3, 10, requires_grad=True),
            'stay': torch.randn(5, 15, requires_grad=True),
        }
        
        edge_index_dict = {
            'patient_to_stay': torch.tensor([[0, 1, 2], [0, 1, 2]]),
            'stay_to_day': torch.empty((2, 0), dtype=torch.long),
            'has_lab': torch.empty((2, 0), dtype=torch.long),
            'has_vital': torch.empty((2, 0), dtype=torch.long),
            'rev_patient_to_stay': torch.tensor([[0, 1, 2], [0, 1, 2]]),
            'rev_stay_to_day': torch.empty((2, 0), dtype=torch.long),
        }
        
        # Forward pass
        outputs = model(x_dict, edge_index_dict)
        
        # Backward pass
        loss = outputs['node_embeddings']['patient'].sum()
        loss.backward()
        
        # Check gradients exist
        assert x_dict['patient'].grad is not None
        assert x_dict['stay'].grad is not None
        
        # Check gradients are not all zero
        assert not torch.allclose(x_dict['patient'].grad, torch.zeros_like(x_dict['patient']))
        # Note: stay gradients might be zero if no edges connect to them
    
    def test_hetero_gnn_empty_edges(self):
        """Test HeteroGNN with empty edge sets."""
        node_channels = {
            'patient': 10,
            'stay': 15,
            'day': 20,
        }
        
        model = HeteroGNN(
            node_channels=node_channels,
            hidden_channels=16,
            out_channels=16,
            num_layers=1,
            heads=1,
            dropout=0.1
        )
        
        x_dict = {
            'patient': torch.randn(2, 10),
            'stay': torch.randn(3, 15),
            'day': torch.randn(4, 20),
        }
        
        # All empty edge sets
        edge_index_dict = {
            'patient_to_stay': torch.empty((2, 0), dtype=torch.long),
            'stay_to_day': torch.empty((2, 0), dtype=torch.long),
            'has_lab': torch.empty((2, 0), dtype=torch.long),
            'has_vital': torch.empty((2, 0), dtype=torch.long),
            'rev_patient_to_stay': torch.empty((2, 0), dtype=torch.long),
            'rev_stay_to_day': torch.empty((2, 0), dtype=torch.long),
        }
        
        # Should still work (nodes just get processed through input/output projections)
        with torch.no_grad():
            outputs = model(x_dict, edge_index_dict)
        
        assert 'node_embeddings' in outputs
        assert 'graph_embeddings' in outputs
        
        # Check shapes are correct
        assert outputs['node_embeddings']['patient'].shape == (2, 16)
        assert outputs['node_embeddings']['stay'].shape == (3, 16)
        assert outputs['node_embeddings']['day'].shape == (4, 16)
    
    def test_hetero_gnn_numerical_stability(self):
        """Test HeteroGNN numerical stability."""
        node_channels = {
            'patient': 5,
        }
        
        model = HeteroGNN(
            node_channels=node_channels,
            hidden_channels=8,
            out_channels=8,
            num_layers=1,
            heads=1,
            dropout=0.0  # No dropout for stability testing
        )
        
        # Test with extreme values
        test_cases = [
            torch.randn(3, 5) * 100,      # Large values
            torch.randn(3, 5) * 1e-6,     # Small values
            torch.zeros(3, 5),            # Zero values
            torch.ones(3, 5) * 1000,      # Large constant values
        ]
        
        edge_index_dict = {
            'patient_to_stay': torch.empty((2, 0), dtype=torch.long),
            'stay_to_day': torch.empty((2, 0), dtype=torch.long),
            'has_lab': torch.empty((2, 0), dtype=torch.long),
            'has_vital': torch.empty((2, 0), dtype=torch.long),
            'rev_patient_to_stay': torch.empty((2, 0), dtype=torch.long),
            'rev_stay_to_day': torch.empty((2, 0), dtype=torch.long),
        }
        
        for x_patient in test_cases:
            x_dict = {'patient': x_patient}
            
            with torch.no_grad():
                outputs = model(x_dict, edge_index_dict)
            
            # Check for numerical issues
            patient_output = outputs['node_embeddings']['patient']
            assert not torch.isnan(patient_output).any(), f"NaN detected with input: {x_patient[0, 0].item()}"
            assert not torch.isinf(patient_output).any(), f"Inf detected with input: {x_patient[0, 0].item()}"
    
    def test_hetero_gnn_batch_consistency(self):
        """Test that HeteroGNN produces consistent results for different batch sizes."""
        node_channels = {
            'patient': 8,
        }
        
        model = HeteroGNN(
            node_channels=node_channels,
            hidden_channels=16,
            out_channels=16,
            num_layers=1,
            heads=1,
            dropout=0.0
        )
        
        # Set model to eval mode for consistent results
        model.eval()
        
        # Create deterministic input
        torch.manual_seed(42)
        single_patient = torch.randn(1, 8)
        
        edge_index_dict = {
            'patient_to_stay': torch.empty((2, 0), dtype=torch.long),
            'stay_to_day': torch.empty((2, 0), dtype=torch.long),
            'has_lab': torch.empty((2, 0), dtype=torch.long),
            'has_vital': torch.empty((2, 0), dtype=torch.long),
            'rev_patient_to_stay': torch.empty((2, 0), dtype=torch.long),
            'rev_stay_to_day': torch.empty((2, 0), dtype=torch.long),
        }
        
        # Process single patient
        x_dict_single = {'patient': single_patient}
        
        with torch.no_grad():
            output_single = model(x_dict_single, edge_index_dict)
        
        # Process same patient in batch of 2
        x_dict_batch = {'patient': single_patient.repeat(2, 1)}
        
        with torch.no_grad():
            output_batch = model(x_dict_batch, edge_index_dict)
        
        # Results should be identical (up to numerical precision)
        single_result = output_single['node_embeddings']['patient'][0]
        batch_result_0 = output_batch['node_embeddings']['patient'][0]
        batch_result_1 = output_batch['node_embeddings']['patient'][1]
        
        assert torch.allclose(single_result, batch_result_0, atol=1e-6)
        assert torch.allclose(single_result, batch_result_1, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])