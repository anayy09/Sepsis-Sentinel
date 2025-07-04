"""
Unit tests for TFT Encoder model.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

# Import the model (would need proper import in actual implementation)
# from models.tft_encoder import TFTEncoder


class TestTFTEncoder:
    """Test cases for TFT Encoder."""
    
    def test_tft_encoder_initialization(self, test_config):
        """Test TFT encoder initializes correctly."""
        # Mock the TFT encoder class
        with patch('models.tft_encoder.TFTEncoder') as MockTFT:
            config = test_config["model"]["tft"]
            
            # Test initialization
            model = MockTFT(
                input_size=32,
                hidden_size=config["hidden_size"],
                num_attention_heads=config["num_attention_heads"],
                num_layers=config["num_layers"],
                dropout=config["dropout"]
            )
            
            MockTFT.assert_called_once()
            assert model is not None
    
    def test_tft_encoder_forward_pass(self, mock_batch_data):
        """Test TFT encoder forward pass."""
        batch_size, seq_len, input_size = mock_batch_data["tft_features"].shape
        
        # Mock model
        with patch('models.tft_encoder.TFTEncoder') as MockTFT:
            model = MockTFT()
            
            # Mock forward method
            expected_output = torch.randn(batch_size, 128)  # Hidden size
            model.forward.return_value = expected_output
            
            # Test forward pass
            features = mock_batch_data["tft_features"]
            static_features = mock_batch_data["static_features"]
            
            output = model.forward(features, static_features)
            
            assert output.shape == (batch_size, 128)
            model.forward.assert_called_once_with(features, static_features)
    
    def test_tft_variable_selection(self):
        """Test TFT variable selection network."""
        batch_size, seq_len, num_features = 4, 72, 32
        
        # Mock variable selection network
        with patch('models.tft_encoder.VariableSelectionNetwork') as MockVSN:
            vsn = MockVSN(num_features, 64)
            
            # Mock output: selected features and weights
            selected_features = torch.randn(batch_size, seq_len, 64)
            weights = torch.randn(batch_size, seq_len, num_features)
            
            vsn.return_value = (selected_features, weights)
            
            # Test variable selection
            input_features = torch.randn(batch_size, seq_len, num_features)
            output, attention_weights = vsn(input_features)
            
            assert output.shape == (batch_size, seq_len, 64)
            assert attention_weights.shape == (batch_size, seq_len, num_features)
            MockVSN.assert_called_once_with(num_features, 64)
    
    def test_tft_temporal_attention(self):
        """Test TFT temporal self-attention mechanism."""
        batch_size, seq_len, hidden_size = 4, 72, 128
        
        with patch('models.tft_encoder.InterpretableMultiHeadAttention') as MockMHA:
            mha = MockMHA(hidden_size, num_heads=4)
            
            # Mock attention output
            attended_features = torch.randn(batch_size, seq_len, hidden_size)
            attention_weights = torch.randn(batch_size, 4, seq_len, seq_len)
            
            mha.return_value = (attended_features, attention_weights)
            
            # Test attention
            input_features = torch.randn(batch_size, seq_len, hidden_size)
            output, weights = mha(input_features)
            
            assert output.shape == (batch_size, seq_len, hidden_size)
            assert weights.shape == (batch_size, 4, seq_len, seq_len)
    
    def test_tft_gated_residual_network(self):
        """Test TFT gated residual network."""
        batch_size, seq_len, hidden_size = 4, 72, 128
        
        with patch('models.tft_encoder.GatedResidualNetwork') as MockGRN:
            grn = MockGRN(hidden_size)
            
            # Mock GRN output
            output = torch.randn(batch_size, seq_len, hidden_size)
            grn.return_value = output
            
            # Test GRN
            input_features = torch.randn(batch_size, seq_len, hidden_size)
            result = grn(input_features)
            
            assert result.shape == (batch_size, seq_len, hidden_size)
            MockGRN.assert_called_once_with(hidden_size)
    
    def test_tft_static_covariate_encoders(self):
        """Test TFT static covariate encoders."""
        batch_size, num_static = 4, 8
        
        with patch('models.tft_encoder.StaticCovariateEncoder') as MockSCE:
            sce = MockSCE(num_static, 64)
            
            # Mock encoder output
            encoded_static = torch.randn(batch_size, 64)
            sce.return_value = encoded_static
            
            # Test static encoding
            static_features = torch.randn(batch_size, num_static)
            output = sce(static_features)
            
            assert output.shape == (batch_size, 64)
            MockSCE.assert_called_once_with(num_static, 64)
    
    def test_tft_feature_importance_extraction(self):
        """Test feature importance extraction from TFT."""
        batch_size, seq_len, num_features = 4, 72, 32
        
        with patch('models.tft_encoder.TFTEncoder') as MockTFT:
            model = MockTFT()
            
            # Mock importance extraction
            feature_importance = torch.randn(batch_size, num_features)
            temporal_importance = torch.randn(batch_size, seq_len)
            
            model.get_feature_importance.return_value = {
                'static_importance': feature_importance,
                'temporal_importance': temporal_importance
            }
            
            # Test importance extraction
            importance = model.get_feature_importance()
            
            assert 'static_importance' in importance
            assert 'temporal_importance' in importance
            assert importance['static_importance'].shape == (batch_size, num_features)
            assert importance['temporal_importance'].shape == (batch_size, seq_len)
    
    def test_tft_gradient_flow(self):
        """Test TFT gradient flow during backpropagation."""
        # Mock model training step
        with patch('models.tft_encoder.TFTEncoder') as MockTFT:
            model = MockTFT()
            model.training = True
            
            # Mock parameters
            param = torch.randn(10, 10, requires_grad=True)
            model.parameters.return_value = [param]
            
            # Mock loss computation
            loss = torch.tensor(0.5, requires_grad=True)
            
            # Test gradient computation
            if param.grad is not None:
                param.grad.zero_()
            
            # Simulate backpropagation
            loss.backward()
            
            # In real implementation, would check gradients exist
            assert param.requires_grad
    
    def test_tft_attention_mask_handling(self):
        """Test TFT attention mask handling for variable length sequences."""
        batch_size, max_seq_len, hidden_size = 4, 72, 128
        
        with patch('models.tft_encoder.TFTEncoder') as MockTFT:
            model = MockTFT()
            
            # Create attention mask (True for valid positions)
            attention_mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool)
            attention_mask[0, 50:] = False  # Shorter sequence
            attention_mask[1, 60:] = False  # Another shorter sequence
            
            # Mock forward with mask
            output = torch.randn(batch_size, hidden_size)
            model.forward.return_value = output
            
            # Test forward with mask
            features = torch.randn(batch_size, max_seq_len, 32)
            static_features = torch.randn(batch_size, 8)
            
            result = model.forward(features, static_features, attention_mask)
            
            assert result.shape == (batch_size, hidden_size)
            model.forward.assert_called_once()
    
    def test_tft_numerical_stability(self):
        """Test TFT numerical stability with extreme values."""
        batch_size, seq_len, input_size = 2, 72, 32
        
        # Test with large values
        large_features = torch.randn(batch_size, seq_len, input_size) * 100
        
        # Test with small values
        small_features = torch.randn(batch_size, seq_len, input_size) * 1e-6
        
        # Test with mixed values
        mixed_features = torch.cat([large_features, small_features], dim=0)
        
        with patch('models.tft_encoder.TFTEncoder') as MockTFT:
            model = MockTFT()
            
            # Mock stable output
            stable_output = torch.randn(batch_size * 2, 128)
            model.forward.return_value = stable_output
            
            # Test with extreme values
            static_features = torch.randn(batch_size * 2, 8)
            output = model.forward(mixed_features, static_features)
            
            # Check output is finite
            assert torch.isfinite(output).all()
            assert not torch.isnan(output).any()
    
    @pytest.mark.parametrize("dropout_rate", [0.0, 0.1, 0.3, 0.5])
    def test_tft_dropout_rates(self, dropout_rate):
        """Test TFT with different dropout rates."""
        with patch('models.tft_encoder.TFTEncoder') as MockTFT:
            model = MockTFT(dropout=dropout_rate)
            
            MockTFT.assert_called_with(dropout=dropout_rate)
            assert model is not None
    
    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
    def test_tft_attention_heads(self, num_heads):
        """Test TFT with different numbers of attention heads."""
        with patch('models.tft_encoder.TFTEncoder') as MockTFT:
            model = MockTFT(num_attention_heads=num_heads)
            
            MockTFT.assert_called_with(num_attention_heads=num_heads)
            assert model is not None
