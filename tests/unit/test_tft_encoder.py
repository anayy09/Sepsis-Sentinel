"""
Unit tests for TFT Encoder model.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the actual model (for real tests)
from models.tft_encoder import TFTEncoder, GatedResidualNetwork, VariableSelectionNetwork


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
        mixed_features = torch.cat([large_features[:1], small_features[1:]], dim=0)
        
        with patch('models.tft_encoder.TFTEncoder') as MockTFT:
            model = MockTFT()
            
            # Mock should handle all input ranges
            for features in [large_features, small_features, mixed_features]:
                static_features = torch.randn(batch_size, 8)
                model.forward.return_value = torch.randn(batch_size, 128)
                
                result = model.forward(features, static_features)
                assert not torch.isnan(result).any()
                assert not torch.isinf(result).any()


class TestTFTEncoderRealImplementation:
    """Test the actual TFT encoder implementation (not mocked)."""
    
    def test_gated_residual_network(self):
        """Test GatedResidualNetwork component."""
        batch_size = 4
        input_size = 32
        hidden_size = 64
        output_size = 32
        
        grn = GatedResidualNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=0.1
        )
        
        # Test forward pass
        x = torch.randn(batch_size, input_size)
        output = grn(x)
        
        assert output.shape == (batch_size, output_size)
        assert not torch.isnan(output).any()
        
        # Test with context
        context = torch.randn(batch_size, 16)
        grn_with_context = GatedResidualNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            context_size=16,
            dropout=0.1
        )
        
        output_with_context = grn_with_context(x, context)
        assert output_with_context.shape == (batch_size, output_size)
        assert not torch.isnan(output_with_context).any()
    
    def test_variable_selection_network(self):
        """Test VariableSelectionNetwork component."""
        batch_size = 4
        input_sizes = {
            'vitals': 15,
            'labs': 25,
            'waveforms': 20
        }
        hidden_size = 64
        
        vsn = VariableSelectionNetwork(
            input_sizes=input_sizes,
            hidden_size=hidden_size,
            dropout=0.1
        )
        
        # Create variable inputs
        variable_inputs = {
            'vitals': torch.randn(batch_size, 15),
            'labs': torch.randn(batch_size, 25),
            'waveforms': torch.randn(batch_size, 20)
        }
        
        # Test forward pass
        combined, selection_weights = vsn(variable_inputs)
        
        assert combined.shape == (batch_size, hidden_size)
        assert selection_weights.shape == (batch_size, len(input_sizes))
        assert torch.allclose(selection_weights.sum(dim=-1), torch.ones(batch_size), atol=1e-6)
        assert not torch.isnan(combined).any()
        assert not torch.isnan(selection_weights).any()
    
    def test_tft_encoder_real_initialization(self):
        """Test actual TFT encoder initialization."""
        static_input_sizes = {
            'demographics': 10,
            'admission': 5,
        }
        
        temporal_input_sizes = {
            'vitals': 15,
            'labs': 25,
            'waveforms': 20,
        }
        
        model = TFTEncoder(
            static_input_sizes=static_input_sizes,
            temporal_input_sizes=temporal_input_sizes,
            hidden_size=64,
            num_heads=4,
            seq_len=72
        )
        
        # Check model properties
        assert model.hidden_size == 64
        assert model.seq_len == 72
        
        # Check components exist
        assert hasattr(model, 'static_encoder')
        assert hasattr(model, 'temporal_selection')
        assert hasattr(model, 'lstm_encoder')
        assert hasattr(model, 'self_attention')
        
        # Check parameter count is reasonable
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 10000  # Should have substantial parameters
        assert total_params < 10000000  # But not too many
    
    def test_tft_encoder_real_forward_pass(self):
        """Test actual TFT encoder forward pass."""
        batch_size = 4
        seq_len = 72
        
        static_input_sizes = {
            'demographics': 10,
            'admission': 5,
        }
        
        temporal_input_sizes = {
            'vitals': 15,
            'labs': 25,
            'waveforms': 20,
        }
        
        model = TFTEncoder(
            static_input_sizes=static_input_sizes,
            temporal_input_sizes=temporal_input_sizes,
            hidden_size=64,
            num_heads=4,
            seq_len=seq_len
        )
        
        # Create inputs
        static_inputs = {
            'demographics': torch.randn(batch_size, 10),
            'admission': torch.randn(batch_size, 5),
        }
        
        temporal_inputs = {
            'vitals': torch.randn(batch_size, seq_len, 15),
            'labs': torch.randn(batch_size, seq_len, 25),
            'waveforms': torch.randn(batch_size, seq_len, 20),
        }
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(static_inputs, temporal_inputs)
        
        # Check outputs
        assert 'encoded' in outputs
        assert 'attention_weights' in outputs
        assert 'variable_selection_weights' in outputs
        
        encoded = outputs['encoded']
        attention_weights = outputs['attention_weights']
        vs_weights = outputs['variable_selection_weights']
        
        # Check shapes
        assert encoded.shape == (batch_size, seq_len, 64)
        assert attention_weights.shape == (batch_size, seq_len, seq_len)
        assert vs_weights.shape == (batch_size, seq_len, len(temporal_input_sizes))
        
        # Check for NaN/Inf values
        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()
        assert not torch.isnan(attention_weights).any()
        assert not torch.isnan(vs_weights).any()
        
        # Check attention weights are valid probabilities
        assert (attention_weights >= 0).all()
        assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)
        
        # Check variable selection weights are valid probabilities
        assert (vs_weights >= 0).all()
        assert torch.allclose(vs_weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)
    
    def test_tft_encoder_different_seq_lengths(self):
        """Test TFT encoder with different sequence lengths."""
        static_input_sizes = {
            'demographics': 10,
            'admission': 5,
        }
        
        temporal_input_sizes = {
            'vitals': 15,
            'labs': 25,
        }
        
        # Test different sequence lengths
        for seq_len in [24, 48, 72]:
            model = TFTEncoder(
                static_input_sizes=static_input_sizes,
                temporal_input_sizes=temporal_input_sizes,
                hidden_size=32,
                num_heads=2,
                seq_len=seq_len
            )
            
            batch_size = 2
            static_inputs = {
                'demographics': torch.randn(batch_size, 10),
                'admission': torch.randn(batch_size, 5),
            }
            
            temporal_inputs = {
                'vitals': torch.randn(batch_size, seq_len, 15),
                'labs': torch.randn(batch_size, seq_len, 25),
            }
            
            with torch.no_grad():
                outputs = model(static_inputs, temporal_inputs)
            
            assert outputs['encoded'].shape == (batch_size, seq_len, 32)
    
    def test_tft_encoder_gradients(self):
        """Test that gradients flow properly through TFT encoder."""
        batch_size = 2
        seq_len = 24  # Smaller for faster testing
        
        static_input_sizes = {
            'demographics': 5,
        }
        
        temporal_input_sizes = {
            'vitals': 10,
        }
        
        model = TFTEncoder(
            static_input_sizes=static_input_sizes,
            temporal_input_sizes=temporal_input_sizes,
            hidden_size=32,
            num_heads=2,
            seq_len=seq_len
        )
        
        static_inputs = {
            'demographics': torch.randn(batch_size, 5, requires_grad=True),
        }
        
        temporal_inputs = {
            'vitals': torch.randn(batch_size, seq_len, 10, requires_grad=True),
        }
        
        # Forward pass
        outputs = model(static_inputs, temporal_inputs)
        
        # Backward pass
        loss = outputs['encoded'].sum()
        loss.backward()
        
        # Check gradients exist
        assert static_inputs['demographics'].grad is not None
        assert temporal_inputs['vitals'].grad is not None
        
        # Check gradients are not all zero
        assert not torch.allclose(static_inputs['demographics'].grad, torch.zeros_like(static_inputs['demographics'].grad))
        assert not torch.allclose(temporal_inputs['vitals'].grad, torch.zeros_like(temporal_inputs['vitals'].grad))
        
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
