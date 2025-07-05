"""
Unit tests for Fusion Head model.
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
from models.fusion_head import FusionHead, FocalLoss, AttentionFusion


class TestFocalLoss:
    """Test cases for Focal Loss."""
    
    def test_focal_loss_initialization(self):
        """Test FocalLoss initialization."""
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        
        assert focal_loss.alpha == 0.25
        assert focal_loss.gamma == 2.0
        assert focal_loss.reduction == 'mean'
    
    def test_focal_loss_forward_balanced(self):
        """Test FocalLoss with balanced data."""
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        
        batch_size = 8
        logits = torch.randn(batch_size)
        targets = torch.randint(0, 2, (batch_size,)).float()
        
        loss = focal_loss(logits, targets)
        
        assert loss.shape == torch.Size([])  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_focal_loss_forward_imbalanced(self):
        """Test FocalLoss with imbalanced data."""
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        
        batch_size = 100
        logits = torch.randn(batch_size)
        
        # Create imbalanced targets (90% negative, 10% positive)
        targets = torch.zeros(batch_size)
        targets[:10] = 1.0
        
        loss = focal_loss(logits, targets)
        
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_focal_loss_reduction_modes(self):
        """Test different reduction modes."""
        batch_size = 4
        logits = torch.randn(batch_size)
        targets = torch.randint(0, 2, (batch_size,)).float()
        
        # Test mean reduction
        focal_loss_mean = FocalLoss(reduction='mean')
        loss_mean = focal_loss_mean(logits, targets)
        assert loss_mean.shape == torch.Size([])
        
        # Test sum reduction
        focal_loss_sum = FocalLoss(reduction='sum')
        loss_sum = focal_loss_sum(logits, targets)
        assert loss_sum.shape == torch.Size([])
        
        # Test none reduction
        focal_loss_none = FocalLoss(reduction='none')
        loss_none = focal_loss_none(logits, targets)
        assert loss_none.shape == (batch_size,)
        
        # Sum should be approximately batch_size * mean (for this case)
        assert torch.allclose(loss_sum, loss_none.sum(), atol=1e-6)
    
    def test_focal_loss_gradient_flow(self):
        """Test gradient flow through FocalLoss."""
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
        batch_size = 4
        logits = torch.randn(batch_size, requires_grad=True)
        targets = torch.randint(0, 2, (batch_size,)).float()
        
        loss = focal_loss(logits, targets)
        loss.backward()
        
        assert logits.grad is not None
        assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad))


class TestAttentionFusion:
    """Test cases for Attention Fusion."""
    
    def test_attention_fusion_initialization(self):
        """Test AttentionFusion initialization."""
        tft_dim = 256
        gnn_dim = 64
        hidden_dim = 128
        
        fusion = AttentionFusion(
            tft_dim=tft_dim,
            gnn_dim=gnn_dim,
            hidden_dim=hidden_dim,
            dropout=0.1
        )
        
        assert fusion.tft_dim == tft_dim
        assert fusion.gnn_dim == gnn_dim
        assert fusion.hidden_dim == hidden_dim
        
        # Check components
        assert hasattr(fusion, 'tft_proj')
        assert hasattr(fusion, 'gnn_proj')
        assert hasattr(fusion, 'attention')
        assert hasattr(fusion, 'cross_attention')
        assert hasattr(fusion, 'layer_norm')
        assert hasattr(fusion, 'dropout')
    
    def test_attention_fusion_forward(self):
        """Test AttentionFusion forward pass."""
        batch_size = 4
        tft_dim = 256
        gnn_dim = 64
        hidden_dim = 128
        
        fusion = AttentionFusion(
            tft_dim=tft_dim,
            gnn_dim=gnn_dim,
            hidden_dim=hidden_dim,
            dropout=0.1
        )
        
        tft_features = torch.randn(batch_size, tft_dim)
        gnn_features = torch.randn(batch_size, gnn_dim)
        
        fused, enhanced_concat, attention_info = fusion(tft_features, gnn_features)
        
        # Check output shapes
        assert fused.shape == (batch_size, hidden_dim)
        assert enhanced_concat.shape == (batch_size, hidden_dim * 2)  # tft + gnn enhanced
        
        # Check attention info
        assert isinstance(attention_info, dict)
        assert 'self_attention' in attention_info
        assert 'tft_cross_attention' in attention_info
        assert 'gnn_cross_attention' in attention_info
        
        # Check for numerical stability
        assert not torch.isnan(fused).any()
        assert not torch.isnan(enhanced_concat).any()
        assert not torch.isinf(fused).any()
        assert not torch.isinf(enhanced_concat).any()
    
    def test_attention_fusion_different_dims(self):
        """Test AttentionFusion with different input dimensions."""
        test_configs = [
            {'tft_dim': 128, 'gnn_dim': 32, 'hidden_dim': 64},
            {'tft_dim': 512, 'gnn_dim': 128, 'hidden_dim': 256},
            {'tft_dim': 64, 'gnn_dim': 64, 'hidden_dim': 32},
        ]
        
        batch_size = 4
        
        for config in test_configs:
            fusion = AttentionFusion(**config, dropout=0.1)
            
            tft_features = torch.randn(batch_size, config['tft_dim'])
            gnn_features = torch.randn(batch_size, config['gnn_dim'])
            
            fused, enhanced_concat, attention_info = fusion(tft_features, gnn_features)
            
            assert fused.shape == (batch_size, config['hidden_dim'])
            assert enhanced_concat.shape == (batch_size, config['hidden_dim'] * 2)


class TestFusionHead:
    """Test cases for Fusion Head."""
    
    def test_fusion_head_initialization(self):
        """Test FusionHead initialization."""
        tft_dim = 256
        gnn_dim = 64
        hidden_dims = [256, 64]
        
        fusion_head = FusionHead(
            tft_dim=tft_dim,
            gnn_dim=gnn_dim,
            hidden_dims=hidden_dims,
            dropout=0.1,
            use_attention_fusion=True,
            focal_loss_alpha=0.25,
            focal_loss_gamma=2.0
        )
        
        # Check basic properties
        assert fusion_head.tft_dim == tft_dim
        assert fusion_head.gnn_dim == gnn_dim
        assert fusion_head.use_attention_fusion == True
        
        # Check components
        assert hasattr(fusion_head, 'fusion')  # Attention fusion
        assert hasattr(fusion_head, 'mlp')
        assert hasattr(fusion_head, 'focal_loss')
        assert hasattr(fusion_head, 'tft_aux_head')
        assert hasattr(fusion_head, 'gnn_aux_head')
    
    def test_fusion_head_forward_without_targets(self):
        """Test FusionHead forward pass without targets (inference)."""
        batch_size = 4
        tft_dim = 256
        gnn_dim = 64
        hidden_dims = [256, 64]
        
        fusion_head = FusionHead(
            tft_dim=tft_dim,
            gnn_dim=gnn_dim,
            hidden_dims=hidden_dims,
            dropout=0.1,
            use_attention_fusion=True
        )
        
        tft_features = torch.randn(batch_size, tft_dim)
        gnn_features = torch.randn(batch_size, gnn_dim)
        
        outputs = fusion_head(tft_features, gnn_features, return_attention=True)
        
        # Check required outputs
        assert 'logits' in outputs
        assert 'probabilities' in outputs
        assert 'tft_aux_logits' in outputs
        assert 'gnn_aux_logits' in outputs
        assert 'tft_aux_probabilities' in outputs
        assert 'gnn_aux_probabilities' in outputs
        
        # Check shapes
        assert outputs['logits'].shape == (batch_size,)
        assert outputs['probabilities'].shape == (batch_size,)
        assert outputs['tft_aux_logits'].shape == (batch_size,)
        assert outputs['gnn_aux_logits'].shape == (batch_size,)
        
        # Check probability ranges
        assert (outputs['probabilities'] >= 0).all()
        assert (outputs['probabilities'] <= 1).all()
        assert (outputs['tft_aux_probabilities'] >= 0).all()
        assert (outputs['tft_aux_probabilities'] <= 1).all()
        assert (outputs['gnn_aux_probabilities'] >= 0).all()
        assert (outputs['gnn_aux_probabilities'] <= 1).all()
        
        # Check for numerical stability
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                assert not torch.isnan(value).any(), f"NaN found in {key}"
                assert not torch.isinf(value).any(), f"Inf found in {key}"
    
    def test_fusion_head_forward_with_targets(self):
        """Test FusionHead forward pass with targets (training)."""
        batch_size = 4
        tft_dim = 256
        gnn_dim = 64
        
        fusion_head = FusionHead(
            tft_dim=tft_dim,
            gnn_dim=gnn_dim,
            hidden_dims=[128, 32],
            dropout=0.1,
            use_attention_fusion=True
        )
        
        tft_features = torch.randn(batch_size, tft_dim)
        gnn_features = torch.randn(batch_size, gnn_dim)
        targets = torch.randint(0, 2, (batch_size,)).float()
        
        outputs = fusion_head(tft_features, gnn_features, targets=targets, return_attention=True)
        
        # Check all outputs are present
        required_keys = [
            'logits', 'probabilities', 'tft_aux_logits', 'gnn_aux_logits',
            'tft_aux_probabilities', 'gnn_aux_probabilities',
            'loss', 'main_loss', 'tft_aux_loss', 'gnn_aux_loss'
        ]
        
        for key in required_keys:
            assert key in outputs, f"Missing key: {key}"
        
        # Check loss components
        assert outputs['loss'].shape == torch.Size([])  # Scalar
        assert outputs['main_loss'].shape == torch.Size([])
        assert outputs['tft_aux_loss'].shape == torch.Size([])
        assert outputs['gnn_aux_loss'].shape == torch.Size([])
        
        # Check loss values are reasonable
        assert outputs['loss'].item() >= 0
        assert outputs['main_loss'].item() >= 0
        assert outputs['tft_aux_loss'].item() >= 0
        assert outputs['gnn_aux_loss'].item() >= 0
        
        # Total loss should be combination of components
        expected_loss = outputs['main_loss'] + 0.3 * outputs['tft_aux_loss'] + 0.3 * outputs['gnn_aux_loss']
        assert torch.allclose(outputs['loss'], expected_loss, atol=1e-6)
    
    def test_fusion_head_without_attention_fusion(self):
        """Test FusionHead with simple concatenation instead of attention fusion."""
        batch_size = 4
        tft_dim = 128
        gnn_dim = 32
        
        fusion_head = FusionHead(
            tft_dim=tft_dim,
            gnn_dim=gnn_dim,
            hidden_dims=[64, 16],
            dropout=0.1,
            use_attention_fusion=False  # Simple concatenation
        )
        
        tft_features = torch.randn(batch_size, tft_dim)
        gnn_features = torch.randn(batch_size, gnn_dim)
        
        outputs = fusion_head(tft_features, gnn_features)
        
        # Should still work
        assert 'logits' in outputs
        assert 'probabilities' in outputs
        assert outputs['logits'].shape == (batch_size,)
        assert outputs['probabilities'].shape == (batch_size,)
        
        # Attention info should be empty
        assert 'attention_weights' not in outputs or outputs['attention_weights'] == {}
    
    def test_fusion_head_predict_method(self):
        """Test FusionHead predict method."""
        batch_size = 4
        tft_dim = 128
        gnn_dim = 32
        
        fusion_head = FusionHead(
            tft_dim=tft_dim,
            gnn_dim=gnn_dim,
            hidden_dims=[64, 16],
            dropout=0.0  # No dropout for consistent results
        )
        
        tft_features = torch.randn(batch_size, tft_dim)
        gnn_features = torch.randn(batch_size, gnn_dim)
        
        # Test predict method
        predictions = fusion_head.predict(tft_features, gnn_features, threshold=0.5)
        
        assert predictions.shape == (batch_size,)
        assert predictions.dtype == torch.long
        assert ((predictions == 0) | (predictions == 1)).all()  # Should be 0 or 1
    
    def test_fusion_head_gradients(self):
        """Test gradient flow through FusionHead."""
        batch_size = 2
        tft_dim = 64
        gnn_dim = 32
        
        fusion_head = FusionHead(
            tft_dim=tft_dim,
            gnn_dim=gnn_dim,
            hidden_dims=[32, 8],
            dropout=0.1
        )
        
        tft_features = torch.randn(batch_size, tft_dim, requires_grad=True)
        gnn_features = torch.randn(batch_size, gnn_dim, requires_grad=True)
        targets = torch.randint(0, 2, (batch_size,)).float()
        
        outputs = fusion_head(tft_features, gnn_features, targets=targets)
        loss = outputs['loss']
        
        loss.backward()
        
        # Check gradients exist
        assert tft_features.grad is not None
        assert gnn_features.grad is not None
        
        # Check gradients are not all zero
        assert not torch.allclose(tft_features.grad, torch.zeros_like(tft_features.grad))
        assert not torch.allclose(gnn_features.grad, torch.zeros_like(gnn_features.grad))
        
        # Check model parameter gradients
        model_has_grad = False
        for param in fusion_head.parameters():
            if param.grad is not None and not torch.allclose(param.grad, torch.zeros_like(param.grad)):
                model_has_grad = True
                break
        assert model_has_grad, "Model parameters should have non-zero gradients"
    
    def test_fusion_head_different_batch_sizes(self):
        """Test FusionHead with different batch sizes."""
        tft_dim = 128
        gnn_dim = 64
        
        fusion_head = FusionHead(
            tft_dim=tft_dim,
            gnn_dim=gnn_dim,
            hidden_dims=[64, 16],
            dropout=0.1
        )
        
        # Test different batch sizes
        for batch_size in [1, 4, 8, 16]:
            tft_features = torch.randn(batch_size, tft_dim)
            gnn_features = torch.randn(batch_size, gnn_dim)
            
            outputs = fusion_head(tft_features, gnn_features)
            
            assert outputs['logits'].shape == (batch_size,)
            assert outputs['probabilities'].shape == (batch_size,)
    
    def test_fusion_head_numerical_stability(self):
        """Test FusionHead numerical stability with extreme values."""
        tft_dim = 64
        gnn_dim = 32
        
        fusion_head = FusionHead(
            tft_dim=tft_dim,
            gnn_dim=gnn_dim,
            hidden_dims=[32, 8],
            dropout=0.0
        )
        
        batch_size = 2
        
        # Test with extreme input values
        test_cases = [
            (torch.randn(batch_size, tft_dim) * 100, torch.randn(batch_size, gnn_dim) * 100),  # Large values
            (torch.randn(batch_size, tft_dim) * 1e-6, torch.randn(batch_size, gnn_dim) * 1e-6),  # Small values
            (torch.zeros(batch_size, tft_dim), torch.zeros(batch_size, gnn_dim)),  # Zero values
        ]
        
        for tft_features, gnn_features in test_cases:
            with torch.no_grad():
                outputs = fusion_head(tft_features, gnn_features)
            
            # Check for numerical issues
            assert not torch.isnan(outputs['probabilities']).any()
            assert not torch.isinf(outputs['probabilities']).any()
            assert (outputs['probabilities'] >= 0).all()
            assert (outputs['probabilities'] <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])