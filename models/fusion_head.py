"""
Fusion Head for combining TFT and GNN representations
Implements the final prediction layer with focal loss for sepsis prediction.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in sepsis prediction.
    Implementation of Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits [batch_size, 1] or [batch_size]
            targets: Ground truth labels [batch_size] (0 or 1)
        
        Returns:
            Focal loss value
        """
        # Ensure inputs are 1D
        if inputs.dim() > 1:
            inputs = inputs.squeeze(-1)
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        
        # Compute probabilities
        p_t = torch.sigmoid(inputs)
        p_t = torch.where(targets == 1, p_t, 1 - p_t)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha balancing
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AttentionFusion(nn.Module):
    """Attention-based fusion of TFT and GNN representations."""
    
    def __init__(self,
                 tft_dim: int,
                 gnn_dim: int,
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        
        self.tft_dim = tft_dim
        self.gnn_dim = gnn_dim
        self.hidden_dim = hidden_dim
        
        # Projection layers
        self.tft_proj = nn.Linear(tft_dim, hidden_dim)
        self.gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-attention between modalities
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tft_features: Tensor, gnn_features: Tensor) -> Tuple[Tensor, Dict]:
        """
        Fuse TFT and GNN features using attention.
        
        Args:
            tft_features: TFT output [batch_size, seq_len, tft_dim] or [batch_size, tft_dim]
            gnn_features: GNN output [batch_size, gnn_dim]
        
        Returns:
            Fused features and attention weights
        """
        batch_size = tft_features.size(0)
        
        # Handle different TFT input shapes
        if tft_features.dim() == 3:
            # Use last time step or pool over sequence
            tft_pooled = tft_features.mean(dim=1)  # Global average pooling
        else:
            tft_pooled = tft_features
        
        # Project to common dimension
        tft_proj = self.tft_proj(tft_pooled)  # [batch_size, hidden_dim]
        gnn_proj = self.gnn_proj(gnn_features)  # [batch_size, hidden_dim]
        
        # Stack for attention computation
        combined = torch.stack([tft_proj, gnn_proj], dim=1)  # [batch_size, 2, hidden_dim]
        
        # Self-attention across modalities
        attended, attention_weights = self.attention(combined, combined, combined)
        attended = self.layer_norm(attended + combined)
        attended = self.dropout(attended)
        
        # Cross-attention: TFT queries GNN, GNN queries TFT
        tft_cross, tft_cross_weights = self.cross_attention(
            tft_proj.unsqueeze(1), gnn_proj.unsqueeze(1), gnn_proj.unsqueeze(1)
        )
        gnn_cross, gnn_cross_weights = self.cross_attention(
            gnn_proj.unsqueeze(1), tft_proj.unsqueeze(1), tft_proj.unsqueeze(1)
        )
        
        # Combine original and cross-attended features
        tft_enhanced = tft_proj + tft_cross.squeeze(1)
        gnn_enhanced = gnn_proj + gnn_cross.squeeze(1)
        
        # Final fusion
        fused = attended.mean(dim=1)  # Pool over modalities
        enhanced_concat = torch.cat([tft_enhanced, gnn_enhanced], dim=-1)
        
        attention_info = {
            'self_attention': attention_weights,
            'tft_cross_attention': tft_cross_weights,
            'gnn_cross_attention': gnn_cross_weights
        }
        
        return fused, enhanced_concat, attention_info


class FusionHead(nn.Module):
    """
    Fusion head that combines TFT temporal features with GNN structural features
    for sepsis prediction.
    """
    
    def __init__(self,
                 tft_dim: int,
                 gnn_dim: int,
                 hidden_dims: List[int] = [256, 64],
                 dropout: float = 0.1,
                 use_attention_fusion: bool = True,
                 focal_loss_alpha: float = 0.25,
                 focal_loss_gamma: float = 2.0):
        super().__init__()
        
        self.tft_dim = tft_dim
        self.gnn_dim = gnn_dim
        self.hidden_dims = hidden_dims
        self.use_attention_fusion = use_attention_fusion
        
        # Fusion strategy
        if use_attention_fusion:
            self.fusion = AttentionFusion(
                tft_dim=tft_dim,
                gnn_dim=gnn_dim,
                hidden_dim=hidden_dims[0] // 2,
                dropout=dropout
            )
            # Input dimension after attention fusion
            fusion_dim = hidden_dims[0] // 2 + hidden_dims[0]  # fused + enhanced_concat
        else:
            # Simple concatenation
            self.fusion = None
            fusion_dim = tft_dim + gnn_dim
        
        # MLP layers
        layers = []
        prev_dim = fusion_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final prediction layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Loss function
        self.focal_loss = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        
        # Auxiliary losses for regularization
        self.tft_aux_head = nn.Sequential(
            nn.Linear(tft_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0] // 2, 1)
        )
        
        self.gnn_aux_head = nn.Sequential(
            nn.Linear(gnn_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0] // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self,
                tft_features: Tensor,
                gnn_features: Tensor,
                targets: Optional[Tensor] = None,
                return_attention: bool = False) -> Dict[str, Tensor]:
        """
        Forward pass of the fusion head.
        
        Args:
            tft_features: TFT encoded features [batch_size, tft_dim]
            gnn_features: GNN encoded features [batch_size, gnn_dim]
            targets: Ground truth labels for loss computation [batch_size]
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary containing predictions, losses, and optional attention weights
        """
        batch_size = tft_features.size(0)
        
        # Fusion strategy
        if self.use_attention_fusion:
            fused_features, enhanced_concat, attention_info = self.fusion(
                tft_features, gnn_features
            )
            # Combine attention-fused and enhanced features
            combined_features = torch.cat([fused_features, enhanced_concat], dim=-1)
        else:
            # Simple concatenation
            combined_features = torch.cat([tft_features, gnn_features], dim=-1)
            attention_info = {}
        
        # Main prediction
        logits = self.mlp(combined_features)
        probabilities = torch.sigmoid(logits)
        
        # Auxiliary predictions for regularization
        tft_aux_logits = self.tft_aux_head(tft_features)
        gnn_aux_logits = self.gnn_aux_head(gnn_features)
        
        tft_aux_probs = torch.sigmoid(tft_aux_logits)
        gnn_aux_probs = torch.sigmoid(gnn_aux_logits)
        
        results = {
            'logits': logits.squeeze(-1),
            'probabilities': probabilities.squeeze(-1),
            'tft_aux_logits': tft_aux_logits.squeeze(-1),
            'gnn_aux_logits': gnn_aux_logits.squeeze(-1),
            'tft_aux_probabilities': tft_aux_probs.squeeze(-1),
            'gnn_aux_probabilities': gnn_aux_probs.squeeze(-1),
        }
        
        # Compute losses if targets provided
        if targets is not None:
            # Main focal loss
            main_loss = self.focal_loss(logits.squeeze(-1), targets)
            
            # Auxiliary losses with reduced weight
            tft_aux_loss = self.focal_loss(tft_aux_logits.squeeze(-1), targets)
            gnn_aux_loss = self.focal_loss(gnn_aux_logits.squeeze(-1), targets)
            
            # Combined loss
            total_loss = main_loss + 0.3 * tft_aux_loss + 0.3 * gnn_aux_loss
            
            results.update({
                'loss': total_loss,
                'main_loss': main_loss,
                'tft_aux_loss': tft_aux_loss,
                'gnn_aux_loss': gnn_aux_loss,
            })
        
        # Add attention weights if requested
        if return_attention and self.use_attention_fusion:
            results['attention_weights'] = attention_info
        
        return results
    
    def predict_proba(self, tft_features: Tensor, gnn_features: Tensor) -> Tensor:
        """Get prediction probabilities."""
        with torch.no_grad():
            results = self.forward(tft_features, gnn_features)
            return results['probabilities']
    
    def predict(self, tft_features: Tensor, gnn_features: Tensor, threshold: float = 0.5) -> Tensor:
        """Get binary predictions."""
        probabilities = self.predict_proba(tft_features, gnn_features)
        return (probabilities > threshold).long()


class SepsisClassificationHead(nn.Module):
    """
    Complete classification head with temporal feature extraction from TFT output.
    Handles sequence-to-one prediction for sepsis classification.
    """
    
    def __init__(self,
                 tft_sequence_dim: int,
                 tft_hidden_dim: int,
                 gnn_dim: int,
                 sequence_length: int = 72,
                 pooling_strategy: str = 'attention',
                 **fusion_kwargs):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.pooling_strategy = pooling_strategy
        
        # Temporal pooling strategies
        if pooling_strategy == 'attention':
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=tft_hidden_dim,
                num_heads=8,
                batch_first=True
            )
            self.temporal_query = nn.Parameter(torch.randn(1, 1, tft_hidden_dim))
            
        elif pooling_strategy == 'last':
            # Use last time step
            pass
        elif pooling_strategy == 'mean':
            # Global average pooling
            pass
        elif pooling_strategy == 'max':
            # Global max pooling
            pass
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        # Fusion head
        self.fusion_head = FusionHead(
            tft_dim=tft_hidden_dim,
            gnn_dim=gnn_dim,
            **fusion_kwargs
        )
    
    def forward(self,
                tft_sequence: Tensor,
                gnn_features: Tensor,
                targets: Optional[Tensor] = None,
                return_attention: bool = False) -> Dict[str, Tensor]:
        """
        Forward pass with temporal pooling.
        
        Args:
            tft_sequence: TFT output sequence [batch_size, seq_len, hidden_dim]
            gnn_features: GNN features [batch_size, gnn_dim]
            targets: Ground truth labels [batch_size]
            return_attention: Whether to return attention weights
        
        Returns:
            Prediction results dictionary
        """
        batch_size = tft_sequence.size(0)
        
        # Temporal pooling
        if self.pooling_strategy == 'attention':
            # Attention-based pooling
            query = self.temporal_query.expand(batch_size, -1, -1)
            pooled_tft, temporal_attention = self.temporal_attention(
                query, tft_sequence, tft_sequence
            )
            pooled_tft = pooled_tft.squeeze(1)  # [batch_size, hidden_dim]
            
        elif self.pooling_strategy == 'last':
            pooled_tft = tft_sequence[:, -1, :]  # Last time step
            temporal_attention = None
            
        elif self.pooling_strategy == 'mean':
            pooled_tft = tft_sequence.mean(dim=1)  # Average pooling
            temporal_attention = None
            
        elif self.pooling_strategy == 'max':
            pooled_tft, _ = tft_sequence.max(dim=1)  # Max pooling
            temporal_attention = None
        
        # Apply fusion head
        results = self.fusion_head(
            tft_features=pooled_tft,
            gnn_features=gnn_features,
            targets=targets,
            return_attention=return_attention
        )
        
        # Add temporal attention if available
        if temporal_attention is not None and return_attention:
            if 'attention_weights' not in results:
                results['attention_weights'] = {}
            results['attention_weights']['temporal'] = temporal_attention
        
        return results


if __name__ == "__main__":
    # Example usage and testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test parameters
    batch_size = 32
    seq_length = 72
    tft_hidden = 256
    gnn_dim = 64
    
    # Create model
    model = SepsisClassificationHead(
        tft_sequence_dim=seq_length,
        tft_hidden_dim=tft_hidden,
        gnn_dim=gnn_dim,
        sequence_length=seq_length,
        pooling_strategy='attention',
        hidden_dims=[256, 64],
        dropout=0.1,
        use_attention_fusion=True
    ).to(device)
    
    # Create dummy data
    tft_sequence = torch.randn(batch_size, seq_length, tft_hidden).to(device)
    gnn_features = torch.randn(batch_size, gnn_dim).to(device)
    targets = torch.randint(0, 2, (batch_size,)).to(device)
    
    # Forward pass with training
    model.train()
    results = model(tft_sequence, gnn_features, targets, return_attention=True)
    
    print("Training Results:")
    print(f"Logits shape: {results['logits'].shape}")
    print(f"Probabilities shape: {results['probabilities'].shape}")
    print(f"Total loss: {results['loss'].item():.4f}")
    print(f"Main loss: {results['main_loss'].item():.4f}")
    
    # Forward pass without targets (inference)
    model.eval()
    with torch.no_grad():
        inference_results = model(tft_sequence, gnn_features, return_attention=True)
        predictions = model.fusion_head.predict(
            tft_sequence.mean(dim=1), gnn_features, threshold=0.5
        )
    
    print(f"\nInference Results:")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predicted probabilities: {inference_results['probabilities'][:5]}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
