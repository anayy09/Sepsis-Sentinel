"""
Temporal Fusion Transformer (TFT) Encoder Implementation
Based on Lim et al. 2021 "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GLU(nn.Module):
    """Gated Linear Unit activation function."""
    
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, input_size * 2)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        return F.glu(x, dim=-1)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) component."""
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: Optional[int] = None,
                 dropout: float = 0.1,
                 context_size: Optional[int] = None):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.context_size = context_size
        
        # Main pathway
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, self.output_size)
        
        # Context pathway (for static covariates)
        if context_size is not None:
            self.context_dense = nn.Linear(context_size, hidden_size, bias=False)
        
        # Gating and normalization
        self.glu = GLU(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_size)
        
        # Skip connection projection if needed
        if input_size != self.output_size:
            self.skip_projection = nn.Linear(input_size, self.output_size, bias=False)
        else:
            self.skip_projection = None
    
    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        # Main pathway
        residual = x
        x = self.dense1(x)
        
        # Add context if provided
        if context is not None and self.context_size is not None:
            x = x + self.context_dense(context)
        
        # Apply gating and nonlinearity
        x = self.glu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        
        # Skip connection
        if self.skip_projection is not None:
            residual = self.skip_projection(residual)
        
        return self.layer_norm(x + residual)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for feature selection."""
    
    def __init__(self,
                 input_sizes: Dict[str, int],
                 hidden_size: int,
                 dropout: float = 0.1,
                 context_size: Optional[int] = None):
        super().__init__()
        
        self.input_sizes = input_sizes
        self.hidden_size = hidden_size
        self.num_variables = len(input_sizes)
        
        # Individual variable processing
        self.variable_grns = nn.ModuleDict()
        for name, size in input_sizes.items():
            self.variable_grns[name] = GatedResidualNetwork(
                input_size=size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
                context_size=context_size
            )
        
        # Variable selection weights
        total_input_size = sum(input_sizes.values())
        self.selection_grn = GatedResidualNetwork(
            input_size=total_input_size,
            hidden_size=hidden_size,
            output_size=self.num_variables,
            dropout=dropout,
            context_size=context_size
        )
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, 
                variable_inputs: Dict[str, Tensor],
                context: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # Process each variable
        processed_vars = []
        for name in self.input_sizes.keys():
            processed = self.variable_grns[name](variable_inputs[name], context)
            processed_vars.append(processed)
        
        # Concatenate for selection weight computation
        flattened_inputs = torch.cat([
            var_input for var_input in variable_inputs.values()
        ], dim=-1)
        
        # Compute selection weights
        selection_weights = self.selection_grn(flattened_inputs, context)
        selection_weights = self.softmax(selection_weights)
        
        # Apply selection weights
        selected_vars = []
        for i, processed in enumerate(processed_vars):
            weight = selection_weights[..., i:i+1]
            selected_vars.append(weight * processed)
        
        # Combine selected variables
        combined = torch.stack(selected_vars, dim=-2)  # [batch, num_vars, hidden]
        combined = torch.sum(combined, dim=-2)  # [batch, hidden]
        
        return combined, selection_weights


class StaticCovariateEncoder(nn.Module):
    """Encodes static covariates and generates context vectors."""
    
    def __init__(self,
                 static_input_sizes: Dict[str, int],
                 hidden_size: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.static_input_sizes = static_input_sizes
        self.hidden_size = hidden_size
        
        # Static covariate encoders
        self.static_encoders = nn.ModuleDict()
        for name, size in static_input_sizes.items():
            self.static_encoders[name] = nn.Linear(size, hidden_size)
        
        # Context networks for different components
        total_static_size = len(static_input_sizes) * hidden_size
        
        self.enrichment_grn = GatedResidualNetwork(
            input_size=total_static_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )
        
        self.selection_grn = GatedResidualNetwork(
            input_size=total_static_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )
        
        self.lstm_grn = GatedResidualNetwork(
            input_size=total_static_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )
        
        self.attention_grn = GatedResidualNetwork(
            input_size=total_static_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )
    
    def forward(self, static_inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Encode static variables
        encoded_statics = []
        for name in self.static_input_sizes.keys():
            encoded = self.static_encoders[name](static_inputs[name])
            encoded_statics.append(encoded)
        
        # Concatenate all static features
        combined_static = torch.cat(encoded_statics, dim=-1)
        
        # Generate context vectors for different components
        contexts = {
            'enrichment': self.enrichment_grn(combined_static),
            'selection': self.selection_grn(combined_static),
            'lstm': self.lstm_grn(combined_static),
            'attention': self.attention_grn(combined_static)
        }
        
        return contexts


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with interpretability."""
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len = query.size(0), query.size(1)
        residual = query
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        # Average attention weights across heads for interpretability
        avg_attention = attention_weights.mean(dim=1)  # [batch, seq_len, seq_len]
        
        return output, avg_attention


class TFTEncoder(nn.Module):
    """Temporal Fusion Transformer Encoder."""
    
    def __init__(self,
                 static_input_sizes: Dict[str, int],
                 temporal_input_sizes: Dict[str, int],
                 hidden_size: int = 256,
                 num_heads: int = 8,
                 num_lstm_layers: int = 2,
                 dropout: float = 0.1,
                 seq_len: int = 72):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        
        # Static covariate encoder
        self.static_encoder = StaticCovariateEncoder(
            static_input_sizes=static_input_sizes,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # Temporal variable selection
        self.temporal_selection = VariableSelectionNetwork(
            input_sizes=temporal_input_sizes,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size
        )
        
        # LSTM encoder/decoder
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Gated skip connection for LSTM
        self.lstm_skip_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # Static enrichment layer
        self.static_enrichment = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size
        )
        
        # Self-attention layer
        self.self_attention = InterpretableMultiHeadAttention(
            d_model=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Position-wise feed forward
        self.position_wise_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size * 4,
            output_size=hidden_size,
            dropout=dropout
        )
    
    def forward(self,
                static_inputs: Dict[str, Tensor],
                temporal_inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Forward pass of TFT encoder.
        
        Args:
            static_inputs: Dictionary of static features [batch_size, feature_dim]
            temporal_inputs: Dictionary of temporal features [batch_size, seq_len, feature_dim]
        
        Returns:
            Dictionary containing:
                - 'encoded': Final encoded representation [batch_size, seq_len, hidden_size]
                - 'attention_weights': Attention weights for interpretability
                - 'variable_selection_weights': Variable selection weights
        """
        batch_size = next(iter(temporal_inputs.values())).size(0)
        
        # Encode static covariates
        static_contexts = self.static_encoder(static_inputs)
        
        # Temporal variable selection
        temporal_selected_list = []
        variable_selection_weights_list = []
        
        for t in range(self.seq_len):
            # Extract features at time t
            temporal_t = {
                name: temporal_inputs[name][:, t, :]
                for name in temporal_inputs.keys()
            }
            
            # Apply variable selection
            selected, selection_weights = self.temporal_selection(
                temporal_t, 
                static_contexts['selection']
            )
            
            temporal_selected_list.append(selected.unsqueeze(1))
            variable_selection_weights_list.append(selection_weights.unsqueeze(1))
        
        # Concatenate temporal features
        temporal_selected = torch.cat(temporal_selected_list, dim=1)  # [batch, seq_len, hidden]
        variable_selection_weights = torch.cat(variable_selection_weights_list, dim=1)
        
        # LSTM encoding
        lstm_input = temporal_selected
        lstm_output, _ = self.lstm_encoder(lstm_input)
        
        # LSTM skip connection
        lstm_skip = self.lstm_skip_grn(temporal_selected)
        lstm_output = lstm_output + lstm_skip
        
        # Static enrichment
        enriched = self.static_enrichment(
            lstm_output, 
            static_contexts['enrichment'].unsqueeze(1).expand(-1, self.seq_len, -1)
        )
        
        # Self-attention
        attended, attention_weights = self.self_attention(
            enriched, enriched, enriched
        )
        
        # Position-wise feed forward
        encoded = self.position_wise_grn(attended)
        
        return {
            'encoded': encoded,
            'attention_weights': attention_weights,
            'variable_selection_weights': variable_selection_weights,
            'static_contexts': static_contexts
        }


if __name__ == "__main__":
    # Example usage and testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define input sizes
    static_input_sizes = {
        'demographics': 10,  # age, gender, race, etc.
        'admission': 5,      # admission type, location, etc.
    }
    
    temporal_input_sizes = {
        'vitals': 15,        # heart rate, blood pressure, etc.
        'labs': 25,          # lab values
        'waveforms': 20,     # waveform features
    }
    
    # Create model
    model = TFTEncoder(
        static_input_sizes=static_input_sizes,
        temporal_input_sizes=temporal_input_sizes,
        hidden_size=256,
        num_heads=8,
        seq_len=72
    ).to(device)
    
    # Create dummy data
    batch_size = 32
    
    static_inputs = {
        'demographics': torch.randn(batch_size, 10).to(device),
        'admission': torch.randn(batch_size, 5).to(device),
    }
    
    temporal_inputs = {
        'vitals': torch.randn(batch_size, 72, 15).to(device),
        'labs': torch.randn(batch_size, 72, 25).to(device),
        'waveforms': torch.randn(batch_size, 72, 20).to(device),
    }
    
    # Forward pass
    with torch.no_grad():
        outputs = model(static_inputs, temporal_inputs)
        
        print(f"Encoded shape: {outputs['encoded'].shape}")
        print(f"Attention weights shape: {outputs['attention_weights'].shape}")
        print(f"Variable selection weights shape: {outputs['variable_selection_weights'].shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
