"""
Heterogeneous Graph Neural Network for Medical Data
Implements patient-stay-day hierarchy with lab and vital edge types.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor


class MedicalHeteroGATConv(MessagePassing):
    """
    Heterogeneous GAT layer specialized for medical data.
    Handles different node and edge types in the patient-stay-day hierarchy.
    """
    
    def __init__(self,
                 in_channels: Union[int, Dict[str, int]],
                 out_channels: int,
                 heads: int = 1,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 add_self_loops: bool = False,
                 bias: bool = True):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        
        # Linear transformations for different node types
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, bias=False)
            self.lin_dst = Linear(in_channels[1], heads * out_channels, bias=False)
        
        # Attention mechanism
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Optional[Tuple[int, int]] = None,
                return_attention_weights: bool = False):
        
        H, C = self.heads, self.out_channels
        
        # Handle different input formats
        if isinstance(x, Tensor):
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:
            x_src, x_dst = x
            x_src = self.lin_src(x_src).view(-1, H, C)
            x_dst = self.lin_dst(x_dst).view(-1, H, C)
        
        # Compute attention coefficients
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
        alpha = alpha_src + alpha_dst
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Message passing
        out = self.propagate(edge_index, x=(x_src, x_dst), alpha=alpha, size=size)
        
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)
        
        if self.bias is not None:
            out = out + self.bias
        
        if return_attention_weights:
            # Simplified attention weight extraction
            return out, alpha
        else:
            return out
    
    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: Tensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        alpha = alpha_j + alpha_i
        alpha = F.softmax(alpha, dim=0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class HeteroGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for medical time series.
    
    Node types: 'patient', 'stay', 'day'
    Edge types: 'has_lab', 'has_vital', 'patient_to_stay', 'stay_to_day'
    """
    
    def __init__(self,
                 node_channels: Dict[str, int],
                 hidden_channels: int = 64,
                 out_channels: int = 64,
                 num_layers: int = 2,
                 heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.node_channels = node_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        
        # Input projections for different node types
        self.input_projections = nn.ModuleDict()
        for node_type, in_channels in node_channels.items():
            self.input_projections[node_type] = Linear(in_channels, hidden_channels)
        
        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            
            # Define edge types and their source/target node types
            edge_types = [
                ('patient', 'patient_to_stay', 'stay'),
                ('stay', 'stay_to_day', 'day'),
                ('day', 'has_lab', 'day'),      # self-loop with lab info
                ('day', 'has_vital', 'day'),    # self-loop with vital info
                ('stay', 'rev_patient_to_stay', 'patient'),  # reverse edges
                ('day', 'rev_stay_to_day', 'stay'),
            ]
            
            for src_type, edge_type, dst_type in edge_types:
                if i == 0:
                    in_channels = hidden_channels
                else:
                    in_channels = hidden_channels
                
                conv_dict[edge_type] = MedicalHeteroGATConv(
                    in_channels=in_channels,
                    out_channels=hidden_channels // heads if i < num_layers - 1 else out_channels // heads,
                    heads=heads,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False
                )
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        # Output projections
        self.output_projections = nn.ModuleDict()
        for node_type in node_channels.keys():
            self.output_projections[node_type] = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_channels, out_channels)
            )
        
        # Global pooling layers for different node types
        self.global_pools = nn.ModuleDict()
        for node_type in node_channels.keys():
            self.global_pools[node_type] = nn.Sequential(
                nn.Linear(out_channels, out_channels // 2),
                nn.ReLU(),
                nn.Linear(out_channels // 2, out_channels // 4)
            )
    
    def forward(self,
                x_dict: Dict[str, Tensor],
                edge_index_dict: Dict[str, Tensor],
                batch_dict: Optional[Dict[str, Tensor]] = None) -> Dict[str, Tensor]:
        """
        Forward pass of the heterogeneous GNN.
        
        Args:
            x_dict: Node features for each node type
            edge_index_dict: Edge indices for each edge type
            batch_dict: Batch assignment for each node type (for batching)
        
        Returns:
            Dictionary of processed node embeddings and pooled representations
        """
        
        # Project input features
        x_dict = {
            node_type: self.input_projections[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        # Store attention weights for interpretability
        attention_weights = {}
        
        # Apply heterogeneous convolutions
        for i, conv in enumerate(self.convs):
            x_dict_new = {}
            
            # Apply convolution for each node type
            for node_type in x_dict.keys():
                x_dict_new[node_type] = []
            
            # Process each edge type
            for edge_type, edge_index in edge_index_dict.items():
                if edge_type in conv.convs:
                    # Determine source and target node types
                    if edge_type == 'patient_to_stay':
                        src_type, dst_type = 'patient', 'stay'
                    elif edge_type == 'stay_to_day':
                        src_type, dst_type = 'stay', 'day'
                    elif edge_type == 'has_lab':
                        src_type, dst_type = 'day', 'day'
                    elif edge_type == 'has_vital':
                        src_type, dst_type = 'day', 'day'
                    elif edge_type == 'rev_patient_to_stay':
                        src_type, dst_type = 'stay', 'patient'
                    elif edge_type == 'rev_stay_to_day':
                        src_type, dst_type = 'day', 'stay'
                    else:
                        continue
                    
                    # Apply convolution
                    if src_type == dst_type:
                        # Self-loops
                        out = conv.convs[edge_type](x_dict[src_type], edge_index)
                    else:
                        # Heterogeneous edges
                        out = conv.convs[edge_type](
                            (x_dict[src_type], x_dict[dst_type]), 
                            edge_index
                        )
                    
                    if dst_type not in x_dict_new:
                        x_dict_new[dst_type] = []
                    x_dict_new[dst_type].append(out)
            
            # Aggregate messages for each node type
            for node_type in x_dict.keys():
                if node_type in x_dict_new and x_dict_new[node_type]:
                    # Sum all incoming messages
                    aggregated = torch.stack(x_dict_new[node_type], dim=0).sum(dim=0)
                    x_dict[node_type] = aggregated
                # If no incoming messages, keep original features
            
            # Apply activation and residual connection (except last layer)
            if i < len(self.convs) - 1:
                x_dict = {
                    node_type: F.relu(x) + (x if x.size(-1) == self.hidden_channels else x)
                    for node_type, x in x_dict.items()
                }
        
        # Apply output projections
        processed_dict = {}
        for node_type, x in x_dict.items():
            processed_dict[node_type] = self.output_projections[node_type](x)
        
        # Global pooling for graph-level representations
        pooled_dict = {}
        for node_type, x in processed_dict.items():
            if batch_dict is not None and node_type in batch_dict:
                # Batch-wise pooling
                batch = batch_dict[node_type]
                pooled = self._global_pool(x, batch)
            else:
                # Simple mean pooling
                pooled = x.mean(dim=0, keepdim=True)
            
            pooled_dict[node_type] = self.global_pools[node_type](pooled)
        
        return {
            'node_embeddings': processed_dict,
            'graph_embeddings': pooled_dict,
            'attention_weights': attention_weights
        }
    
    def _global_pool(self, x: Tensor, batch: Tensor) -> Tensor:
        """Global pooling operation for batched graphs."""
        # Simple manual implementation of scatter_mean for single batch
        # from torch_scatter import scatter_mean
        # return scatter_mean(x, batch, dim=0)
        
        if batch is None:
            return x.mean(dim=0, keepdim=True)
        
        # Manual scatter mean implementation
        unique_batch = torch.unique(batch)
        result = []
        for b in unique_batch:
            mask = batch == b
            result.append(x[mask].mean(dim=0))
        return torch.stack(result)


class MedicalGraphBuilder:
    """Utility class to build medical heterogeneous graphs from patient data."""
    
    @staticmethod
    def build_hetero_data(patient_data: Dict,
                         lab_data: Dict,
                         vital_data: Dict,
                         static_data: Dict) -> HeteroData:
        """
        Build heterogeneous graph data from medical records.
        
        Args:
            patient_data: Patient-level features
            lab_data: Laboratory measurements
            vital_data: Vital signs
            static_data: Static patient information
        
        Returns:
            HeteroData object representing the medical graph
        """
        data = HeteroData()
        
        # Node features
        data['patient'].x = torch.tensor(static_data['features'], dtype=torch.float)
        data['stay'].x = torch.tensor(patient_data['features'], dtype=torch.float)
        data['day'].x = torch.tensor(lab_data['features'], dtype=torch.float)
        
        # Edge indices (simplified - would need proper mapping in practice)
        # Patient to stay relationships
        data['patient', 'patient_to_stay', 'stay'].edge_index = torch.tensor(
            patient_data['patient_stay_edges'], dtype=torch.long
        )
        
        # Stay to day relationships
        data['stay', 'stay_to_day', 'day'].edge_index = torch.tensor(
            patient_data['stay_day_edges'], dtype=torch.long
        )
        
        # Lab measurements (self-loops on day nodes)
        data['day', 'has_lab', 'day'].edge_index = torch.tensor(
            lab_data['edges'], dtype=torch.long
        )
        
        # Vital measurements (self-loops on day nodes)
        data['day', 'has_vital', 'day'].edge_index = torch.tensor(
            vital_data['edges'], dtype=torch.long
        )
        
        # Reverse edges
        data['stay', 'rev_patient_to_stay', 'patient'].edge_index = \
            data['patient', 'patient_to_stay', 'stay'].edge_index[[1, 0]]
        
        data['day', 'rev_stay_to_day', 'stay'].edge_index = \
            data['stay', 'stay_to_day', 'day'].edge_index[[1, 0]]
        
        return data


if __name__ == "__main__":
    # Example usage and testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define node channels
    node_channels = {
        'patient': 20,  # static patient features
        'stay': 30,     # stay-level features
        'day': 50,      # daily aggregated features
    }
    
    # Create model
    model = HeteroGNN(
        node_channels=node_channels,
        hidden_channels=64,
        out_channels=64,
        num_layers=2,
        heads=4,
        dropout=0.1
    ).to(device)
    
    # Create dummy data
    num_patients = 100
    num_stays = 150
    num_days = 500
    
    x_dict = {
        'patient': torch.randn(num_patients, 20).to(device),
        'stay': torch.randn(num_stays, 30).to(device),
        'day': torch.randn(num_days, 50).to(device),
    }
    
    # Create dummy edge indices
    edge_index_dict = {
        'patient_to_stay': torch.randint(0, min(num_patients, num_stays), (2, 200)).to(device),
        'stay_to_day': torch.randint(0, min(num_stays, num_days), (2, 300)).to(device),
        'has_lab': torch.stack([torch.arange(num_days), torch.arange(num_days)]).to(device),
        'has_vital': torch.stack([torch.arange(num_days), torch.arange(num_days)]).to(device),
        'rev_patient_to_stay': torch.randint(0, min(num_stays, num_patients), (2, 200)).to(device),
        'rev_stay_to_day': torch.randint(0, min(num_days, num_stays), (2, 300)).to(device),
    }
    
    # Forward pass
    with torch.no_grad():
        outputs = model(x_dict, edge_index_dict)
        
        print("Node embeddings shapes:")
        for node_type, embedding in outputs['node_embeddings'].items():
            print(f"  {node_type}: {embedding.shape}")
        
        print("\nGraph embeddings shapes:")
        for node_type, embedding in outputs['graph_embeddings'].items():
            print(f"  {node_type}: {embedding.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
