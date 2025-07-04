"""
PyTorch Lightning Module for Sepsis Sentinel Training
Handles training, validation, and testing workflows.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torchmetrics import AUROC, AveragePrecision, Accuracy, Precision, Recall, F1Score
from torchmetrics.classification import BinaryConfusionMatrix
import wandb

from ..models.tft_encoder import TFTEncoder
from ..models.hetero_gnn import HeteroGNN
from ..models.fusion_head import SepsisClassificationHead

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SepsisSentinelLightning(pl.LightningModule):
    """
    PyTorch Lightning module for the complete Sepsis Sentinel model.
    """
    
    def __init__(self,
                 # Model architecture parameters
                 static_input_sizes: Dict[str, int],
                 temporal_input_sizes: Dict[str, int],
                 gnn_node_channels: Dict[str, int],
                 tft_hidden_size: int = 256,
                 gnn_hidden_channels: int = 64,
                 fusion_hidden_dims: List[int] = [256, 64],
                 num_heads: int = 8,
                 num_lstm_layers: int = 2,
                 gnn_num_layers: int = 2,
                 gnn_heads: int = 4,
                 seq_len: int = 72,
                 dropout: float = 0.1,
                 # Training parameters
                 learning_rate: float = 3e-4,
                 weight_decay: float = 1e-2,
                 scheduler_type: str = "cosine",
                 warmup_steps: int = 1000,
                 # Loss parameters
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 aux_loss_weight: float = 0.3,
                 # Monitoring parameters
                 monitor_metric: str = "val_auroc",
                 save_top_k: int = 3,
                 **kwargs):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Initialize models
        self.tft_encoder = TFTEncoder(
            static_input_sizes=static_input_sizes,
            temporal_input_sizes=temporal_input_sizes,
            hidden_size=tft_hidden_size,
            num_heads=num_heads,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            seq_len=seq_len
        )
        
        self.gnn_model = HeteroGNN(
            node_channels=gnn_node_channels,
            hidden_channels=gnn_hidden_channels,
            out_channels=gnn_hidden_channels,
            num_layers=gnn_num_layers,
            heads=gnn_heads,
            dropout=dropout
        )
        
        self.classification_head = SepsisClassificationHead(
            tft_sequence_dim=seq_len,
            tft_hidden_dim=tft_hidden_size,
            gnn_dim=gnn_hidden_channels,
            sequence_length=seq_len,
            pooling_strategy='attention',
            hidden_dims=fusion_hidden_dims,
            dropout=dropout,
            focal_loss_alpha=focal_alpha,
            focal_loss_gamma=focal_gamma
        )
        
        # Initialize metrics
        self._init_metrics()
        
        # Training state
        self.automatic_optimization = True
        
    def _init_metrics(self):
        """Initialize evaluation metrics."""
        metric_kwargs = {"task": "binary"}
        
        # Main metrics
        self.train_auroc = AUROC(**metric_kwargs)
        self.val_auroc = AUROC(**metric_kwargs)
        self.test_auroc = AUROC(**metric_kwargs)
        
        self.train_auprc = AveragePrecision(**metric_kwargs)
        self.val_auprc = AveragePrecision(**metric_kwargs)
        self.test_auprc = AveragePrecision(**metric_kwargs)
        
        # Additional metrics
        self.train_accuracy = Accuracy(**metric_kwargs)
        self.val_accuracy = Accuracy(**metric_kwargs)
        self.test_accuracy = Accuracy(**metric_kwargs)
        
        self.train_precision = Precision(**metric_kwargs)
        self.val_precision = Precision(**metric_kwargs)
        self.test_precision = Precision(**metric_kwargs)
        
        self.train_recall = Recall(**metric_kwargs)
        self.val_recall = Recall(**metric_kwargs)
        self.test_recall = Recall(**metric_kwargs)
        
        self.train_f1 = F1Score(**metric_kwargs)
        self.val_f1 = F1Score(**metric_kwargs)
        self.test_f1 = F1Score(**metric_kwargs)
        
        # Confusion matrices
        self.train_cm = BinaryConfusionMatrix()
        self.val_cm = BinaryConfusionMatrix()
        self.test_cm = BinaryConfusionMatrix()
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """
        Forward pass of the complete model.
        
        Args:
            batch: Batch containing TFT and GNN inputs
        
        Returns:
            Model outputs including predictions and intermediate representations
        """
        # Extract TFT inputs
        tft_static = {
            key: batch[f'tft_static_{key}'] 
            for key in self.hparams.static_input_sizes.keys()
        }
        
        tft_temporal = {
            key: batch[f'tft_temporal_{key}']
            for key in self.hparams.temporal_input_sizes.keys()
        }
        
        # TFT forward pass
        tft_outputs = self.tft_encoder(tft_static, tft_temporal)
        tft_encoded = tft_outputs['encoded']  # [batch, seq_len, hidden]
        
        # Extract GNN inputs
        gnn_x_dict = {
            node_type: batch[f'gnn_x_{node_type}']
            for node_type in self.hparams.gnn_node_channels.keys()
        }
        
        gnn_edge_dict = {
            edge_type: batch[f'gnn_edge_{edge_type}']
            for edge_type in ['patient_to_stay', 'stay_to_day', 'has_lab', 'has_vital',
                             'rev_patient_to_stay', 'rev_stay_to_day']
        }
        
        gnn_batch_dict = batch.get('gnn_batch_dict', None)
        
        # GNN forward pass
        gnn_outputs = self.gnn_model(gnn_x_dict, gnn_edge_dict, gnn_batch_dict)
        
        # Use patient-level graph embeddings
        gnn_features = gnn_outputs['graph_embeddings']['patient']
        
        # Classification forward pass
        targets = batch.get('targets', None)
        classification_outputs = self.classification_head(
            tft_sequence=tft_encoded,
            gnn_features=gnn_features,
            targets=targets,
            return_attention=True
        )
        
        # Combine all outputs
        outputs = {
            'tft_outputs': tft_outputs,
            'gnn_outputs': gnn_outputs,
            'classification_outputs': classification_outputs,
            'predictions': classification_outputs['probabilities'],
            'logits': classification_outputs['logits']
        }
        
        if targets is not None:
            outputs['loss'] = classification_outputs['loss']
        
        return outputs
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Training step."""
        outputs = self(batch)
        loss = outputs['loss']
        predictions = outputs['predictions']
        targets = batch['targets']
        
        # Update metrics
        self.train_auroc(predictions, targets)
        self.train_auprc(predictions, targets)
        self.train_accuracy(predictions, targets)
        self.train_precision(predictions, targets)
        self.train_recall(predictions, targets)
        self.train_f1(predictions, targets)
        self.train_cm(predictions, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_auprc', self.train_auprc, on_step=False, on_epoch=True)
        
        # Log auxiliary losses if available
        if 'main_loss' in outputs['classification_outputs']:
            self.log('train_main_loss', outputs['classification_outputs']['main_loss'], on_step=True, on_epoch=True)
            self.log('train_tft_aux_loss', outputs['classification_outputs']['tft_aux_loss'], on_step=True, on_epoch=True)
            self.log('train_gnn_aux_loss', outputs['classification_outputs']['gnn_aux_loss'], on_step=True, on_epoch=True)
        
        return {'loss': loss, 'predictions': predictions.detach(), 'targets': targets.detach()}
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Validation step."""
        outputs = self(batch)
        loss = outputs['loss']
        predictions = outputs['predictions']
        targets = batch['targets']
        
        # Update metrics
        self.val_auroc(predictions, targets)
        self.val_auprc(predictions, targets)
        self.val_accuracy(predictions, targets)
        self.val_precision(predictions, targets)
        self.val_recall(predictions, targets)
        self.val_f1(predictions, targets)
        self.val_cm(predictions, targets)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_auprc', self.val_auprc, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'val_loss': loss, 'predictions': predictions, 'targets': targets}
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Test step."""
        outputs = self(batch)
        loss = outputs['loss']
        predictions = outputs['predictions']
        targets = batch['targets']
        
        # Update metrics
        self.test_auroc(predictions, targets)
        self.test_auprc(predictions, targets)
        self.test_accuracy(predictions, targets)
        self.test_precision(predictions, targets)
        self.test_recall(predictions, targets)
        self.test_f1(predictions, targets)
        self.test_cm(predictions, targets)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_auroc', self.test_auroc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_auprc', self.test_auprc, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'test_loss': loss, 'predictions': predictions, 'targets': targets}
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Log confusion matrix
        cm = self.train_cm.compute()
        self.logger.experiment.log({
            "train_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=cm.cpu().numpy().flatten(),
                preds=None,
                class_names=["No Sepsis", "Sepsis"]
            )
        })
        
        # Reset metrics
        self.train_cm.reset()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Log all validation metrics
        metrics = {
            'val_accuracy': self.val_accuracy.compute(),
            'val_precision': self.val_precision.compute(),
            'val_recall': self.val_recall.compute(),
            'val_f1': self.val_f1.compute()
        }
        
        for name, value in metrics.items():
            self.log(name, value, sync_dist=True)
        
        # Log confusion matrix
        cm = self.val_cm.compute()
        if self.logger and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({
                "val_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=cm.cpu().numpy().flatten(),
                    preds=None,
                    class_names=["No Sepsis", "Sepsis"]
                )
            })
        
        # Reset metrics
        self.val_cm.reset()
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        # Log all test metrics
        metrics = {
            'test_accuracy': self.test_accuracy.compute(),
            'test_precision': self.test_precision.compute(),
            'test_recall': self.test_recall.compute(),
            'test_f1': self.test_f1.compute()
        }
        
        for name, value in metrics.items():
            self.log(name, value, sync_dist=True)
        
        # Log confusion matrix and final results
        cm = self.test_cm.compute()
        if self.logger and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({
                "test_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=cm.cpu().numpy().flatten(),
                    preds=None,
                    class_names=["No Sepsis", "Sepsis"]
                ),
                "final_test_results": {
                    "auroc": self.test_auroc.compute().item(),
                    "auprc": self.test_auprc.compute().item(),
                    "accuracy": metrics['test_accuracy'].item(),
                    "precision": metrics['test_precision'].item(),
                    "recall": metrics['test_recall'].item(),
                    "f1": metrics['test_f1'].item()
                }
            })
        
        # Reset metrics
        self.test_cm.reset()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        
        # Group parameters for different learning rates
        tft_params = list(self.tft_encoder.parameters())
        gnn_params = list(self.gnn_model.parameters())
        fusion_params = list(self.classification_head.parameters())
        
        # Use different learning rates for different components
        param_groups = [
            {'params': tft_params, 'lr': self.hparams.learning_rate},
            {'params': gnn_params, 'lr': self.hparams.learning_rate * 0.5},  # Lower LR for GNN
            {'params': fusion_params, 'lr': self.hparams.learning_rate}
        ]
        
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
            eps=1e-8
        )
        
        # Configure scheduler
        if self.hparams.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.hparams.learning_rate * 0.01
            )
        elif self.hparams.scheduler_type == "cosine_warmup":
            from torch.optim.lr_scheduler import LinearLR, SequentialLR
            
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.hparams.warmup_steps
            )
            
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs - self.hparams.warmup_steps,
                eta_min=self.hparams.learning_rate * 0.01
            )
            
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.hparams.warmup_steps]
            )
        elif self.hparams.scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                threshold=0.001,
                min_lr=self.hparams.learning_rate * 0.001
            )
        else:
            scheduler = None
        
        if scheduler is None:
            return optimizer
        
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
        
        if self.hparams.scheduler_type == "plateau":
            scheduler_config["monitor"] = self.hparams.monitor_metric
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Tensor]:
        """Prediction step for inference."""
        outputs = self(batch)
        
        return {
            'predictions': outputs['predictions'],
            'logits': outputs['logits'],
            'attention_weights': outputs.get('attention_weights', None)
        }
    
    def get_model_summary(self) -> str:
        """Get a summary of the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        tft_params = sum(p.numel() for p in self.tft_encoder.parameters())
        gnn_params = sum(p.numel() for p in self.gnn_model.parameters())
        fusion_params = sum(p.numel() for p in self.classification_head.parameters())
        
        summary = f"""
Sepsis Sentinel Model Summary:
==============================
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}

Component Breakdown:
- TFT Encoder: {tft_params:,} parameters
- GNN Model: {gnn_params:,} parameters
- Fusion Head: {fusion_params:,} parameters

Architecture:
- TFT Hidden Size: {self.hparams.tft_hidden_size}
- GNN Hidden Channels: {self.hparams.gnn_hidden_channels}
- Sequence Length: {self.hparams.seq_len}
- Number of Heads: {self.hparams.num_heads}
- Dropout: {self.hparams.dropout}
"""
        return summary


if __name__ == "__main__":
    # Example usage for testing
    static_input_sizes = {
        'demographics': 10,
        'admission': 5,
    }
    
    temporal_input_sizes = {
        'vitals': 15,
        'labs': 25,
        'waveforms': 20,
    }
    
    gnn_node_channels = {
        'patient': 20,
        'stay': 30,
        'day': 50,
    }
    
    model = SepsisSentinelLightning(
        static_input_sizes=static_input_sizes,
        temporal_input_sizes=temporal_input_sizes,
        gnn_node_channels=gnn_node_channels,
        learning_rate=3e-4
    )
    
    print(model.get_model_summary())
