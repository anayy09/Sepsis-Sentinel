"""
Main training script for Sepsis Sentinel model.
Handles data loading, model training, and evaluation.
"""

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
import wandb
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
    Timer
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.lightning_module import SepsisSentinelLightning
from data_pipeline.data_module import SepsisDataModule

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_callbacks(config: Dict) -> List[pl.Callback]:
    """Setup PyTorch Lightning callbacks."""
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor=config['training']['monitor_metric'],
        dirpath=config['training']['checkpoint_dir'],
        filename='sepsis-sentinel-{epoch:02d}-{val_auroc:.3f}',
        save_top_k=config['training']['save_top_k'],
        mode='max',
        save_last=True,
        save_weights_only=False,
        auto_insert_metric_name=False,
        every_n_epochs=1,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor=config['training']['monitor_metric'],
        min_delta=config['training']['early_stopping']['min_delta'],
        patience=config['training']['early_stopping']['patience'],
        mode='max',
        verbose=True,
        strict=True
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(
        logging_interval='step',
        log_momentum=False
    )
    callbacks.append(lr_monitor)
    
    # Stochastic Weight Averaging
    if config['training'].get('use_swa', False):
        swa_callback = StochasticWeightAveraging(
            swa_lrs=config['training']['learning_rate'] * 0.1,
            swa_epoch_start=config['training']['swa_epoch_start'],
            annealing_epochs=config['training']['swa_annealing_epochs']
        )
        callbacks.append(swa_callback)
    
    # Timer callback
    timer_callback = Timer(
        duration=config['training'].get('max_time', None),
        interval="epoch"
    )
    callbacks.append(timer_callback)
    
    return callbacks


def setup_logger(config: Dict) -> WandbLogger:
    """Setup Weights & Biases logger."""
    
    # Initialize wandb
    wandb.init(
        project=config['logging']['wandb_project'],
        entity=config['logging'].get('wandb_entity', None),
        name=config['logging']['experiment_name'],
        tags=config['logging'].get('tags', []),
        notes=config['logging'].get('notes', ''),
        config=config
    )
    
    # Create logger
    logger = WandbLogger(
        project=config['logging']['wandb_project'],
        entity=config['logging'].get('wandb_entity', None),
        name=config['logging']['experiment_name'],
        log_model='all',
        save_dir=config['logging']['log_dir']
    )
    
    return logger


def setup_strategy(config: Dict) -> Optional[pl.strategies.Strategy]:
    """Setup training strategy for multi-GPU training."""
    
    if config['training']['strategy'] == 'ddp' and torch.cuda.device_count() > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            static_graph=True
        )
        return strategy
    
    return None


def train_model(config: Dict, data_module: SepsisDataModule) -> pl.LightningModule:
    """Train the Sepsis Sentinel model."""
    
    logger.info("Initializing Sepsis Sentinel model...")
    
    # Initialize model
    model = SepsisSentinelLightning(
        # Architecture parameters
        static_input_sizes=config['model']['static_input_sizes'],
        temporal_input_sizes=config['model']['temporal_input_sizes'],
        gnn_node_channels=config['model']['gnn_node_channels'],
        tft_hidden_size=config['model']['tft_hidden_size'],
        gnn_hidden_channels=config['model']['gnn_hidden_channels'],
        fusion_hidden_dims=config['model']['fusion_hidden_dims'],
        num_heads=config['model']['num_heads'],
        num_lstm_layers=config['model']['num_lstm_layers'],
        gnn_num_layers=config['model']['gnn_num_layers'],
        gnn_heads=config['model']['gnn_heads'],
        seq_len=config['model']['seq_len'],
        dropout=config['model']['dropout'],
        # Training parameters
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        scheduler_type=config['training']['scheduler_type'],
        warmup_steps=config['training']['warmup_steps'],
        # Loss parameters
        focal_alpha=config['training']['focal_alpha'],
        focal_gamma=config['training']['focal_gamma'],
        aux_loss_weight=config['training']['aux_loss_weight'],
        # Monitoring parameters
        monitor_metric=config['training']['monitor_metric'],
        save_top_k=config['training']['save_top_k']
    )
    
    logger.info("Model initialized successfully")
    logger.info(model.get_model_summary())
    
    # Setup callbacks and logger
    callbacks = setup_callbacks(config)
    wandb_logger = setup_logger(config)
    strategy = setup_strategy(config)
    
    # Initialize trainer
    trainer = pl.Trainer(
        # Basic configuration
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training']['accelerator'],
        devices=config['training']['devices'],
        strategy=strategy or "auto",
        precision=config['training']['precision'],
        
        # Callbacks and logging
        callbacks=callbacks,
        logger=wandb_logger,
        
        # Optimization
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        
        # Validation and checkpointing
        check_val_every_n_epoch=config['training']['check_val_every_n_epoch'],
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        
        # Performance
        num_sanity_val_steps=2,
        detect_anomaly=False,
        deterministic=True,
        benchmark=True,
        
        # Debugging (disable in production)
        fast_dev_run=config['training'].get('fast_dev_run', False),
        limit_train_batches=config['training'].get('limit_train_batches', 1.0),
        limit_val_batches=config['training'].get('limit_val_batches', 1.0),
        limit_test_batches=config['training'].get('limit_test_batches', 1.0),
        
        # Profiler
        profiler=config['training'].get('profiler', None)
    )
    
    # Log model architecture to wandb
    wandb_logger.watch(model, log='all', log_freq=100, log_graph=True)
    
    logger.info("Starting training...")
    
    # Train model
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=config['training'].get('resume_from_checkpoint', None)
    )
    
    logger.info("Training completed!")
    
    # Test model
    if not config['training'].get('skip_test', False) and not config['training'].get('fast_dev_run', False):
        logger.info("Starting testing...")
        trainer.test(model, datamodule=data_module, ckpt_path='best')
        logger.info("Testing completed!")
    elif config['training'].get('fast_dev_run', False):
        logger.info("Skipping testing in fast_dev_run mode")
    
    return model


def evaluate_model(model: pl.LightningModule,
                  data_module: SepsisDataModule,
                  config: Dict) -> Dict:
    """Evaluate trained model and generate detailed metrics."""
    
    logger.info("Performing detailed model evaluation...")
    
    # Initialize trainer for evaluation
    trainer = pl.Trainer(
        accelerator=config['training']['accelerator'],
        devices=1,  # Use single device for evaluation
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True
    )
    
    # Get predictions on test set
    predictions = trainer.predict(model, data_module.test_dataloader())
    
    # Combine predictions
    all_predictions = torch.cat([p['predictions'] for p in predictions])
    all_logits = torch.cat([p['logits'] for p in predictions])
    
    # Get ground truth labels
    test_targets = []
    for batch in data_module.test_dataloader():
        test_targets.append(batch['targets'])
    all_targets = torch.cat(test_targets)
    
    # Calculate additional metrics
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, accuracy_score,
        precision_score, recall_score, f1_score, confusion_matrix,
        classification_report
    )
    import numpy as np
    
    predictions_np = all_predictions.cpu().numpy()
    targets_np = all_targets.cpu().numpy()
    
    # Binary predictions at different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_metrics = {}
    
    for threshold in thresholds:
        binary_preds = (predictions_np > threshold).astype(int)
        
        threshold_metrics[f'threshold_{threshold}'] = {
            'accuracy': accuracy_score(targets_np, binary_preds),
            'precision': precision_score(targets_np, binary_preds),
            'recall': recall_score(targets_np, binary_preds),
            'f1': f1_score(targets_np, binary_preds),
            'confusion_matrix': confusion_matrix(targets_np, binary_preds).tolist()
        }
    
    # Overall metrics
    evaluation_results = {
        'auroc': roc_auc_score(targets_np, predictions_np),
        'auprc': average_precision_score(targets_np, predictions_np),
        'threshold_metrics': threshold_metrics,
        'classification_report': classification_report(
            targets_np, 
            (predictions_np > 0.5).astype(int),
            output_dict=True
        )
    }
    
    # Log to wandb
    wandb.log({
        'final_evaluation': evaluation_results,
        'roc_curve': wandb.plot.roc_curve(targets_np, predictions_np),
        'pr_curve': wandb.plot.pr_curve(targets_np, predictions_np)
    })
    
    logger.info(f"Final AUROC: {evaluation_results['auroc']:.4f}")
    logger.info(f"Final AUPRC: {evaluation_results['auprc']:.4f}")
    
    return evaluation_results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Sepsis Sentinel model")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-path", 
        type=str, 
        required=True,
        help="Path to processed data"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./outputs",
        help="Output directory for models and logs"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with fast dev run"
    )
    parser.add_argument(
        "--test-only", 
        action="store_true",
        help="Only run testing with pre-trained model"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config['training']['resume_from_checkpoint'] = args.resume
    config['training']['fast_dev_run'] = args.debug
    config['logging']['log_dir'] = args.output_dir
    config['training']['checkpoint_dir'] = os.path.join(args.output_dir, 'checkpoints')
    
    if args.debug:
        config['training']['limit_train_batches'] = 0.01
        config['training']['limit_val_batches'] = 0.01
        config['training']['max_epochs'] = 2
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    # Set random seeds for reproducibility
    pl.seed_everything(config['training']['random_seed'], workers=True)
    
    # Initialize data module
    logger.info("Initializing data module...")
    data_module = SepsisDataModule(
        data_path=args.data_path,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        **config['data']
    )
    
    # Setup data module
    data_module.setup()
    logger.info(f"Data module initialized with {len(data_module.train_dataset)} training samples")
    
    if args.test_only:
        # Load pre-trained model and test
        if args.resume is None:
            raise ValueError("Must provide checkpoint path for test-only mode")
        
        model = SepsisSentinelLightning.load_from_checkpoint(args.resume)
        
        trainer = pl.Trainer(
            accelerator=config['training']['accelerator'],
            devices=1,
            logger=setup_logger(config),
            enable_checkpointing=False
        )
        
        trainer.test(model, datamodule=data_module)
        
    else:
        # Train model
        model = train_model(config, data_module)
        
        # Skip evaluation for now due to dimensional mismatch
        logger.info("Training completed successfully! Skipping detailed evaluation for now.")
        
        # # Evaluate model
        # evaluation_results = evaluate_model(model, data_module, config)
        
        # # Save evaluation results
        # import json
        # eval_path = os.path.join(args.output_dir, 'evaluation_results.json')
        # with open(eval_path, 'w') as f:
        #     json.dump(evaluation_results, f, indent=2)
        
        # logger.info(f"Evaluation results saved to: {eval_path}")
    
    # Finish wandb run
    wandb.finish()
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
