"""
Model evaluation and benchmarking utilities for Sepsis Sentinel.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from utils import Timer, setup_logging, save_json

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for sepsis prediction."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 threshold: float = 0.5):
        """
        Initialize evaluator.
        
        Args:
            model: Trained PyTorch model
            device: Device to run evaluation on
            threshold: Classification threshold
        """
        self.model = model
        self.device = device
        self.threshold = threshold
        self.model.eval()
        
        # Results storage
        self.results = {}
        self.predictions = {}
        self.metrics = {}
    
    def evaluate_dataset(self, 
                        dataloader: torch.utils.data.DataLoader,
                        dataset_name: str = "test") -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            dataset_name: Name of the dataset (for logging)
        
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating on {dataset_name} dataset...")
        
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_patient_ids = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Measure inference time
                start_time = time.time()
                
                # Forward pass
                outputs = self.model(batch)
                probabilities = outputs['predictions']  # Assuming model returns probabilities
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Get predictions and targets
                predictions = (probabilities > self.threshold).long()
                targets = batch['targets']
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Store patient IDs if available
                if 'patient_id' in batch:
                    all_patient_ids.extend(batch['patient_id'])
                else:
                    all_patient_ids.extend([f"patient_{batch_idx}_{i}" for i in range(len(targets))])
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        targets = np.array(all_targets)
        
        # Calculate metrics
        metrics = self._calculate_metrics(targets, predictions, probabilities)
        
        # Add performance metrics
        metrics['inference_time_mean'] = np.mean(inference_times)
        metrics['inference_time_std'] = np.std(inference_times)
        metrics['throughput_samples_per_second'] = len(targets) / sum(inference_times)
        
        # Store detailed results
        detailed_results = {
            'patient_ids': all_patient_ids,
            'targets': targets.tolist(),
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'metrics': metrics
        }
        
        self.results[dataset_name] = detailed_results
        self.metrics[dataset_name] = metrics
        
        logger.info(f"{dataset_name} evaluation completed:")
        logger.info(f"  AUROC: {metrics['auroc']:.4f}")
        logger.info(f"  AUPRC: {metrics['auprc']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1']:.4f}")
        logger.info(f"  Inference time: {metrics['inference_time_mean']:.4f}s ± {metrics['inference_time_std']:.4f}s")
        
        return detailed_results
    
    def _calculate_metrics(self, 
                          targets: np.ndarray, 
                          predictions: np.ndarray, 
                          probabilities: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(targets, predictions)
        metrics['precision'] = precision_score(targets, predictions, zero_division=0)
        metrics['recall'] = recall_score(targets, predictions, zero_division=0)
        metrics['f1'] = f1_score(targets, predictions, zero_division=0)
        metrics['specificity'] = self._calculate_specificity(targets, predictions)
        metrics['npv'] = self._calculate_npv(targets, predictions)
        
        # ROC and PR metrics
        try:
            metrics['auroc'] = roc_auc_score(targets, probabilities)
            metrics['auprc'] = average_precision_score(targets, probabilities)
        except ValueError:
            # Handle case where only one class is present
            metrics['auroc'] = 0.0
            metrics['auprc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
        
        # Balanced accuracy
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        # Class distribution
        metrics['positive_rate'] = np.mean(targets)
        metrics['prediction_positive_rate'] = np.mean(predictions)
        
        # Probability statistics
        metrics['prob_mean'] = np.mean(probabilities)
        metrics['prob_std'] = np.std(probabilities)
        metrics['prob_min'] = np.min(probabilities)
        metrics['prob_max'] = np.max(probabilities)
        
        return metrics
    
    def _calculate_specificity(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn = np.sum((targets == 0) & (predictions == 0))
        fp = np.sum((targets == 0) & (predictions == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _calculate_npv(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate negative predictive value."""
        tn = np.sum((targets == 0) & (predictions == 0))
        fn = np.sum((targets == 1) & (predictions == 0))
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    def threshold_analysis(self, 
                          dataset_name: str = "test",
                          thresholds: Optional[List[float]] = None) -> Dict:
        """
        Analyze performance across different thresholds.
        
        Args:
            dataset_name: Name of dataset to analyze
            thresholds: List of thresholds to evaluate
        
        Returns:
            Dictionary of threshold analysis results
        """
        if dataset_name not in self.results:
            raise ValueError(f"Dataset {dataset_name} not evaluated yet")
        
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1).tolist()
        
        targets = np.array(self.results[dataset_name]['targets'])
        probabilities = np.array(self.results[dataset_name]['probabilities'])
        
        threshold_results = []
        
        for threshold in thresholds:
            predictions = (probabilities > threshold).astype(int)
            metrics = self._calculate_metrics(targets, predictions, probabilities)
            metrics['threshold'] = threshold
            threshold_results.append(metrics)
        
        # Find optimal threshold based on F1 score
        f1_scores = [r['f1'] for r in threshold_results]
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        analysis_results = {
            'threshold_metrics': threshold_results,
            'optimal_threshold': optimal_threshold,
            'optimal_f1': f1_scores[optimal_idx]
        }
        
        logger.info(f"Optimal threshold for {dataset_name}: {optimal_threshold:.3f} (F1: {f1_scores[optimal_idx]:.4f})")
        
        return analysis_results
    
    def calibration_analysis(self, dataset_name: str = "test", n_bins: int = 10) -> Dict:
        """
        Analyze model calibration using reliability diagrams.
        
        Args:
            dataset_name: Name of dataset to analyze
            n_bins: Number of bins for calibration analysis
        
        Returns:
            Calibration analysis results
        """
        if dataset_name not in self.results:
            raise ValueError(f"Dataset {dataset_name} not evaluated yet")
        
        targets = np.array(self.results[dataset_name]['targets'])
        probabilities = np.array(self.results[dataset_name]['probabilities'])
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = targets[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                count_in_bin = in_bin.sum()
            else:
                accuracy_in_bin = 0
                avg_confidence_in_bin = 0
                count_in_bin = 0
            
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(count_in_bin)
        
        # Calculate Expected Calibration Error (ECE)
        ece = 0
        total_samples = len(targets)
        for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
            ece += (count / total_samples) * abs(acc - conf)
        
        calibration_results = {
            'bin_boundaries': bin_boundaries.tolist(),
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts,
            'expected_calibration_error': ece
        }
        
        logger.info(f"Expected Calibration Error for {dataset_name}: {ece:.4f}")
        
        return calibration_results
    
    def generate_plots(self, 
                      dataset_name: str = "test", 
                      output_dir: str = "./evaluation_plots") -> Dict[str, str]:
        """
        Generate evaluation plots.
        
        Args:
            dataset_name: Name of dataset to plot
            output_dir: Directory to save plots
        
        Returns:
            Dictionary mapping plot names to file paths
        """
        if dataset_name not in self.results:
            raise ValueError(f"Dataset {dataset_name} not evaluated yet")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        targets = np.array(self.results[dataset_name]['targets'])
        probabilities = np.array(self.results[dataset_name]['probabilities'])
        
        plot_paths = {}
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(targets, probabilities)
        auroc = roc_auc_score(targets, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUROC = {auroc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {dataset_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        roc_path = output_dir / f'roc_curve_{dataset_name}.png'
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['roc_curve'] = str(roc_path)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(targets, probabilities)
        auprc = average_precision_score(targets, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUPRC = {auprc:.3f})')
        plt.axhline(y=np.mean(targets), color='k', linestyle='--', label='Random')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {dataset_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        pr_path = output_dir / f'pr_curve_{dataset_name}.png'
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['pr_curve'] = str(pr_path)
        
        # Confusion Matrix
        predictions = (probabilities > self.threshold).astype(int)
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Sepsis', 'Sepsis'],
                   yticklabels=['No Sepsis', 'Sepsis'])
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = output_dir / f'confusion_matrix_{dataset_name}.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['confusion_matrix'] = str(cm_path)
        
        # Probability Distribution
        plt.figure(figsize=(10, 6))
        
        # Separate distributions for positive and negative cases
        pos_probs = probabilities[targets == 1]
        neg_probs = probabilities[targets == 0]
        
        plt.hist(neg_probs, bins=50, alpha=0.7, label='No Sepsis', density=True, color='blue')
        plt.hist(pos_probs, bins=50, alpha=0.7, label='Sepsis', density=True, color='red')
        plt.axvline(x=self.threshold, color='black', linestyle='--', label=f'Threshold ({self.threshold})')
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title(f'Probability Distribution - {dataset_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        dist_path = output_dir / f'probability_distribution_{dataset_name}.png'
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['probability_distribution'] = str(dist_path)
        
        logger.info(f"Evaluation plots saved to {output_dir}")
        
        return plot_paths
    
    def save_results(self, output_path: str):
        """Save evaluation results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        results_to_save = {
            'metrics': self.metrics,
            'evaluation_config': {
                'threshold': self.threshold,
                'device': str(self.device)
            },
            'summary': self._generate_summary()
        }
        
        save_json(results_to_save, output_path)
        logger.info(f"Evaluation results saved to {output_path}")
    
    def _generate_summary(self) -> Dict:
        """Generate a summary of evaluation results."""
        summary = {}
        
        for dataset_name, metrics in self.metrics.items():
            summary[dataset_name] = {
                'sample_count': len(self.results[dataset_name]['targets']),
                'positive_rate': metrics['positive_rate'],
                'auroc': metrics['auroc'],
                'auprc': metrics['auprc'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'specificity': metrics['specificity'],
                'balanced_accuracy': metrics['balanced_accuracy'],
                'inference_time_mean': metrics.get('inference_time_mean', 0),
                'throughput_samples_per_second': metrics.get('throughput_samples_per_second', 0)
            }
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of evaluation results."""
        print("\n" + "="*60)
        print("SEPSIS SENTINEL MODEL EVALUATION SUMMARY")
        print("="*60)
        
        for dataset_name, metrics in self.metrics.items():
            print(f"\n{dataset_name.upper()} DATASET:")
            print("-" * 30)
            print(f"Sample Count:       {len(self.results[dataset_name]['targets']):,}")
            print(f"Positive Rate:      {metrics['positive_rate']:.3f}")
            print(f"AUROC:             {metrics['auroc']:.4f}")
            print(f"AUPRC:             {metrics['auprc']:.4f}")
            print(f"Accuracy:          {metrics['accuracy']:.4f}")
            print(f"Precision:         {metrics['precision']:.4f}")
            print(f"Recall:            {metrics['recall']:.4f}")
            print(f"F1-Score:          {metrics['f1']:.4f}")
            print(f"Specificity:       {metrics['specificity']:.4f}")
            print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            
            if 'inference_time_mean' in metrics:
                print(f"Inference Time:    {metrics['inference_time_mean']:.4f}s")
                print(f"Throughput:        {metrics['throughput_samples_per_second']:.1f} samples/s")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    # This would be used for standalone evaluation
    print("Model Evaluator - Use as a module or import into evaluation scripts")