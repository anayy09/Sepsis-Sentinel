"""
Integrated Gradients Runner for Sepsis Sentinel
Provides attribution-based explanations using Integrated Gradients from Captum.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch import Tensor
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    DeepLift,
    DeepLiftShap,
    LayerConductance,
    LayerIntegratedGradients,
    LayerGradientXActivation,
    Saliency,
    InputXGradient,
    GuidedBackprop,
    GuidedGradCam
)
from captum.attr._utils.visualization import visualize_image_attr

from ..training.lightning_module import SepsisSentinelLightning

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure matplotlib for headless environments
import matplotlib
matplotlib.use('Agg')


class SepsisModelForAttribution(nn.Module):
    """Wrapper model for attribution methods that require specific input/output format."""
    
    def __init__(self, model: SepsisSentinelLightning, device: torch.device):
        super().__init__()
        self.model = model
        self.device = device
        self.model.eval()
        
        # Disable dropout and batch norm for consistent attributions
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()
    
    def forward(self, combined_input: Tensor) -> Tensor:
        """
        Forward pass for combined TFT and GNN features.
        
        Args:
            combined_input: Combined features [batch_size, tft_dim + gnn_dim]
        
        Returns:
            Prediction logits [batch_size, 1]
        """
        # Split combined input
        tft_dim = self.model.hparams.tft_hidden_size
        tft_features = combined_input[:, :tft_dim]
        gnn_features = combined_input[:, tft_dim:]
        
        # Get predictions from fusion head
        outputs = self.model.classification_head.fusion_head(tft_features, gnn_features)
        return outputs['logits'].unsqueeze(-1)  # Add dimension for attribution


class IntegratedGradientsRunner:
    """Integrated Gradients explainability runner for Sepsis Sentinel model."""
    
    def __init__(self,
                 model: SepsisSentinelLightning,
                 device: torch.device,
                 feature_names: Optional[Dict[str, List[str]]] = None,
                 output_dir: str = "./reports/explain"):
        self.model = model
        self.device = device
        self.feature_names = feature_names or self._get_default_feature_names()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model dimensions
        self.tft_dim = model.hparams.tft_hidden_size
        self.gnn_dim = model.hparams.gnn_hidden_channels
        self.total_dim = self.tft_dim + self.gnn_dim
        
        # Attribution model
        self.attribution_model = SepsisModelForAttribution(model, device)
        
        # Attribution methods
        self.attribution_methods = self._initialize_attribution_methods()
        
    def _get_default_feature_names(self) -> Dict[str, List[str]]:
        """Generate default feature names."""
        return {
            'tft': [f'tft_feature_{i}' for i in range(256)],
            'gnn': [f'gnn_feature_{i}' for i in range(64)],
            'combined': [f'tft_feature_{i}' for i in range(256)] + 
                      [f'gnn_feature_{i}' for i in range(64)]
        }
    
    def _initialize_attribution_methods(self) -> Dict:
        """Initialize various attribution methods."""
        methods = {
            'integrated_gradients': IntegratedGradients(self.attribution_model),
            'gradient_shap': GradientShap(self.attribution_model),
            'deep_lift': DeepLift(self.attribution_model),
            'deep_lift_shap': DeepLiftShap(self.attribution_model),
            'saliency': Saliency(self.attribution_model),
            'input_x_gradient': InputXGradient(self.attribution_model),
            'guided_backprop': GuidedBackprop(self.attribution_model)
        }
        
        # Layer-specific methods (if needed)
        fusion_layers = [
            layer for name, layer in self.attribution_model.model.classification_head.fusion_head.named_modules()
            if isinstance(layer, nn.Linear)
        ]
        
        if fusion_layers:
            # Use the first linear layer for layer-specific attributions
            target_layer = fusion_layers[0]
            methods.update({
                'layer_conductance': LayerConductance(self.attribution_model, target_layer),
                'layer_integrated_gradients': LayerIntegratedGradients(
                    self.attribution_model, target_layer
                ),
                'layer_gradient_x_activation': LayerGradientXActivation(
                    self.attribution_model, target_layer
                )
            })
        
        return methods
    
    def create_baselines(self,
                        instances: Dict[str, np.ndarray],
                        baseline_type: str = "zeros") -> Tensor:
        """
        Create baseline inputs for attribution methods.
        
        Args:
            instances: Dictionary containing TFT and GNN features
            baseline_type: Type of baseline ('zeros', 'mean', 'gaussian_noise')
        
        Returns:
            Baseline tensor
        """
        batch_size = len(instances['tft_features'])
        
        if baseline_type == "zeros":
            baseline = torch.zeros(batch_size, self.total_dim).to(self.device)
            
        elif baseline_type == "mean":
            # Use mean of the instances as baseline
            tft_mean = np.mean(instances['tft_features'], axis=0)
            gnn_mean = np.mean(instances['gnn_features'], axis=0)
            combined_mean = np.concatenate([tft_mean, gnn_mean])
            baseline = torch.tensor(combined_mean).unsqueeze(0).repeat(batch_size, 1).to(self.device)
            
        elif baseline_type == "gaussian_noise":
            # Random Gaussian noise baseline
            baseline = torch.randn(batch_size, self.total_dim).to(self.device) * 0.1
            
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")
        
        return baseline.float()
    
    def compute_attributions(self,
                           instances: Dict[str, np.ndarray],
                           method_name: str = "integrated_gradients",
                           baseline_type: str = "zeros",
                           n_steps: int = 50,
                           target_class: int = 0,
                           **kwargs) -> np.ndarray:
        """
        Compute attributions for given instances using specified method.
        
        Args:
            instances: Dictionary containing TFT and GNN features
            method_name: Name of attribution method to use
            baseline_type: Type of baseline to use
            n_steps: Number of steps for integrated gradients
            target_class: Target class for attribution
            **kwargs: Additional arguments for attribution method
        
        Returns:
            Attribution values as numpy array
        """
        logger.info(f"Computing {method_name} attributions...")
        
        # Prepare inputs
        tft_features = torch.tensor(instances['tft_features'], dtype=torch.float32).to(self.device)
        gnn_features = torch.tensor(instances['gnn_features'], dtype=torch.float32).to(self.device)
        combined_input = torch.cat([tft_features, gnn_features], dim=1)
        
        # Create baseline
        baseline = self.create_baselines(instances, baseline_type)
        
        # Get attribution method
        if method_name not in self.attribution_methods:
            raise ValueError(f"Unknown attribution method: {method_name}")
        
        attribution_method = self.attribution_methods[method_name]
        
        # Compute attributions
        with torch.no_grad():
            # First get predictions to verify model works
            predictions = self.attribution_model(combined_input)
            logger.info(f"Model predictions shape: {predictions.shape}")
        
        # Compute attributions based on method type
        if method_name in ['integrated_gradients', 'gradient_shap', 'deep_lift', 'deep_lift_shap']:
            if method_name == 'integrated_gradients':
                attributions = attribution_method.attribute(
                    combined_input,
                    baselines=baseline,
                    target=target_class,
                    n_steps=n_steps,
                    return_convergence_delta=False,
                    **kwargs
                )
            elif method_name == 'gradient_shap':
                # GradientShap needs noise baseline
                noise_baseline = torch.randn_like(combined_input) * 0.1
                attributions = attribution_method.attribute(
                    combined_input,
                    baselines=noise_baseline,
                    target=target_class,
                    n_samples=n_steps,
                    **kwargs
                )
            else:
                attributions = attribution_method.attribute(
                    combined_input,
                    baselines=baseline,
                    target=target_class,
                    **kwargs
                )
        
        elif method_name in ['saliency', 'input_x_gradient', 'guided_backprop']:
            attributions = attribution_method.attribute(
                combined_input,
                target=target_class,
                **kwargs
            )
        
        elif method_name.startswith('layer_'):
            # Layer-specific methods
            if method_name == 'layer_integrated_gradients':
                attributions = attribution_method.attribute(
                    combined_input,
                    baselines=baseline,
                    target=target_class,
                    n_steps=n_steps,
                    **kwargs
                )
            else:
                attributions = attribution_method.attribute(
                    combined_input,
                    target=target_class,
                    **kwargs
                )
        
        else:
            raise ValueError(f"Attribution computation not implemented for: {method_name}")
        
        # Convert to numpy
        attributions_np = attributions.detach().cpu().numpy()
        
        logger.info(f"Attributions computed. Shape: {attributions_np.shape}")
        return attributions_np
    
    def run_multiple_methods(self,
                           instances: Dict[str, np.ndarray],
                           methods: List[str] = None,
                           baseline_type: str = "zeros",
                           n_steps: int = 50) -> Dict[str, np.ndarray]:
        """
        Run multiple attribution methods on the same instances.
        
        Args:
            instances: Dictionary containing TFT and GNN features
            methods: List of method names to run
            baseline_type: Type of baseline to use
            n_steps: Number of steps for gradient-based methods
        
        Returns:
            Dictionary mapping method names to attribution arrays
        """
        if methods is None:
            methods = ['integrated_gradients', 'gradient_shap', 'saliency', 'input_x_gradient']
        
        attributions_dict = {}
        
        for method_name in methods:
            try:
                attributions = self.compute_attributions(
                    instances=instances,
                    method_name=method_name,
                    baseline_type=baseline_type,
                    n_steps=n_steps
                )
                attributions_dict[method_name] = attributions
                logger.info(f"Successfully computed {method_name} attributions")
                
            except Exception as e:
                logger.error(f"Failed to compute {method_name} attributions: {str(e)}")
                continue
        
        return attributions_dict
    
    def create_attribution_plots(self,
                               attributions_dict: Dict[str, np.ndarray],
                               instances: Dict[str, np.ndarray],
                               instance_idx: int = 0,
                               top_k: int = 20) -> None:
        """
        Create visualization plots for attributions.
        
        Args:
            attributions_dict: Dictionary of attribution arrays
            instances: Original instance data
            instance_idx: Index of instance to visualize
            top_k: Number of top features to show
        """
        logger.info(f"Creating attribution plots for instance {instance_idx}...")
        
        plots_dir = self.output_dir / "ig_plots"
        plots_dir.mkdir(exist_ok=True)
        
        feature_names = self.feature_names['combined']
        
        for method_name, attributions in attributions_dict.items():
            if len(attributions.shape) > 2:
                # Handle layer attributions (might have different shapes)
                attributions = attributions.mean(axis=tuple(range(2, len(attributions.shape))))
            
            # Get attributions for specific instance
            instance_attributions = attributions[instance_idx]
            
            # Get top-k features by absolute attribution value
            abs_attributions = np.abs(instance_attributions)
            top_indices = np.argsort(abs_attributions)[-top_k:][::-1]
            
            # Create bar plot
            plt.figure(figsize=(12, 8))
            colors = ['red' if attr < 0 else 'blue' for attr in instance_attributions[top_indices]]
            
            plt.barh(
                range(len(top_indices)),
                instance_attributions[top_indices],
                color=colors,
                alpha=0.7
            )
            
            plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
            plt.xlabel('Attribution Value')
            plt.title(f'{method_name.replace("_", " ").title()} - Instance {instance_idx}\nTop {top_k} Feature Attributions')
            plt.gca().invert_yaxis()
            
            # Add value labels on bars
            for i, (idx, value) in enumerate(zip(top_indices, instance_attributions[top_indices])):
                plt.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}', 
                        ha='left' if value >= 0 else 'right', va='center')
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{method_name}_instance_{instance_idx}_top{top_k}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create comparison plot
        if len(attributions_dict) > 1:
            self._create_method_comparison_plot(
                attributions_dict, instance_idx, top_k, plots_dir
            )
        
        logger.info(f"Attribution plots saved to {plots_dir}")
    
    def _create_method_comparison_plot(self,
                                     attributions_dict: Dict[str, np.ndarray],
                                     instance_idx: int,
                                     top_k: int,
                                     plots_dir: Path) -> None:
        """Create comparison plot across different attribution methods."""
        
        feature_names = self.feature_names['combined']
        
        # Collect attributions from all methods
        all_attributions = []
        method_names = []
        
        for method_name, attributions in attributions_dict.items():
            if len(attributions.shape) > 2:
                attributions = attributions.mean(axis=tuple(range(2, len(attributions.shape))))
            
            all_attributions.append(attributions[instance_idx])
            method_names.append(method_name)
        
        # Find features that appear in top-k for any method
        all_top_features = set()
        for attributions in all_attributions:
            abs_attr = np.abs(attributions)
            top_indices = np.argsort(abs_attr)[-top_k:]
            all_top_features.update(top_indices)
        
        # Create comparison DataFrame
        comparison_data = []
        for feature_idx in all_top_features:
            for method_idx, method_name in enumerate(method_names):
                comparison_data.append({
                    'feature': feature_names[feature_idx],
                    'method': method_name,
                    'attribution': all_attributions[method_idx][feature_idx]
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create grouped bar plot
        plt.figure(figsize=(15, 10))
        
        # Pivot for easier plotting
        pivot_df = comparison_df.pivot(index='feature', columns='method', values='attribution')
        
        # Sort by max absolute attribution across methods
        max_abs_attr = pivot_df.abs().max(axis=1)
        pivot_df = pivot_df.loc[max_abs_attr.nlargest(top_k).index]
        
        # Create plot
        ax = pivot_df.plot(kind='barh', figsize=(15, 10), width=0.8)
        plt.xlabel('Attribution Value')
        plt.title(f'Attribution Method Comparison - Instance {instance_idx}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"method_comparison_instance_{instance_idx}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_attribution_stability(self,
                                    instances: Dict[str, np.ndarray],
                                    method_name: str = "integrated_gradients",
                                    n_runs: int = 5,
                                    noise_level: float = 0.01) -> Dict:
        """
        Analyze stability of attributions by adding small amounts of noise.
        
        Args:
            instances: Dictionary containing TFT and GNN features
            method_name: Attribution method to analyze
            n_runs: Number of noise runs
            noise_level: Standard deviation of noise to add
        
        Returns:
            Dictionary containing stability analysis results
        """
        logger.info(f"Analyzing attribution stability for {method_name}...")
        
        original_attributions = self.compute_attributions(instances, method_name)
        
        noisy_attributions = []
        for run in range(n_runs):
            # Add noise to instances
            noise_tft = np.random.normal(0, noise_level, instances['tft_features'].shape)
            noise_gnn = np.random.normal(0, noise_level, instances['gnn_features'].shape)
            
            noisy_instances = {
                'tft_features': instances['tft_features'] + noise_tft,
                'gnn_features': instances['gnn_features'] + noise_gnn
            }
            
            attributions = self.compute_attributions(noisy_instances, method_name)
            noisy_attributions.append(attributions)
        
        # Calculate stability metrics
        noisy_attributions = np.array(noisy_attributions)
        
        # Standard deviation across runs
        attribution_std = np.std(noisy_attributions, axis=0)
        
        # Correlation with original attributions
        correlations = []
        for run_attributions in noisy_attributions:
            corr_matrix = np.corrcoef(original_attributions.flatten(), run_attributions.flatten())
            correlations.append(corr_matrix[0, 1])
        
        stability_results = {
            'original_attributions': original_attributions,
            'noisy_attributions': noisy_attributions,
            'attribution_std': attribution_std,
            'correlations': correlations,
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'mean_std': np.mean(attribution_std),
            'stability_score': np.mean(correlations)  # Higher is more stable
        }
        
        logger.info(f"Stability analysis completed. Mean correlation: {stability_results['mean_correlation']:.4f}")
        return stability_results
    
    def run_complete_analysis(self,
                            instances: Dict[str, np.ndarray],
                            methods: List[str] = None,
                            baseline_type: str = "zeros",
                            n_steps: int = 50,
                            max_instances: int = 10) -> Dict:
        """
        Run complete Integrated Gradients analysis.
        
        Args:
            instances: Dictionary containing TFT and GNN features
            methods: List of attribution methods to run
            baseline_type: Type of baseline to use
            n_steps: Number of steps for gradient methods
            max_instances: Maximum instances to analyze
        
        Returns:
            Complete analysis results
        """
        logger.info("Starting complete Integrated Gradients analysis...")
        
        # Limit instances
        n_instances = min(len(instances['tft_features']), max_instances)
        limited_instances = {
            'tft_features': instances['tft_features'][:n_instances],
            'gnn_features': instances['gnn_features'][:n_instances]
        }
        
        # Run multiple attribution methods
        attributions_dict = self.run_multiple_methods(
            limited_instances, methods, baseline_type, n_steps
        )
        
        # Create visualizations for first few instances
        n_plots = min(3, n_instances)
        for i in range(n_plots):
            self.create_attribution_plots(attributions_dict, limited_instances, i)
        
        # Analyze stability
        stability_results = {}
        for method_name in attributions_dict.keys():
            if method_name in ['integrated_gradients', 'gradient_shap']:  # Only for main methods
                stability = self.analyze_attribution_stability(
                    limited_instances, method_name
                )
                stability_results[method_name] = stability
        
        # Save results
        results_dir = self.output_dir / "ig_results"
        results_dir.mkdir(exist_ok=True)
        
        complete_results = {
            'attributions': attributions_dict,
            'stability_analysis': stability_results,
            'metadata': {
                'n_instances_analyzed': n_instances,
                'methods_used': list(attributions_dict.keys()),
                'baseline_type': baseline_type,
                'n_steps': n_steps
            }
        }
        
        # Save to disk
        with open(results_dir / "complete_ig_analysis.pkl", "wb") as f:
            pickle.dump(complete_results, f)
        
        # Save attributions as CSV for easy access
        for method_name, attributions in attributions_dict.items():
            if len(attributions.shape) == 2:  # Only 2D arrays
                attr_df = pd.DataFrame(
                    attributions,
                    columns=self.feature_names['combined']
                )
                attr_df.to_csv(results_dir / f"{method_name}_attributions.csv", index=False)
        
        logger.info(f"Complete IG analysis saved to {results_dir}")
        return complete_results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run IG analysis for Sepsis Sentinel")
    parser.add_argument("--model-path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data-path", required=True, help="Path to test data")
    parser.add_argument("--output-dir", default="./reports/explain", help="Output directory")
    parser.add_argument("--max-instances", type=int, default=10, help="Max instances to analyze")
    parser.add_argument("--methods", nargs="+", default=["integrated_gradients", "saliency"],
                       help="Attribution methods to use")
    
    args = parser.parse_args()
    
    # This would need to be implemented with proper data loading
    logger.info("IG analysis utility ready. Implement data loading for actual analysis.")
