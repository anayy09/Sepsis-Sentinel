"""
SHAP (SHapley Additive exPlanations) Runner for Sepsis Sentinel
Provides model interpretability using SHAP values for multimodal sepsis prediction.
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
import shap
import torch
import torch.nn as nn
from torch import Tensor

from ..models.fusion_head import SepsisClassificationHead
from ..training.lightning_module import SepsisSentinelLightning

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure matplotlib for headless environments
import matplotlib
matplotlib.use('Agg')


class SepsisModelWrapper:
    """Wrapper for Sepsis Sentinel model to work with SHAP explainers."""
    
    def __init__(self, model: SepsisSentinelLightning, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def __call__(self, tft_features: np.ndarray, gnn_features: np.ndarray) -> np.ndarray:
        """
        Model prediction function for SHAP.
        
        Args:
            tft_features: TFT features as numpy array [batch_size, tft_dim]
            gnn_features: GNN features as numpy array [batch_size, gnn_dim]
        
        Returns:
            Prediction probabilities as numpy array [batch_size]
        """
        # Convert to tensors
        tft_tensor = torch.tensor(tft_features, dtype=torch.float32).to(self.device)
        gnn_tensor = torch.tensor(gnn_features, dtype=torch.float32).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            # Use the fusion head directly for simplified explanation
            outputs = self.model.classification_head.fusion_head(tft_tensor, gnn_tensor)
            probabilities = outputs['probabilities']
        
        return probabilities.cpu().numpy()


class CombinedModelWrapper:
    """Wrapper that combines TFT and GNN features for unified SHAP explanation."""
    
    def __init__(self, model: SepsisSentinelLightning, device: torch.device, 
                 tft_dim: int, gnn_dim: int):
        self.model = model
        self.device = device
        self.tft_dim = tft_dim
        self.gnn_dim = gnn_dim
        self.model.eval()
    
    def __call__(self, combined_features: np.ndarray) -> np.ndarray:
        """
        Model prediction function for combined features.
        
        Args:
            combined_features: Combined features [batch_size, tft_dim + gnn_dim]
        
        Returns:
            Prediction probabilities [batch_size]
        """
        # Split combined features
        tft_features = combined_features[:, :self.tft_dim]
        gnn_features = combined_features[:, self.tft_dim:]
        
        # Convert to tensors
        tft_tensor = torch.tensor(tft_features, dtype=torch.float32).to(self.device)
        gnn_tensor = torch.tensor(gnn_features, dtype=torch.float32).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model.classification_head.fusion_head(tft_tensor, gnn_tensor)
            probabilities = outputs['probabilities']
        
        return probabilities.cpu().numpy()


class SHAPRunner:
    """SHAP explainability runner for Sepsis Sentinel model."""
    
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
        
        # Model wrappers
        self.tft_dim = model.hparams.tft_hidden_size
        self.gnn_dim = model.hparams.gnn_hidden_channels
        
        self.model_wrapper = SepsisModelWrapper(model, device)
        self.combined_wrapper = CombinedModelWrapper(
            model, device, self.tft_dim, self.gnn_dim
        )
        
        # SHAP explainers
        self.explainers = {}
        
    def _get_default_feature_names(self) -> Dict[str, List[str]]:
        """Generate default feature names."""
        return {
            'tft': [f'tft_feature_{i}' for i in range(256)],
            'gnn': [f'gnn_feature_{i}' for i in range(64)],
            'combined': [f'tft_feature_{i}' for i in range(256)] + 
                      [f'gnn_feature_{i}' for i in range(64)]
        }
    
    def create_explainers(self,
                         background_data: Dict[str, np.ndarray],
                         explainer_type: str = "tree") -> None:
        """
        Create SHAP explainers for different model components.
        
        Args:
            background_data: Background dataset for SHAP
            explainer_type: Type of SHAP explainer ('tree', 'linear', 'deep', 'kernel')
        """
        logger.info(f"Creating SHAP explainers with type: {explainer_type}")
        
        # Combined features for unified explanation
        combined_background = np.concatenate([
            background_data['tft_features'],
            background_data['gnn_features']
        ], axis=1)
        
        if explainer_type == "tree":
            # Tree explainer (for tree-based models or approximation)
            self.explainers['combined'] = shap.TreeExplainer(
                self.combined_wrapper,
                combined_background
            )
        elif explainer_type == "linear":
            # Linear explainer (for linear models)
            self.explainers['combined'] = shap.LinearExplainer(
                self.combined_wrapper,
                combined_background
            )
        elif explainer_type == "deep":
            # Deep explainer (for neural networks)
            self.explainers['combined'] = shap.DeepExplainer(
                self.combined_wrapper,
                combined_background
            )
        elif explainer_type == "kernel":
            # Kernel explainer (model-agnostic)
            self.explainers['combined'] = shap.KernelExplainer(
                self.combined_wrapper,
                combined_background
            )
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
        
        # Separate explainers for TFT and GNN components
        if explainer_type == "kernel":
            # Only kernel explainer supports multi-input functions
            def tft_only_wrapper(tft_features):
                # Use mean GNN features as background
                gnn_background = np.mean(background_data['gnn_features'], axis=0, keepdims=True)
                gnn_expanded = np.repeat(gnn_background, tft_features.shape[0], axis=0)
                return self.model_wrapper(tft_features, gnn_expanded)
            
            def gnn_only_wrapper(gnn_features):
                # Use mean TFT features as background
                tft_background = np.mean(background_data['tft_features'], axis=0, keepdims=True)
                tft_expanded = np.repeat(tft_background, gnn_features.shape[0], axis=0)
                return self.model_wrapper(tft_expanded, gnn_features)
            
            self.explainers['tft'] = shap.KernelExplainer(
                tft_only_wrapper, background_data['tft_features']
            )
            self.explainers['gnn'] = shap.KernelExplainer(
                gnn_only_wrapper, background_data['gnn_features']
            )
        
        logger.info("SHAP explainers created successfully")
    
    def explain_instances(self,
                         instances: Dict[str, np.ndarray],
                         max_instances: int = 100,
                         save_individual: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate SHAP explanations for given instances.
        
        Args:
            instances: Dictionary containing TFT and GNN features
            max_instances: Maximum number of instances to explain
            save_individual: Whether to save individual explanations
        
        Returns:
            Dictionary containing SHAP values
        """
        logger.info(f"Generating SHAP explanations for {min(len(instances['tft_features']), max_instances)} instances")
        
        # Limit number of instances
        n_instances = min(len(instances['tft_features']), max_instances)
        tft_subset = instances['tft_features'][:n_instances]
        gnn_subset = instances['gnn_features'][:n_instances]
        
        # Combined features
        combined_subset = np.concatenate([tft_subset, gnn_subset], axis=1)
        
        # Generate SHAP values
        shap_values = {}
        
        # Combined explanation
        logger.info("Computing combined SHAP values...")
        shap_values['combined'] = self.explainers['combined'].shap_values(combined_subset)
        
        # Component-specific explanations (if available)
        if 'tft' in self.explainers:
            logger.info("Computing TFT SHAP values...")
            shap_values['tft'] = self.explainers['tft'].shap_values(tft_subset)
        
        if 'gnn' in self.explainers:
            logger.info("Computing GNN SHAP values...")
            shap_values['gnn'] = self.explainers['gnn'].shap_values(gnn_subset)
        
        # Save SHAP values
        if save_individual:
            self._save_shap_values(shap_values, instances, n_instances)
        
        return shap_values
    
    def _save_shap_values(self,
                         shap_values: Dict[str, np.ndarray],
                         instances: Dict[str, np.ndarray],
                         n_instances: int) -> None:
        """Save SHAP values to disk."""
        logger.info("Saving SHAP values...")
        
        shap_dir = self.output_dir / "shap_values"
        shap_dir.mkdir(exist_ok=True)
        
        # Save raw SHAP values
        for component, values in shap_values.items():
            np.save(shap_dir / f"{component}_shap_values.npy", values)
        
        # Save instance data
        for component, data in instances.items():
            if component.endswith('_features'):
                np.save(shap_dir / f"{component[:n_instances]}.npy", data[:n_instances])
        
        # Save feature names
        with open(shap_dir / "feature_names.pkl", "wb") as f:
            pickle.dump(self.feature_names, f)
        
        logger.info(f"SHAP values saved to {shap_dir}")
    
    def create_summary_plots(self,
                           shap_values: Dict[str, np.ndarray],
                           instances: Dict[str, np.ndarray],
                           plot_types: List[str] = ['summary', 'bar', 'waterfall']) -> None:
        """
        Create SHAP summary plots.
        
        Args:
            shap_values: SHAP values dictionary
            instances: Instance data dictionary
            plot_types: Types of plots to create
        """
        logger.info("Creating SHAP summary plots...")
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Combined features and SHAP values
        if 'combined' in shap_values:
            combined_features = np.concatenate([
                instances['tft_features'],
                instances['gnn_features']
            ], axis=1)
            
            combined_shap = shap_values['combined']
            feature_names = self.feature_names['combined']
            
            # Summary plot
            if 'summary' in plot_types:
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    combined_shap, combined_features,
                    feature_names=feature_names,
                    show=False, max_display=20
                )
                plt.tight_layout()
                plt.savefig(plots_dir / "combined_summary_plot.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Bar plot (feature importance)
            if 'bar' in plot_types:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    combined_shap, combined_features,
                    feature_names=feature_names,
                    plot_type="bar", show=False, max_display=20
                )
                plt.tight_layout()
                plt.savefig(plots_dir / "combined_importance_bar.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Waterfall plots for individual instances
            if 'waterfall' in plot_types:
                n_waterfalls = min(5, len(combined_shap))
                for i in range(n_waterfalls):
                    plt.figure(figsize=(12, 8))
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=combined_shap[i],
                            base_values=self.explainers['combined'].expected_value,
                            data=combined_features[i],
                            feature_names=feature_names
                        ),
                        show=False, max_display=15
                    )
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"waterfall_instance_{i}.png", dpi=300, bbox_inches='tight')
                    plt.close()
        
        # Component-specific plots
        for component in ['tft', 'gnn']:
            if component in shap_values:
                features = instances[f'{component}_features']
                shap_vals = shap_values[component]
                feature_names = self.feature_names[component]
                
                # Summary plot
                if 'summary' in plot_types:
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(
                        shap_vals, features,
                        feature_names=feature_names,
                        show=False, max_display=15
                    )
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"{component}_summary_plot.png", dpi=300, bbox_inches='tight')
                    plt.close()
        
        logger.info(f"SHAP plots saved to {plots_dir}")
    
    def create_feature_importance_analysis(self,
                                         shap_values: Dict[str, np.ndarray],
                                         instances: Dict[str, np.ndarray],
                                         save_results: bool = True) -> Dict:
        """
        Create comprehensive feature importance analysis.
        
        Args:
            shap_values: SHAP values dictionary
            instances: Instance data dictionary
            save_results: Whether to save results to disk
        
        Returns:
            Dictionary containing importance analysis results
        """
        logger.info("Creating feature importance analysis...")
        
        analysis_results = {}
        
        # Combined feature importance
        if 'combined' in shap_values:
            combined_shap = shap_values['combined']
            feature_names = self.feature_names['combined']
            
            # Global feature importance (mean absolute SHAP values)
            global_importance = np.mean(np.abs(combined_shap), axis=0)
            
            # Sort features by importance
            importance_order = np.argsort(global_importance)[::-1]
            
            analysis_results['combined'] = {
                'feature_names': [feature_names[i] for i in importance_order],
                'importance_scores': global_importance[importance_order],
                'raw_shap_values': combined_shap,
                'top_10_features': [feature_names[i] for i in importance_order[:10]],
                'top_10_scores': global_importance[importance_order[:10]]
            }
        
        # Component-specific importance
        for component in ['tft', 'gnn']:
            if component in shap_values:
                shap_vals = shap_values[component]
                feature_names = self.feature_names[component]
                
                global_importance = np.mean(np.abs(shap_vals), axis=0)
                importance_order = np.argsort(global_importance)[::-1]
                
                analysis_results[component] = {
                    'feature_names': [feature_names[i] for i in importance_order],
                    'importance_scores': global_importance[importance_order],
                    'raw_shap_values': shap_vals,
                    'top_10_features': [feature_names[i] for i in importance_order[:10]],
                    'top_10_scores': global_importance[importance_order[:10]]
                }
        
        # Save results
        if save_results:
            analysis_dir = self.output_dir / "analysis"
            analysis_dir.mkdir(exist_ok=True)
            
            # Save feature importance rankings
            for component, results in analysis_results.items():
                # Create DataFrame for easy viewing
                importance_df = pd.DataFrame({
                    'feature_name': results['feature_names'],
                    'importance_score': results['importance_scores']
                })
                
                importance_df.to_csv(
                    analysis_dir / f"{component}_feature_importance.csv",
                    index=False
                )
                
                # Create importance plot
                plt.figure(figsize=(12, 8))
                top_features = importance_df.head(20)
                sns.barplot(
                    data=top_features, 
                    x='importance_score', 
                    y='feature_name',
                    palette='viridis'
                )
                plt.title(f'Top 20 {component.upper()} Feature Importance (SHAP)')
                plt.xlabel('Mean |SHAP Value|')
                plt.tight_layout()
                plt.savefig(
                    analysis_dir / f"{component}_importance_ranking.png",
                    dpi=300, bbox_inches='tight'
                )
                plt.close()
        
        logger.info("Feature importance analysis completed")
        return analysis_results
    
    def run_complete_analysis(self,
                            background_data: Dict[str, np.ndarray],
                            test_instances: Dict[str, np.ndarray],
                            max_instances: int = 100,
                            explainer_type: str = "kernel") -> Dict:
        """
        Run complete SHAP analysis pipeline.
        
        Args:
            background_data: Background dataset for SHAP
            test_instances: Test instances to explain
            max_instances: Maximum instances to analyze
            explainer_type: Type of SHAP explainer
        
        Returns:
            Complete analysis results
        """
        logger.info("Starting complete SHAP analysis...")
        
        # Create explainers
        self.create_explainers(background_data, explainer_type)
        
        # Generate explanations
        shap_values = self.explain_instances(
            test_instances, max_instances, save_individual=True
        )
        
        # Create visualizations
        self.create_summary_plots(shap_values, test_instances)
        
        # Perform importance analysis
        importance_results = self.create_feature_importance_analysis(
            shap_values, test_instances
        )
        
        # Combine results
        complete_results = {
            'shap_values': shap_values,
            'importance_analysis': importance_results,
            'metadata': {
                'n_instances_analyzed': min(len(test_instances['tft_features']), max_instances),
                'explainer_type': explainer_type,
                'feature_counts': {
                    'tft': len(self.feature_names['tft']),
                    'gnn': len(self.feature_names['gnn']),
                    'combined': len(self.feature_names['combined'])
                }
            }
        }
        
        # Save complete results
        results_path = self.output_dir / "complete_shap_analysis.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(complete_results, f)
        
        logger.info(f"Complete SHAP analysis saved to {results_path}")
        return complete_results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SHAP analysis for Sepsis Sentinel")
    parser.add_argument("--model-path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data-path", required=True, help="Path to test data")
    parser.add_argument("--background-path", required=True, help="Path to background data")
    parser.add_argument("--output-dir", default="./reports/explain", help="Output directory")
    parser.add_argument("--max-instances", type=int, default=100, help="Max instances to analyze")
    parser.add_argument("--explainer-type", default="kernel", choices=["kernel", "tree", "linear", "deep"])
    
    args = parser.parse_args()
    
    # This would need to be implemented with proper data loading
    logger.info("SHAP analysis utility ready. Implement data loading for actual analysis.")
