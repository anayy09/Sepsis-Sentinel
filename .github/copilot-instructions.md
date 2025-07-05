# Copilot Instructions for Sepsis Sentinel Project

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Overview
This is the Sepsis Sentinel project - an explainable multimodal early sepsis prediction system using MIMIC-IV waveform and EHR data.

## Key Components
- **Data Pipeline**: PySpark ETL for MIMIC-IV processing with Delta Lake storage
- **Models**: Temporal Fusion Transformer (TFT) + Heterogeneous Graph Neural Network (GNN)
- **Training**: PyTorch Lightning with Optuna hyperparameter optimization
- **Explainability**: SHAP and Integrated Gradients for model interpretability
- **Deployment**: Triton Inference Server + FastAPI + Flask dashboard

## Code Style Guidelines
- Follow PEP-8 with black formatter
- Use type hints (mypy strict mode)
- NumPy-style docstrings
- PyTorch Lightning for training workflows
- Minimal external dependencies as specified in requirements.txt

## Architecture Patterns
- Use dependency injection for configuration
- Implement proper error handling and logging
- Follow single responsibility principle
- Use factory patterns for model creation
- Implement proper data validation with Pydantic

## Testing Requirements
- Maintain 90% test coverage
- Write unit tests for all public functions
- Include integration tests for end-to-end workflows
- Use pytest fixtures for test data

## Performance Considerations
- Target single-GPU workstation deployment
- Use mixed precision training (bf16)
- Implement efficient data loading with TorchArrow
- Optimize for 6-hour prediction window with 30-min sliding windows

## Medical Domain Context
- Focus on Sepsis-3 criteria for labeling
- Implement proper bias auditing across demographics
- Ensure clinical interpretability of features
- Follow healthcare data privacy best practices
