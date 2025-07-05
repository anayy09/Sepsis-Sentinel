"""
Command Line Interface for Sepsis Sentinel
Provides CLI commands for training, inference, and deployment.
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import main as train_main
from models.export_onnx import main as export_main

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_train_parser(subparsers):
    """Create parser for training command."""
    train_parser = subparsers.add_parser(
        'train',
        help='Train Sepsis Sentinel model',
        description='Train the multimodal sepsis prediction model'
    )
    
    train_parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training configuration file'
    )
    
    train_parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to processed training data'
    )
    
    train_parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Output directory for models and logs'
    )
    
    train_parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume training from'
    )
    
    train_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with reduced data and epochs'
    )
    
    train_parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only run testing with pre-trained model'
    )
    
    return train_parser


def create_export_parser(subparsers):
    """Create parser for model export command."""
    export_parser = subparsers.add_parser(
        'export',
        help='Export trained model to ONNX format',
        description='Export Sepsis Sentinel model for deployment'
    )
    
    export_parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained PyTorch model checkpoint'
    )
    
    export_parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Output path for ONNX model'
    )
    
    export_parser.add_argument(
        '--model-type',
        choices=['complete', 'simplified'],
        default='simplified',
        help='Type of model to export'
    )
    
    export_parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate exported model against PyTorch version'
    )
    
    export_parser.add_argument(
        '--optimize',
        action='store_true',
        help='Apply optimization to exported model'
    )
    
    return export_parser


def create_serve_parser(subparsers):
    """Create parser for serving command."""
    serve_parser = subparsers.add_parser(
        'serve',
        help='Start prediction server',
        description='Start FastAPI server for model inference'
    )
    
    serve_parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to ONNX model file'
    )
    
    serve_parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host to bind server to'
    )
    
    serve_parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind server to'
    )
    
    serve_parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes'
    )
    
    return serve_parser


def create_predict_parser(subparsers):
    """Create parser for single prediction command."""
    predict_parser = subparsers.add_parser(
        'predict',
        help='Make single prediction',
        description='Make sepsis prediction for a single patient'
    )
    
    predict_parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to ONNX model file'
    )
    
    predict_parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Path to patient data JSON file'
    )
    
    predict_parser.add_argument(
        '--output-file',
        type=str,
        help='Path to save prediction results'
    )
    
    predict_parser.add_argument(
        '--explain',
        action='store_true',
        help='Generate explanations with prediction'
    )
    
    return predict_parser


def create_etl_parser(subparsers):
    """Create parser for ETL command."""
    etl_parser = subparsers.add_parser(
        'etl',
        help='Run data ETL pipeline',
        description='Process MIMIC-IV data for training'
    )
    
    etl_parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to ETL configuration file'
    )
    
    etl_parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help='Path to raw MIMIC-IV data'
    )
    
    etl_parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Output path for processed data'
    )
    
    etl_parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of Spark workers'
    )
    
    return etl_parser


def handle_train_command(args):
    """Handle training command."""
    logger.info("Starting training...")
    
    # Set up arguments for training main function
    import sys
    original_argv = sys.argv.copy()
    
    sys.argv = [
        'train.py',
        '--config', args.config,
        '--data-path', args.data_path,
        '--output-dir', args.output_dir
    ]
    
    if args.resume:
        sys.argv.extend(['--resume', args.resume])
    
    if args.debug:
        sys.argv.append('--debug')
    
    if args.test_only:
        sys.argv.append('--test-only')
    
    try:
        train_main()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
    finally:
        sys.argv = original_argv


def handle_export_command(args):
    """Handle model export command."""
    logger.info("Starting model export...")
    
    # Set up arguments for export main function
    import sys
    original_argv = sys.argv.copy()
    
    sys.argv = [
        'export_onnx.py',
        '--model-path', args.model_path,
        '--output-path', args.output_path,
        '--model-type', args.model_type
    ]
    
    if args.validate:
        sys.argv.append('--validate')
    
    if args.optimize:
        sys.argv.append('--optimize')
    
    try:
        export_main()
        logger.info("Model export completed successfully!")
    except Exception as e:
        logger.error(f"Model export failed: {e}")
        sys.exit(1)
    finally:
        sys.argv = original_argv


def handle_serve_command(args):
    """Handle serve command."""
    logger.info(f"Starting server on {args.host}:{args.port}...")
    
    try:
        import uvicorn
        from deploy.api.main import app
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers
        )
    except ImportError:
        logger.error("uvicorn not installed. Please install with: pip install uvicorn")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


def handle_predict_command(args):
    """Handle single prediction command."""
    logger.info("Making prediction...")
    
    try:
        # Load patient data
        with open(args.input_file, 'r') as f:
            patient_data = json.load(f)
        
        # Mock prediction for now (would use actual ONNX model)
        prediction_result = {
            "patient_id": patient_data.get("patient_id", "unknown"),
            "sepsis_probability": 0.25,  # Mock value
            "sepsis_risk_level": "medium",
            "confidence_score": 0.82,
            "processing_time_ms": 150.0
        }
        
        if args.explain:
            prediction_result["explanations"] = {
                "top_features": [
                    {"feature": "lactate", "importance": 0.15},
                    {"feature": "heart_rate", "importance": 0.12},
                    {"feature": "white_blood_cells", "importance": 0.10}
                ]
            }
        
        # Output results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(prediction_result, f, indent=2)
            logger.info(f"Prediction saved to {args.output_file}")
        else:
            print(json.dumps(prediction_result, indent=2))
        
        logger.info("Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


def handle_etl_command(args):
    """Handle ETL command."""
    logger.info("Starting ETL pipeline...")
    
    try:
        from data_pipeline.spark_etl import MIMICETLPipeline
        
        # Initialize pipeline
        pipeline = MIMICETLPipeline(args.config)
        
        # Run ETL
        pipeline.run_pipeline(
            waveform_path=f"{args.input_path}/waveforms",
            ehr_path=f"{args.input_path}/ehr",
            output_path=args.output_path
        )
        
        pipeline.stop()
        logger.info("ETL pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        sys.exit(1)


def create_parser():
    """Create main argument parser."""
    parser = argparse.ArgumentParser(
        prog='sepsis-sentinel',
        description='Sepsis Sentinel: Early sepsis prediction using multimodal AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model
  sepsis-sentinel train --config configs/train_tft_gnn.yaml --data-path data/processed

  # Export model
  sepsis-sentinel export --model-path models/best.ckpt --output-path models/sepsis.onnx

  # Start server
  sepsis-sentinel serve --model-path models/sepsis.onnx --port 8000

  # Make prediction
  sepsis-sentinel predict --model-path models/sepsis.onnx --input-file patient.json

  # Run ETL
  sepsis-sentinel etl --config configs/schema.yaml --input-path data/raw --output-path data/processed
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Sepsis Sentinel 1.0.0'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='<command>'
    )
    
    # Add command parsers
    create_train_parser(subparsers)
    create_export_parser(subparsers)
    create_serve_parser(subparsers)
    create_predict_parser(subparsers)
    create_etl_parser(subparsers)
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle commands
    if args.command == 'train':
        handle_train_command(args)
    elif args.command == 'export':
        handle_export_command(args)
    elif args.command == 'serve':
        handle_serve_command(args)
    elif args.command == 'predict':
        handle_predict_command(args)
    elif args.command == 'etl':
        handle_etl_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()