import json
import logging
from pathlib import Path
from typing import Dict, Any
from pdf_processor import EnhancedPDFProcessor
import argparse
import sys
import os
import Finetuning_Pipeline
# Import the VLLMManager class from the inference script
from VLLM_Inference import VLLMManager

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
# Add file handler for persistent logging
file_handler = logging.FileHandler('pdf_processing.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def read_config(config_input: str) -> Dict[str, Any]:
    """Read configuration from a JSON string or file."""
    try:
        # First, try to parse it as a JSON string
        cleaned_input = config_input.strip()
        if cleaned_input.startswith("'") and cleaned_input.endswith("'"):
            cleaned_input = cleaned_input[1:-1]
        config = json.loads(cleaned_input)
        return config
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse as JSON string: {e}")
        # Treat as file path
        try:
            if os.path.exists(config_input):
                with open(config_input, 'r') as f:
                    return json.load(f)
            else:
                logger.error(f"Config file not found: {config_input}")
                return {}
        except Exception as e:
            logger.error(f"Error reading configuration: {e}")
            return {}

def process_pdf_command_line(config: Dict[str, Any]) -> Dict[str, Any]:
    """Process PDF using EnhancedPDFProcessor based on the provided configuration."""
    try:
        if not config.get('pdf_file'):
            return {'status': 'error', 'message': 'Missing required parameter: pdf_file'}
        if not os.path.exists(config.get('pdf_file')):
            return {'status': 'error', 'message': f'PDF file not found: {config.get("pdf_file")}'}

        # Create processor instance
        processor = EnhancedPDFProcessor()

        config_json = json.dumps(config)

        if not processor.process_input(["pdf_processor", config_json]):
            return {'status': 'error', 'message': 'Failed to process configuration'}
        result = processor.process_pdf()

        if result['status'] == 'success':
            logger.info(f"QA pairs generated: {result['qa_pairs_count']}\nOutput saved to: {result['output_file']}")

        return result

    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

def process_fine_tuning(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run fine-tuning based on the provided configuration."""
    try:
        # Extract required parameters for fine-tuning
        required_fields = [
            'json_file', 'base_model', 'system_prompt', 'repo_id',
            'max_step', 'learning_rate', 'epochs'
        ]

        # Prepare the fine-tuning configuration
        finetune_config = {}
        for field in required_fields:
            if field in config:
                finetune_config[field] = config[field]
            else:
                return {'status': 'error', 'message': f"Missing required field: {field}"}

        # Additional optional fields
        optional_fields = [
            'push_to_hf', 'hf_token', 'quantize',
            'per_device_train_batch_size', 'gradient_accumulation_steps',
            'warmup_steps', 'logging_steps', 'optim', 'weight_decay',
            'lr_scheduler_type', 'max_grad_norm', 'dataloader_num_workers',
            'gradient_checkpointing', 'adam_beta1', 'adam_beta2',
            'adam_epsilon', 'ddp_find_unused_parameters'
        ]
        for field in optional_fields:
            if field in config:
                finetune_config[field] = config[field]

        # Convert to JSON string
        config_json = json.dumps(finetune_config)

        # Temporarily replace sys.argv to simulate command line arguments
        original_argv = sys.argv
        sys.argv = [original_argv[0], config_json]

        # Create and run the pipeline
        pipeline = Finetuning_Pipeline.FineTuningPipeline()
        result = pipeline.run()

        # Restore original sys.argv
        sys.argv = original_argv

        return result
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

def process_inference(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run model inference using VLLMManager based on the provided configuration."""
    try:
        # Check required parameters for inference
        required_fields = ['base_model', 'adapter_path', 'prompt']
        for field in required_fields:
            if field not in config:
                return {'status': 'error', 'message': f"Missing required field for inference: {field}"}

        # Extract optional parameters with defaults
        max_tokens = config.get('max_tokens', 300)
        temperature = config.get('temperature', 0.7)
        gpu_utilization = config.get('gpu_utilization', 0.8)

        # Initialize VLLMManager
        manager = VLLMManager()

        # Check if we should shut down the server after inference
        shutdown_after = config.get('shutdown_after', False)

        try:
            # Start the vLLM server (or use existing one if compatible)
            manager.start_vllm_server(
                base_model=config['base_model'],
                adapter_path=config['adapter_path'],
                gpu_utilization=gpu_utilization
            )

            # Generate response
            response = manager.generate_response(
                prompt=config['prompt'],
                max_tokens=max_tokens,
                temperature=temperature
            )

            result = {
                'status': 'success',
                'response': response,
                'model': config['base_model'],
                'adapter': config['adapter_path']
            }

            # Shutdown server if requested
            if shutdown_after:
                manager.kill_existing_vllm_process()
                result['server_shutdown'] = True

            return result

        except Exception as e:
            # If error occurs during server management or inference
            logger.error(f"Inference error: {e}")
            # Try to clean up
            manager.kill_existing_vllm_process()
            return {'status': 'error', 'message': f"Inference process failed: {str(e)}"}

    except Exception as e:
        logger.error(f"Error setting up inference: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handler for PDF processing, fine-tuning, and inference jobs")
    parser.add_argument('--config', type=str, required=True,
                      help='Configuration either as a JSON string or path to JSON file')
    parser.add_argument('--mode', type=str, required=True, choices=['pdf', 'fine_tune', 'inference'],
                      help='Task mode: "pdf" for processing PDF, "fine_tune" for model fine-tuning, "inference" for model inference')
    parser.add_argument('--pdf_file', type=str, help='Path to the PDF file (overrides config parameter)')
    parser.add_argument('--base_model', type=str, help='Base model path (overrides config parameter)')
    parser.add_argument('--adapter_path', type=str, help='Path to LoRA adapter (overrides config parameter)')
    parser.add_argument('--prompt', type=str, help='Inference prompt (overrides config parameter)')
    parser.add_argument('--gpu_utilization', type=float, help='GPU utilization (overrides config parameter)')
    parser.add_argument('--num_threads', type=int, help='Number of threads for parallel processing (overrides config parameter)')
    parser.add_argument('--shutdown_after', action='store_true', help='Shutdown vLLM server after inference (overrides config parameter)')
    parser.add_argument('--max_tokens', type=int, help='Maximum tokens for generation (overrides config parameter)')
    parser.add_argument('--temperature', type=float, help='Temperature for generation (overrides config parameter)')
    args = parser.parse_args()

    try:
        config = read_config(args.config)
        if not config:
            logger.error("Failed to parse configuration. Exiting.")
            sys.exit(1)

        # Override with command-line arguments
        if args.pdf_file:
            config['pdf_file'] = args.pdf_file
        if args.base_model:
            config['base_model'] = args.base_model
        if args.gpu_utilization is not None:
            config['gpu_utilization'] = args.gpu_utilization
        if args.num_threads is not None:
            config['num_threads'] = args.num_threads
        if args.adapter_path:
            config['adapter_path'] = args.adapter_path
        if args.prompt:
            config['prompt'] = args.prompt
        if args.shutdown_after:
            config['shutdown_after'] = True
        if args.max_tokens is not None:
            config['max_tokens'] = args.max_tokens
        if args.temperature is not None:
            config['temperature'] = args.temperature

        # Process based on selected mode
        if args.mode == 'pdf':
            result = process_pdf_command_line(config)
        elif args.mode == 'fine_tune':
            result = process_fine_tuning(config)
        elif args.mode == 'inference':
            result = process_inference(config)

        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get('status') == 'success' else 1)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        print(json.dumps({'status': 'error', 'message': str(e)}, indent=2))
        sys.exit(1)