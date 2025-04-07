#VLLM_Inference.py
import json
import requests
import subprocess
import time
import psutil
import shlex
import logging
import argparse
from typing import Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("inference_log.log", mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Global variables to track current model state
current_model = None
current_adapter = None

class VLLMManager:
    def __init__(self):
        self.server_process: Optional[subprocess.Popen] = None
        self.vllm_url = "http://localhost:8000"

    def normalize_path(self, path: str) -> str:
        """Normalize path for comparison."""
        return str(Path(path).resolve())

    def check_server_status(self) -> tuple[bool, Optional[str], Optional[str]]:
        """Check if vLLM server is running and get current model info from API."""
        try:
            response = requests.get(f"{self.vllm_url}/v1/models", timeout=5)
            if response.status_code == 200:
                logger.info("vLLM server is running")

                models_info = response.json()
                logger.info(f"Models info from API: {models_info}")

                current_model = None
                current_adapter = None

                # Look for the base model and LoRA adapter
                for model in models_info.get("data", []):
                    if model.get("id") == "assistant":
                        parent_model = model.get("parent")
                        if parent_model:

                            logger.info(f"Found parent model: {parent_model}")
                            current_model = parent_model

                        # Extract the root value as the adapter path
                        adapter_path = model.get("root")
                        if adapter_path:
                            current_adapter = adapter_path
                            logger.info(f"Found adapter path: {current_adapter}")

                return True, current_model, current_adapter

            return False, None, None
        except Exception as e:
            logger.info(f"No running vLLM server detected: {e}")
            return False, None, None

    def is_same_model(self, base_model: str, adapter_path: str) -> bool:
        """Compare if requested model matches currently loaded model."""
        is_running, curr_model, curr_adapter = self.check_server_status()

        if not is_running:
            return False
        base_model=f"base_model/{base_model}"
        logger.info(f"Current Model: {curr_model}, Requested Model: {base_model}")
        logger.info(f"Current Adapter: {curr_adapter}, Requested Adapter: {adapter_path}")

        # Normalize paths
        norm_curr_model = self.normalize_path(curr_model) if curr_model else None
        norm_request_model = self.normalize_path(base_model) if base_model else None

        norm_curr_adapter = self.normalize_path(curr_adapter) if curr_adapter else None
        norm_request_adapter = self.normalize_path(adapter_path) if adapter_path else None

        model_match = norm_curr_model == norm_request_model
        adapter_match = norm_curr_adapter == norm_request_adapter

        logger.info(f"Model match: {model_match}, Adapter match: {adapter_match}")

        return model_match and adapter_match

    def kill_existing_vllm_process(self):
        """Kill any existing vLLM processes."""
        global current_model, current_adapter

        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Error terminating server process: {e}")
                self.server_process.kill()
            self.server_process = None

        # Kill any other vLLM processes
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] and 'vllm' in proc.info['name']:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired) as e:
                    logger.error(f"Error killing process {proc.pid}: {e}")
                    proc.kill()

        current_model = None
        current_adapter = None
        logger.info("vLLM processes killed")

    def wait_for_server(self, max_attempts: int = 150) -> bool:
        """Wait for the vLLM server to become available."""
        url = f"{self.vllm_url}/health"
        logger.info(f"Waiting for vLLM server at {url}")

        for _ in range(max_attempts):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info("vLLM server is ready!")
                    return True
            except requests.exceptions.RequestException as e:
                # logger.info(f"Attempt {_+1} failed: {str(e)}")
                time.sleep(2)

        logger.error("vLLM server failed to start")
        return False

    def start_vllm_server(self, base_model: str, adapter_path: str, gpu_utilization: float = 0.8) -> None:
        """Start the vLLM server with the specified model, adapter, and GPU utilization."""
        global current_model, current_adapter

        # Check if server is already running with the same model and adapter
        if self.is_same_model(base_model, adapter_path):
            logger.info("Using existing vLLM server")
            return

        # Kill any existing processes
        self.kill_existing_vllm_process()

        # Set up command for starting the server
        lora_modules_json = f'{{"name": "assistant","path": "{shlex.quote(adapter_path)}"}}'
        base_model_path = f"base_model/{base_model}"
        cmd = [
            "vllm",
            "serve",
            base_model_path,
            f"--gpu_memory_utilization={gpu_utilization}",
            "--max_model_len=8096",
            "--enable-lora",
            "--max_lora_rank=32",
            "--lora-modules", lora_modules_json
        ]

        logger.info(f"Starting vLLM server: {' '.join(cmd)}")
        self.server_process = subprocess.Popen(cmd)

        # Wait for server to be ready
        if not self.wait_for_server():
            # If server didn't start, clean up
            self.kill_existing_vllm_process()
            raise Exception("vLLM server failed to start")

        # Update global current model and adapter
        current_model = base_model
        current_adapter = adapter_path
        logger.info(f"Server started with model: {base_model} and adapter: {adapter_path}")

    def generate_response(self, prompt: str, max_tokens: int = 300, temperature: float = 0.7) -> str:
            """Generate a response using the current model."""
            try:
                url = f"{self.vllm_url}/v1/chat/completions"
                headers = {"Content-Type": "application/json"}
                data = {
                    "model": "assistant",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }

                response = requests.post(url, headers=headers, json=data, timeout=300)
                response.raise_for_status()
                response_json = response.json()
                # Handle possible errors in response
                if 'error' in response_json:
                    raise ValueError(response_json['error'])
                return response_json['choices'][0]['message']['content']

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logger.error(error_msg)
                return error_msg

def main():
    parser = argparse.ArgumentParser(description="vLLM Model Interface")
    parser.add_argument("--input", type=str, help="Input JSON with model parameters")
    parser.add_argument("--shut_vllm", action="store_true", help="Shut down vLLM server")
    args = parser.parse_args()

    manager = VLLMManager()

    try:
        if args.shut_vllm:
            logger.info("Shutting down vLLM server...")
            manager.kill_existing_vllm_process()
            return

        if not args.input:
            raise ValueError("Either --input or --shut_vllm must be provided")

        # Parse input JSON
        input_data = json.loads(args.input)
        base_model = input_data.get("base_model")
        adapter_path = input_data.get("adapter_path")
        prompt = input_data.get("prompt")
        max_tokens = input_data.get("max_tokens", 300)
        temperature = input_data.get("temperature", 0.7)
        gpu_utilization = input_data.get("gpu_utilization", 0.8)

        # Validate required parameters
        if not all([base_model, adapter_path, prompt]):
            raise ValueError("base_model, adapter_path, and prompt are required in input JSON")

        # Ensure paths are properly quoted and normalized
        base_model = base_model.strip()
        adapter_path = adapter_path.strip()

        # Start the server if needed
        manager.start_vllm_server(base_model, adapter_path, gpu_utilization)

        # Generate response
        response = manager.generate_response(prompt, max_tokens, temperature)
        print("Generated Response:")
        print(response)

    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        # Cleanup if server wasn't successfully started
        if not (current_model and current_adapter):
            manager.kill_existing_vllm_process()

if __name__ == "__main__":
    main()