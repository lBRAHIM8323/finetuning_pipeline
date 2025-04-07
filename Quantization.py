import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import HfApi
import sys
import logging
import json
# Simplified logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_input_data():
    """Load input data from command line arguments."""
    try:
        # Check if input is provided via command-line argument
        if len(sys.argv) > 1:
            return json.loads(sys.argv[1])
        else:
            logging.error("No input data provided")
            sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON input: {e}")
        sys.exit(1)


def quantize_and_push_awq(repo_id):
    """Quantize the model and push it to Hugging Face Hub."""
    # Initialize Hugging Face API
    api = HfApi()

    model_path = f'repo_id'
    quant_path = f'repo_id-awq'
    repo_id = f'repoid-awq'

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }

    # Load model and tokenizer
    try:
        model = AutoAWQForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, use_cache=False
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    # Quantize the model
    try:
        model.quantize(tokenizer, quant_config=quant_config)
    except Exception as e:
        print(f"Error during model quantization: {e}")
        return

    # Save quantized model and tokenizer
    try:
        model.save_quantized(quant_path)
        tokenizer.save_pretrained(quant_path)
        print(f'Model is quantized and saved at "{quant_path}"')
    except Exception as e:
        print(f"Error saving quantized model: {e}")
        return

        # Create the repository if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
    except Exception as e:
        print(f"Repository creation warning (can be ignored if repo exists): {e}")
    # Upload all files from the quantized model directory
    def upload_folder(local_folder, repo_id, token):
        """Upload the contents of a local folder to the Hugging Face Hub."""
        api.upload_folder(
            folder_path=local_folder,
            repo_id=repo_id,
            repo_type="model",
            token=token
        )

    # Upload the model
    try:
        upload_folder(quant_path, repo_id, hf_token)
        print(f"Successfully uploaded model to {repo_id}")
    except Exception as e:
        print(f"Error during upload: {e}")

if __name__ == "__main__":
    # Load input data
    input_data = load_input_data()
    repo_id = input_data["repo_id"]
    hf_token =input_data["hf_token"]
    quantize_and_push_awq(repo_id,hf_token)