import json
import re
import os
import sys
import time
import logging
import signal
import atexit
import subprocess
import gc
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import requests
from packaging.version import Version as V
from transformers import (
    TrainingArguments,
    TrainerCallback,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM
)
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from huggingface_hub import HfApi, create_repo, upload_file, CommitOperationAdd

import psutil

# Unsloth imports
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

# AWQ imports
from awq import AutoAWQForCausalLM

# Set the GLOO_SOCKET_IFNAME environment variable
os.environ["GLOO_SOCKET_IFNAME"] = "lo"


class Logger:
    """Handles all logging functionality"""

    @staticmethod
    def setup():
        """Configure logging with proper formatting and handlers"""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("finetuning_pipeline.log", mode='a')
            ]
        )
        return logging.getLogger(__name__)

logger =Logger.setup()

class ConfigManager:
    """Handles configuration loading and validation"""

    @staticmethod
    def parse_input():
        """Parse and validate JSON input from command line arguments"""
        try:
            # Validate that we have an argument
            if len(sys.argv) < 2:
                raise ValueError("No input JSON provided")

            # Parse the JSON input from command line argument
            input_data = json.loads(sys.argv[1])
            return input_data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {e}")
            sys.exit(1)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            sys.exit(1)

    @staticmethod
    def validate_config(config):
        """Validate that all required fields are present in the configuration"""
        required_fields = [
            'json_file', 'base_model', 'system_prompt', 'repo_id',
            'max_step', 'learning_rate', 'epochs'
        ]
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        return config


class DatasetManager:
    """Handles dataset loading and preprocessing"""

    def __init__(self, config, tokenizer, logger):
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logger
        self.system_prompt = config['system_prompt']

    def load_data_from_json(self):
        """Load dataset from JSON file specified in config"""
        try:
            json_file = self.config.get("json_file")
            if os.path.exists(json_file):
                self.logger.info(f"Loading dataset from local file: {json_file}")
                return load_dataset("json", data_files=json_file)["train"]
            else:
                raise FileNotFoundError(f"Local JSON file not found: {json_file}")
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            return None

    def formatting_prompts_func(self, examples):
        """Format prompts and responses according to the chat template"""
        prompts = examples['prompt']
        responses = examples['response']
        text = []
        for prompt, response in zip(prompts, responses):
            prompt = str(prompt).strip()
            response = str(response).strip()
            if not prompt.endswith(('?', '.', '!', ':')):
                prompt += '.'
            response = response.capitalize()
            if not response.endswith(('.', '?', '!')):
                response += '.'
            convo = [
                {'from': 'system', 'value': self.system_prompt},
                {'from': 'human', 'value': prompt},
                {'from': 'gpt', 'value': response}
            ]
            text.append(self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False))
        return {"text": text}

    def prepare_dataset(self):
        """Load and preprocess the dataset"""
        dataset = self.load_data_from_json()
        if dataset is None:
            raise ValueError("Failed to load dataset")

        self.logger.info("Dataset loaded successfully!")
        self.logger.info(dataset[:5])
        dataset = dataset.map(self.formatting_prompts_func, batched=True)
        self.logger.info(dataset["text"][0])
        return dataset


class ModelManager:
    """Handles model loading and configuration"""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.max_seq_length = 2048
        self.dtype = None  # Auto detection

    def load_model(self):
        """Load the base model and configure it for fine-tuning"""
        base_model = self.config.get('base_model')
        base_model_path = f"base_model/{base_model}"

        # Check if local model path exists
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"Local model path not found: {base_model_path}")

        # Load model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_path,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit = True
        )

        # Configure PEFT
        model = FastLanguageModel.get_peft_model(
            model,
            r=64,  # LoRA rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=True,
            loftq_config=None,
        )

        # Set up tokenizer with chat template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="chatml",
            mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
            map_eos_token=True,
        )

        return model, tokenizer

    def save_model(self, model, tokenizer):
        """Save model locally and optionally push to Hugging Face Hub"""
        repo_id = self.config['repo_id']
        push_to_hf = self.config.get('push_to_hf', False)
        hf_token = self.config.get('hf_token', None)
        base_model = self.config.get('base_model', 'unknown')
        folder_path = None

        if repo_id:
            # Split repo_id into username and model name
            username, model_name = repo_id.split('/')

            # Create the directory path
            folder_path = os.path.join(username.lower(), model_name.lower())
            os.makedirs(folder_path, exist_ok=True)

            self.logger.info(f"Saving the model locally to {folder_path}")
            # Save the model and tokenizer using the merged approach
            model.save_pretrained(folder_path, tokenizer, save_method="lora")

            # Verify files were saved
            saved_files = os.listdir(folder_path)
            self.logger.info(f"Saved files: {saved_files}")

            # Update model list locally
            model_info = {
                "model": folder_path,
                "base_model": base_model
            }
            self.update_model_list(model_info)
            self.logger.info(f"Model list updated successfully for folder_path: {folder_path}")

        # Push to hub if enabled
        if push_to_hf:
            if not repo_id:
                self.logger.error("repo_id is required to push to Hugging Face Hub")
                raise ValueError("repo_id is required to push to Hugging Face Hub")

            if hf_token is None:
                self.logger.warning("No HF token provided. Attempting to push with cached credentials.")

            self.logger.info(f"Attempting to push model to {repo_id}")
            model.push_to_hub_merged(repo_id, tokenizer, save_method="lora", token=hf_token)

            # Update the centralized model list on Hugging Face
            try:
                self.update_centralized_model_list(repo_id, f"unsloth/{base_model}", hf_token)
            except Exception as e:
                self.logger.error(f"Failed to update centralized model list: {str(e)}")

            model_info = {
                "model": repo_id,
                "base_model": f"unsloth/{base_model}"
            }
            self.logger.info("Successfully pushed model to Hugging Face Hub")

        return folder_path

    def load_huggingface_json(self, repo_id: str, filename: str, save_path: str = None, hf_token=None):
        """
        Load a JSON file from a HuggingFace dataset repository using authentication.
        """
        url = f"https://huggingface.co/datasets/{repo_id}/raw/main/{filename}"
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w') as f:
                    json.dump(data, f, indent=2)

            return data

        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                self.logger.error("Error: Unauthorized access. Check your Hugging Face token and permissions.")
            else:
                self.logger.error(f"Error fetching the file: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON: {e}")
            return None

    def update_centralized_model_list(self, repo_id, base_model, hf_token):
        model_list_repo = "PharynxAI/Model_List"
        model_list_file = "model_list.json"

        # Use self.load_huggingface_json instead of HfApi
        existing_data = self.load_huggingface_json(
            repo_id=model_list_repo,
            filename=model_list_file,
            hf_token=hf_token
        )

        # Ensure model_list is initialized correctly
        model_list = existing_data if isinstance(existing_data, list) else []

        model_info = {
            "model": repo_id,
            "base_model": base_model
        }

        # Check if model entry already exists
        found = False
        for i, entry in enumerate(model_list):
            if entry.get("model") == model_info["model"]:
                model_list[i] = model_info
                found = True
                self.logger.info(f"Updated existing entry for model: {model_info['model']}")
                break

        if not found:
            model_list.append(model_info)
            self.logger.info(f"Added new entry for model: {model_info['model']}")

        temp_local_path = "temp_model_list.json"
        with open(temp_local_path, 'w') as f:
            json.dump(model_list, f, indent=4)

        try:
            # Use HfApi to upload the file
            api = HfApi()
            api.upload_file(
                path_or_fileobj=temp_local_path,
                path_in_repo=model_list_file,
                repo_id=model_list_repo,
                repo_type="dataset",
                token=hf_token,
                commit_message=f"Update model list: {repo_id}"
            )
            self.logger.info(f"Successfully updated model list with {repo_id}")
        except Exception as e:
            self.logger.error(f"Failed to update centralized model list: {str(e)}")
        finally:
            # Remove the temporary file
            if os.path.exists(temp_local_path):
                os.unlink(temp_local_path)

    def update_model_list(self, model_info):
        """Update the local list of available models"""
        model_list_file = "model_list.json"
        try:
            # Ensure the file exists and is not empty
            if os.path.exists(model_list_file) and os.stat(model_list_file).st_size > 0:
                with open(model_list_file, 'r') as f:
                    try:
                        model_list = json.load(f)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Corrupted JSON detected in {model_list_file}, resetting.")
                        model_list = []
            else:
                model_list = []  # Start fresh if file is missing or empty

            # Add or update model information
            found = False
            for i, entry in enumerate(model_list):
                if entry.get("model") == model_info["model"]:
                    model_list[i] = {**entry, **model_info}  # Update existing entry
                    found = True
                    self.logger.info(f"Updated existing entry for model: {model_info['model']}")
                    break

            if not found:
                model_list.append(model_info)  # Add new model info
                self.logger.info(f"Added new entry for model: {model_info['model']}")

            # Save the updated list to the file
            with open(model_list_file, 'w') as f:
                json.dump(model_list, f, indent=4)

            self.logger.debug(f"Saved updated model list locally to {model_list_file}")

        except Exception as e:
            self.logger.error(f"Failed to update model list: {str(e)}")

    def run_quantization(self):
        """Run the quantization process if requested"""
        repo_id = self.config['repo_id']
        access_type = self.config.get('Access_Type', '')
        hf_token = self.config.get('hf_token', None)

        try:
            input_json = json.dumps({
                "repo_id": repo_id,
                "Access_Type": access_type,
                "hf_token": hf_token
            })

            process = subprocess.Popen(
                ["python", "Quantization.py", input_json],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Stream the logs in real time
            for line in iter(process.stdout.readline, ''):
                self.logger.info("STDOUT: %s", line.strip())

            process.stdout.close()
            return_code = process.wait()

            if return_code == 0:
                self.logger.info("Quantization successful")
                return True
            else:
                self.logger.error("Quantization failed with return code: %d", return_code)
                return False

        except Exception as e:
            self.logger.error("Error during quantization: %s", str(e))
            return False


class TensorBoardManager:
    """Manages TensorBoard for training visualization"""

    def __init__(self, log_dir="tensorboard_logs", logger=None):
        self.log_dir = log_dir
        self.tensorboard_process = None
        self.logger = logger
        # Register cleanup on exit
        atexit.register(self.stop_tensorboard)

    def start_tensorboard(self):
        """Start TensorBoard server as a subprocess"""
        try:
            # Create logs directory if it doesn't exist
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)

            # Start TensorBoard process
            self.tensorboard_process = subprocess.Popen(
                ["tensorboard", "--logdir", self.log_dir, "--host", "0.0.0.0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )

            # Give TensorBoard time to start
            time.sleep(5)

            self.logger.info(f"TensorBoard started. Access it at http://localhost:6006")
            return True

        except Exception as e:
            self.logger.info(f"Failed to start TensorBoard: {str(e)}")
            return False

    def stop_tensorboard(self):
        """Stop TensorBoard server"""
        if self.tensorboard_process:
            try:
                # Kill the entire process group
                os.killpg(os.getpgid(self.tensorboard_process.pid), signal.SIGTERM)
                self.tensorboard_process.wait(timeout=5)  # Wait for process to terminate
                self.logger.info("TensorBoard stopped successfully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                os.killpg(os.getpgid(self.tensorboard_process.pid), signal.SIGKILL)
                self.logger.info("TensorBoard force stopped")
            except Exception as e:
                self.logger.info(f"Error stopping TensorBoard: {str(e)}")
            finally:
                self.tensorboard_process = None


class TrackBestModelCallback(TrainerCallback):
    """Callback to track and save the best model during training"""

    def __init__(self, output_dir, writer, logger):
        super().__init__()
        self.best_loss = float('inf')
        self.best_step = 0
        self.output_dir = output_dir
        self.writer = writer
        self.logger = logger
        os.makedirs(self.output_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        train_loss = logs.get("loss")
        if train_loss is not None:
            # Log the training loss to TensorBoard
            self.writer.add_scalar('Training/Loss', train_loss, state.global_step)

            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.best_step = state.global_step
                # Log best loss to TensorBoard
                self.writer.add_scalar('Training/Best_Loss', self.best_loss, state.global_step)

                # Save the model to disk
                model_path = os.path.join(self.output_dir, f"best_model_step_{self.best_step}.pt")
                torch.save(kwargs['model'].state_dict(), model_path)
                self.logger.info(f"New best model saved at {model_path} with loss: {self.best_loss}")

                # Remove the previous best model if it exists
                for file in os.listdir(self.output_dir):
                    if file.startswith("best_model_step_") and file != f"best_model_step_{self.best_step}.pt":
                        os.remove(os.path.join(self.output_dir, file))

    def on_train_end(self, args, state, control, **kwargs):
        self.logger.info("Training ended. Best model is saved on disk.")
        self.writer.close()


class TrainingManager:
    """Manages the training process"""

    def __init__(self, config, model, tokenizer, dataset, logger):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.logger = logger
        self.tensorboard_dir = "tensorboard_logs"
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        self.tensorboard = TensorBoardManager(self.tensorboard_dir, logger)
        self.output_dir = "outputs"
        self.trainer = None
        self.best_model_tracker = None
        self.start_gpu_memory = None

    def configure_trainer(self):
        """Configure the SFT trainer with all parameters"""
        max_seq_length = 2048

        # Extract training parameters from config
        per_device_train_batch_size = self.config.get('per_device_train_batch_size', 4)
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 2)
        warmup_steps = self.config.get('warmup_steps', 100)
        logging_steps = self.config.get('logging_steps', 50)
        optim = self.config.get('optim', "adamw_8bit")
        weight_decay = self.config.get('weight_decay', 0.01)
        lr_scheduler_type = self.config.get('lr_scheduler_type', "cosine")
        max_grad_norm = self.config.get('max_grad_norm', 1.0)
        dataloader_num_workers = self.config.get('dataloader_num_workers', 0)
        gradient_checkpointing = self.config.get('gradient_checkpointing', True)
        adam_beta1 = self.config.get('adam_beta1', 0.9)
        adam_beta2 = self.config.get('adam_beta2', 0.999)
        adam_epsilon = self.config.get('adam_epsilon', 1e-8)
        ddp_find_unused_parameters = self.config.get('ddp_find_unused_parameters', False)
        max_step = self.config.get('max_step')
        learning_rate = self.config.get('learning_rate')
        epochs = self.config.get('epochs')

        # Create training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=epochs,
            max_steps=max_step,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=logging_steps,
            optim=optim,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            seed=3407,
            output_dir=self.output_dir,
            max_grad_norm=max_grad_norm,
            dataloader_num_workers=dataloader_num_workers,
            gradient_checkpointing=gradient_checkpointing,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            ddp_find_unused_parameters=ddp_find_unused_parameters,
            report_to=["tensorboard"],

            eval_steps=None,
            dataloader_pin_memory=True,  # Faster data transfer to GPU
            torch_compile=True,  # Use PyTorch 2.0 compiler for faster execution

        )

        # Create SFT trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=dataloader_num_workers,
            packing=False,
            args=training_args,
        )

        # Add callback to track the best model
        self.best_model_tracker = TrackBestModelCallback(self.output_dir, self.writer, self.logger)
        self.trainer.add_callback(self.best_model_tracker)

    def log_memory_stats(self, is_start=True):
        """Log current GPU memory usage"""
        gpu_stats = torch.cuda.get_device_properties(0)
        current_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        if is_start:
            self.start_gpu_memory = current_gpu_memory
            self.logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
            self.logger.info(f"{self.start_gpu_memory} GB of memory reserved.")
        else:
            used_memory_for_lora = round(current_gpu_memory - self.start_gpu_memory, 3)
            used_percentage = round(current_gpu_memory / max_memory * 100, 3)
            lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
            self.logger.info(f"Peak reserved memory = {current_gpu_memory} GB.")
            self.logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
            self.logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
            self.logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    def train(self):
        """Run the training process"""
        try:
            self.tensorboard.start_tensorboard()
            self.log_memory_stats(is_start=True)

            # Start training
            trainer_stats = self.trainer.train()

            # Log final stats
            self.logger.info("Training completed.")
            self.logger.info(f"Best loss: {self.best_model_tracker.best_loss} at step {self.best_model_tracker.best_step}")
            self.logger.info(f"Final training loss: {trainer_stats.training_loss}")

            # Log time stats
            self.logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
            self.logger.info(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")

            # Log memory stats
            self.log_memory_stats(is_start=False)

            return trainer_stats

        finally:
            # Always stop TensorBoard when done
            self.tensorboard.stop_tensorboard()
            self.writer.close()


class FineTuningPipeline:
    """Main pipeline class that orchestrates the entire fine-tuning process"""

    def __init__(self):
        self.logger = Logger.setup()
        self.config = None
        self.model = None
        self.tokenizer = None
        self.dataset = None

    def load_config(self):
        """Load and validate configuration"""
        config = ConfigManager.parse_input()
        self.config = ConfigManager.validate_config(config)

    # In the FineTuningPipeline.run method, modify the quantize check as follows:
    def run(self):
        """Run the complete fine-tuning pipeline"""
        try:
            # Load configuration
            self.load_config()

            # Initialize model manager and load model
            model_manager = ModelManager(self.config, self.logger)
            self.model, self.tokenizer = model_manager.load_model()

            # Load and prepare dataset
            dataset_manager = DatasetManager(self.config, self.tokenizer, self.logger)
            self.dataset = dataset_manager.prepare_dataset()

            # Train the model
            training_manager = TrainingManager(
                self.config, self.model, self.tokenizer, self.dataset, self.logger
            )
            training_manager.configure_trainer()
            trainer_stats = training_manager.train()

            # Save the final model
            saved_path = model_manager.save_model(self.model, self.tokenizer)

            # Run quantization if requested - FIX HERE
            quantize = self.config.get("quantize", "no")
            # Handle both string and boolean values
            if isinstance(quantize, bool):
                should_quantize = quantize
            else:
                should_quantize = quantize.lower() == "yes"

            if should_quantize:
                self.logger.info("Quantization requested. Starting quantization process...")
                if model_manager.run_quantization():
                    quantized_model_info = {
                        "model": f"{self.config['repo_id']}-awq",
                    }
                    model_manager.update_model_list(quantized_model_info)
                    self.logger.info(f"Quantization completed for {self.config['repo_id']}")
            else:
                self.logger.info("Quantization not requested. Skipping quantization process.")

            return {
                "status": "success",
                "model_path": saved_path,
                "training_stats": trainer_stats.metrics,
                "best_loss": training_manager.best_model_tracker.best_loss,
                "best_step": training_manager.best_model_tracker.best_step
            }

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

# Main entry point
if __name__ == "__main__":
    start_time = time.time()
    pipeline = FineTuningPipeline()
    result = pipeline.run()

    if result["status"] == "success":
        logger = logging.getLogger(__name__)

        logger.info("Fine-tuning pipeline completed successfully!")
        logger.info(f"process_completed in processing_time: {time.time() - start_time}")
        sys.exit(0)
    else:
        logger = logging.getLogger(__name__)
        logger.error(f"Fine-tuning pipeline failed: {result['error']}")
        sys.exit(1)