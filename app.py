import gradio as gr
import json
import os
import logging
from typing import Dict, Any
import pandas as pd
from pathlib import Path
import tempfile

# Import Handler modules directly
import Handler
from pdf_processor import EnhancedPDFProcessor
import Finetuning_Pipeline
from VLLM_Inference import VLLMManager

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
file_handler = logging.FileHandler('gradio_ui.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Initialize global instances
pdf_processor = EnhancedPDFProcessor()
vllm_manager = VLLMManager()

# For storing temporary files and results
temp_dir = Path(tempfile.gettempdir()) / "handler_gradio"
os.makedirs(temp_dir, exist_ok=True)

# Create a StringIO handler to capture logs for display in the UI
class StringIOHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_output = []

    def emit(self, record):
        self.log_output.append(self.format(record))

    def get_logs(self):
        return "\n".join(self.log_output)

    def clear(self):
        self.log_output = []

# Add the custom handler
string_handler = StringIOHandler()
string_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(string_handler)

def process_pdf(
    pdf_file,
    num_threads: int = 4,
    base_model: str = "",
    gpu_utilization: float = 0.8,
    additional_config: str = "{}"
) -> Dict:
    """Process PDF and generate QA pairs"""
    try:
        if not pdf_file:
            return {"status": "error", "message": "No PDF file provided"}, None, None

        # Create config dictionary
        config = {
            "pdf_file": pdf_file.name,
            "num_threads": num_threads,
            "gpu_utilization": gpu_utilization
        }

        if base_model:
            config["base_model"] = base_model

        # Parse additional config if provided
        try:
            additional = json.loads(additional_config)
            config.update(additional)
        except json.JSONDecodeError:
            return {"status": "error", "message": f"Invalid additional config JSON: {additional_config}"}, None, None

        # Process PDF
        logger.info(f"Processing PDF with config: {config}")

        # Use Handler.process_pdf_command_line
        result = Handler.process_pdf_command_line(config)

        if result['status'] == 'success':
            # Load generated QA pairs for display
            qa_pairs = None
            qa_file = None
            if 'output_file' in result and os.path.exists(result['output_file']):
                with open(result['output_file'], 'r') as f:
                    qa_data = json.load(f)
                    qa_pairs = pd.DataFrame(qa_data)
                qa_file = result['output_file']

            return result, qa_pairs, qa_file
        else:
            return result, None, None

    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}, None, None


def run_fine_tuning(
    json_file,
    base_model: str,
    system_prompt: str,
    repo_id: str,
    max_step: int,
    learning_rate: float,
    epochs: int,
    push_to_hf: bool = False,
    hf_token: str = "",
    quantize: bool = False,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    warmup_steps: int = 0,
    logging_steps: int = 1,
    optim: str = "paged_adamw_8bit",
    weight_decay: float = 0.01,
    lr_scheduler_type: str = "cosine",
    max_grad_norm: float = 1.0,
    dataloader_num_workers: int = 4,
    gradient_checkpointing: bool = True,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    ddp_find_unused_parameters: bool = False,
) -> Dict:
    """Run fine-tuning pipeline"""
    # Clear previous logs
    string_handler.clear()

    try:
        if not json_file:
            return {"status": "error", "message": "No JSON file provided"}, ""

        # Prepare config
        config = {
            "json_file": json_file.name,
            "base_model": base_model,
            "system_prompt": system_prompt,
            "repo_id": repo_id,
            "max_step": max_step,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "push_to_hf": push_to_hf,
            "quantize": quantize,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_steps": warmup_steps,
            "logging_steps": logging_steps,
            "optim": optim,
            "weight_decay": weight_decay,
            "lr_scheduler_type": lr_scheduler_type,
            "max_grad_norm": max_grad_norm,
            "dataloader_num_workers": dataloader_num_workers,
            "gradient_checkpointing": gradient_checkpointing,
            "adam_beta1": adam_beta1,
            "adam_beta2": adam_beta2,
            "adam_epsilon": adam_epsilon,
            "ddp_find_unused_parameters": ddp_find_unused_parameters
        }

        if hf_token and push_to_hf:
            config["hf_token"] = hf_token

        logger.info(f"Running fine-tuning with config: {config}")

        # Use Handler.process_fine_tuning
        result = Handler.process_fine_tuning(config)

        # Get captured logs
        logs = string_handler.get_logs()

        return result, logs

    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}", exc_info=True)
        logs = string_handler.get_logs()
        return {"status": "error", "message": str(e)}, logs

def run_inference(
    base_model: str,
    adapter_path: str,
    prompt: str,
    max_tokens: int = 300,
    temperature: float = 0.7,
    gpu_utilization: float = 0.8,
    shutdown_after: bool = False
) -> Dict:
    """Run inference using VLLM"""
    try:
        # Validate inputs
        if not base_model:
            return {"status": "error", "message": "Base model is required"}
        if not adapter_path:
            return {"status": "error", "message": "Adapter path is required"}
        if not prompt:
            return {"status": "error", "message": "Prompt is required"}

        logger.info(f"Running inference with model: {base_model}, adapter: {adapter_path}")

        # Create config for Handler.process_inference
        config = {
            "base_model": base_model,
            "adapter_path": adapter_path,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "gpu_utilization": gpu_utilization,
            "shutdown_after": shutdown_after
        }

        # Use Handler.process_inference
        result = Handler.process_inference(config)

        return result

    except Exception as e:
        logger.error(f"Error setting up inference: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

def clean_temp_files():
    """Clean up temporary files"""
    try:
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
        return {"status": "success", "message": "Temporary files cleaned"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to clean temporary files: {str(e)}"}

def push_qa_to_hf(qa_file_path: str, repo_id: str, hf_token: str) -> Dict:
    """Push QA pairs JSON file to Hugging Face Hub"""
    try:
        import huggingface_hub

        if not qa_file_path or not os.path.exists(qa_file_path):
            return {"status": "error", "message": "QA file doesn't exist"}
        if not repo_id:
            return {"status": "error", "message": "Repository ID is required"}
        if not hf_token:
            return {"status": "error", "message": "Hugging Face token is required"}

        # Get filename
        filename = os.path.basename(qa_file_path)

        # Push to hub
        huggingface_hub.upload_file(
            path_or_fileobj=qa_file_path,
            path_in_repo=filename,
            repo_id=repo_id,
            token=hf_token
        )

        return {
            "status": "success",
            "message": f"Successfully pushed {filename} to {repo_id}",
            "url": f"https://huggingface.co/{repo_id}/blob/main/{filename}"
        }

    except Exception as e:
        logger.error(f"Error pushing to Hugging Face: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

# Define Gradio UI
def create_ui():
    # Define theme
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
    )

    # Create Gradio blocks
    with gr.Blocks(theme=theme, title="AI Model Finetuner") as demo:
        gr.Markdown("# AI Model Finetuner")
        gr.Markdown("Process PDFs, fine-tune models, and run inference with your models")

        # Create tabs for different functionalities
        with gr.Tabs() as tabs:
            # PDF Processing Tab
            with gr.Tab("PDF Processing"):
                with gr.Row():
                        pdf_file = gr.File(label="PDF File", file_types=[".pdf"])
                with gr.Row():
                        pdf_base_model = gr.Textbox(label="Base Model",value="Meta-Llama-3.1-8B-Instruct")
                        pdf_num_threads = gr.Slider(label="Number of Threads", minimum=1, maximum=16, value=4, step=1)
                        pdf_gpu_util = gr.Slider(label="GPU Utilization", minimum=0.1, maximum=1.0, value=0.8, step=0.1)
                        pdf_additional_config = gr.Textbox(label="Additional Config (JSON)", value="{}", lines=3)

                with gr.Row():
                    pdf_process_btn = gr.Button("Process PDF", variant="primary")
                    # pdf_clear_btn = gr.Button("Clear")

                with gr.Row():
                    pdf_output = gr.JSON(label="Processing Result")
                    qa_output = gr.Dataframe(label="Generated QA Pairs", wrap=True)

                with gr.Row():
                    download_btn = gr.Button("Download QA Pairs", variant="primary")
                    clean_btn = gr.Button("Clean Temporary Files")

                with gr.Accordion("Push to Hugging Face (Optional)", open=False):
                    gr.Markdown("*Optionally share your QA pairs to Hugging Face*")
                    hf_repo_id = gr.Textbox(label="Repository ID (e.g. username/repo-name)")
                    hf_token = gr.Textbox(label="Hugging Face Token", type="password")
                    push_btn = gr.Button("Push to Hugging Face")
                    push_result = gr.JSON(label="Push Result")

                # Hidden values for tracking
                qa_file_path = gr.Textbox(visible=False)

                # Event handlers
                pdf_process_btn.click(
                    fn=process_pdf,
                    inputs=[pdf_file, pdf_num_threads, pdf_base_model, pdf_gpu_util, pdf_additional_config],
                    outputs=[pdf_output, qa_output, qa_file_path]
                )

                # pdf_clear_btn.click(
                #     fn=lambda: (None, None, None, "", "", 4, 0.8, "{}"),
                #     inputs=[],
                #     outputs=[pdf_file, pdf_base_model, pdf_output, qa_output, qa_file_path,
                #             pdf_num_threads, pdf_gpu_util, pdf_additional_config]
                # )

                push_btn.click(
                    fn=push_qa_to_hf,
                    inputs=[qa_file_path, hf_repo_id, hf_token],
                    outputs=[push_result]
                )

                download_btn.click(
                    fn=lambda x: x if x else None,
                    inputs=[qa_file_path],
                    outputs=[gr.File(label="Download QA Pairs")]
                )

                # clean_btn.click(
                #     fn=clean_temp_files,
                #     inputs=[],
                #     outputs=[pdf_output]
                # )

            # Fine-tuning Tab
            # Fine-tuning Tab
            with gr.Tab("Fine-tuning"):
                with gr.Row():
                    ft_json_file = gr.File(label="Training Data JSON File", file_types=[".json"])

                # Basic setup
                with gr.Group():
                    with gr.Row():
                        ft_base_model = gr.Textbox(
                            label="Base Model Path",
                            value="Meta-Llama-3.1-8B-Instruct",
                            info="HuggingFace model ID or local path to model"
                        )
                        ft_system_prompt = gr.Textbox(
                            label="System Prompt",
                            value="You are a helpful assistant that provides detailed information based on the provided text.",
                            lines=3,
                            info="System prompt to use for instruction following models"
                        )
                        ft_repo_id = gr.Textbox(
                            label="Output Repository ID",
                            info="Hugging Face repo ID where the model will be saved (e.g., username/model-name)"
                        )

                # Training parameters
                with gr.Accordion("Training Parameters", open=True):
                    with gr.Row():
                        ft_max_step = gr.Number(
                            label="Max Steps",
                            value=150,
                            precision=0,
                            info="Maximum number of training steps"
                        )
                        ft_learning_rate = gr.Number(
                            label="Learning Rate",
                            value=0.0002,
                            info="Learning rate for optimizer"
                        )
                        ft_epochs = gr.Number(
                            label="Epochs",
                            value=2,
                            precision=0,
                            info="Number of training epochs"
                        )
                        ft_batch_size = gr.Number(
                            label="Batch Size",
                            value=4,
                            precision=0,
                            info="Per device training batch size"
                        )
                    with gr.Row():
                        ft_grad_accum = gr.Number(
                            label="Gradient Accumulation Steps",
                            value=2,
                            precision=0,
                            info="Number of updates steps to accumulate before performing a backward/update pass"
                        )
                        ft_warmup = gr.Number(
                            label="Warmup Steps",
                            value=200,
                            precision=0,
                            info="Linear warmup over warmup_steps"
                        )
                        ft_logging_steps = gr.Number(
                            label="Logging Steps",
                            value=50,
                            precision=0,
                            info="Log every X updates steps"
                        )

                    with gr.Row():
                        ft_weight_decay = gr.Number(
                            label="Weight Decay",
                            value=0.01,
                            info="Weight decay for AdamW optimizer"
                        )
                        ft_max_grad_norm = gr.Number(
                            label="Max Gradient Norm",
                            value=1.0,
                            info="Max gradient norm for gradient clipping"
                        )

                    with gr.Row():
                        ft_optim = gr.Dropdown(
                            label="Optimizer",
                            choices=["paged_adamw_8bit", "adamw_torch", "adamw_8bit", "adamw_bnb_8bit"],
                            value="adamw_8bit",
                            info="Optimizer to use for training"
                        )
                        ft_scheduler = gr.Dropdown(
                            label="LR Scheduler",
                            choices=["cosine", "linear", "constant", "constant_with_warmup"],
                            value="cosine",
                            info="Learning rate scheduler to use"
                        )

                # Adam optimizer parameters
                with gr.Accordion("Adam Optimizer Parameters", open=False):
                    with gr.Row():
                        ft_adam_beta1 = gr.Number(
                            label="Adam Beta1",
                            value=0.9,
                            info="Beta1 for Adam optimizer"
                        )
                        ft_adam_beta2 = gr.Number(
                            label="Adam Beta2",
                            value=0.999,
                            info="Beta2 for Adam optimizer"
                        )
                        ft_adam_epsilon = gr.Number(
                            label="Adam Epsilon",
                            value=1e-8,
                            info="Epsilon for Adam optimizer"
                        )

                # Advanced options
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        ft_workers = gr.Number(
                            label="Dataloader Workers",
                            value=4,
                            precision=0,
                            info="Number of subprocesses to use for data loading"
                        )

                    with gr.Row():
                        ft_ddp_find_unused_parameters = gr.Checkbox(
                            label="DDP Find Unused Parameters",
                            value=False,
                            info="Find unused parameters for DistributedDataParallel"
                        )
                        ft_quantize = gr.Checkbox(
                            label="AWQ Quantize Model",
                            value=False,
                            info="Quantize model after training to reduce size"
                        )

                # Hugging Face integration
                with gr.Accordion("Hugging Face Integration", open=True):
                    ft_push_hf = gr.Checkbox(
                        label="Push to Hugging Face",
                        value=False,
                        info="Upload model to Hugging Face Hub after training"
                    )
                    ft_hf_token = gr.Textbox(
                        label="Hugging Face Token",
                        type="password",
                        info="Hugging Face API token (required for pushing to Hub)"
                    )

                with gr.Row():
                    ft_run_btn = gr.Button("Run Fine-tuning", variant="primary")
                    ft_clear_btn = gr.Button("Clear")

                with gr.Row():
                    ft_output = gr.JSON(label="Fine-tuning Result")
                with gr.Row():
                    ft_log_output = gr.Textbox(label="Training Logs", lines=20)

                # Event handlers
                ft_run_btn.click(
                    fn=run_fine_tuning,
                    inputs=[
                        ft_json_file, ft_base_model, ft_system_prompt, ft_repo_id,
                        ft_max_step, ft_learning_rate, ft_epochs, ft_push_hf,
                        ft_hf_token, ft_quantize, ft_batch_size, ft_grad_accum,
                        ft_warmup, ft_logging_steps, ft_optim, ft_weight_decay,
                        ft_scheduler, ft_max_grad_norm, ft_workers,
                        ft_adam_beta1, ft_adam_beta2, ft_adam_epsilon, ft_ddp_find_unused_parameters
                    ],
                    outputs=[ft_output, ft_log_output]
                )

                ft_clear_btn.click(
                    fn=lambda: (
                        None, "Meta-Llama-3.1-8B-Instruct", "", "", 150, 0.0002, 2, False,
                        "", True, 4, 2, 200, 50, "adamw_8bit", 0.01, "cosine", 1.0, 4, True,
                        0.9, 0.999, 1e-8, False, False, False, False, 16, 32, 0.05, "", 0, 0, 2048,
                        False, "{}", None, ""
                    ),
                    inputs=[],
                    outputs=[
                        ft_json_file, ft_base_model, ft_system_prompt, ft_repo_id,
                        ft_max_step, ft_learning_rate, ft_epochs, ft_push_hf,
                        ft_hf_token, ft_quantize, ft_batch_size, ft_grad_accum,
                        ft_warmup, ft_logging_steps, ft_optim, ft_weight_decay,
                        ft_scheduler, ft_max_grad_norm, ft_workers,
                        ft_adam_beta1, ft_adam_beta2, ft_adam_epsilon, ft_ddp_find_unused_parameters,ft_output, ft_log_output
                    ]
                )
            # Inference Tab
            with gr.Tab("Inference"):
                with gr.Row():
                    with gr.Column():
                        inf_base_model = gr.Textbox(label="Base Model Path", value="Meta-Llama-3.1-8B-Instruct")
                        inf_adapter = gr.Textbox(label="Adapter Path")

                        with gr.Row():
                            inf_max_tokens = gr.Slider(label="Max Tokens", minimum=1, maximum=2048, value=300, step=1)
                            inf_temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=0.7, step=0.1)
                            inf_gpu_util = gr.Slider(label="GPU Utilization", minimum=0.1, maximum=1.0, value=0.8, step=0.1)

                        inf_shutdown = gr.Checkbox(label="Shutdown Server After Inference", value=False)
                        inf_prompt = gr.Textbox(label="Prompt", lines=5, placeholder="Enter your prompt here...")

                        with gr.Row():
                            inf_run_btn = gr.Button("Run Inference", variant="primary")
                            inf_clear_btn = gr.Button("Clear")
                            inf_stop_btn = gr.Button("Stop Server", variant="stop")

                with gr.Row():
                    inf_response = gr.Textbox(label="Generated Response", lines=12)
                with gr.Row():
                    inf_output = gr.JSON(label="Inference Result")


                # Event handlers
                def process_inference_result(result):
                    if result.get("status") == "success" and "response" in result:
                        return result, result["response"]
                    return result, "No response generated"

                inf_run_btn.click(
                    fn=lambda *args: process_inference_result(run_inference(*args)),
                    inputs=[inf_base_model, inf_adapter, inf_prompt, inf_max_tokens,
                           inf_temperature, inf_gpu_util, inf_shutdown],
                    outputs=[inf_output, inf_response]
                )

                inf_clear_btn.click(
                    fn=lambda: ("mistralai/Mistral-7B-v0.1", "", "", 300, 0.7, 0.8, False, None, ""),
                    inputs=[],
                    outputs=[inf_base_model, inf_adapter, inf_prompt, inf_max_tokens,
                            inf_temperature, inf_gpu_util, inf_shutdown, inf_output, inf_response]
                )

                inf_stop_btn.click(
                    fn=lambda: {"status": "info", "message": "Server stopped", "server_shutdown": True}
                       if vllm_manager.kill_existing_vllm_process()
                       else {"status": "info", "message": "No server running"},
                    inputs=[],
                    outputs=[inf_output]
                )

    return demo

if __name__ == "__main__":
    # Create and launch the UI
    demo = create_ui()
    demo.launch(share=True)