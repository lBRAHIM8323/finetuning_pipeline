# Fine-Tuning Pipeline Repository

This repository contains a complete pipeline for processing PDF files, fine-tuning a language model, and performing inference using the fine-tuned model. The pipeline is designed to leverage modern computing resources (e.g., CUDA-enabled GPUs) and includes three main scripts: `pdf_processing.py`, `Finetuning_Pipeline.py`, and `VLLM_Inference.py`. Each script is detailed below, including its purpose, dependencies, configuration, and execution instructions.

The current date is **April 07, 2025**, and this README assumes a continuously updated knowledge base with no strict cutoff.

---

## Repository Overview

The pipeline consists of three key stages:
1. **PDF Processing**: Extract data from PDFs and generate Question-Answer (QA) pairs.
2. **Model Fine-Tuning**: Fine-tune a language model using the generated QA pairs.
3. **Inference & Testing**: Run inference with the fine-tuned model and evaluate its performance.

All scripts are designed to work together in a single repository, with configurable parameters passed via JSON inputs.

---

## Scripts

### 1. `pdf_processing.py` - PDF Processing

#### Overview
This script processes PDF files to extract text, tables, and images, then generates QA pairs using the `vLLM` engine. It supports multi-threading and GPU acceleration for efficient processing.

#### Key Features
- **PDF Extraction**: Extracts text and metadata using `PyPDF2`.
- **Tokenization**: Uses Hugging Face's `tiktoken` for text tokenization.
- **QA Generation**: Leverages `vLLM` to generate QA pairs.
- **Multi-threading**: Processes PDF pages in parallel.
- **GPU Utilization**: Optimizes CUDA-enabled GPU usage.
- **Dynamic Resource Management**: Automatically manages server processes.

#### Dependencies
- **Python Libraries**: `PyPDF2`, `tiktoken`, `requests`, `psutil`, `torch`, `concurrent.futures`
- **External Tools**: `vLLM` (pre-installed)

#### System Requirements
- **CPU/GPU**: Multi-threaded CPU and CUDA-enabled GPU
- **Memory**: 16GB+ recommended

#### Configuration Parameters
The script accepts a JSON input with the following parameters:
- `pdf_file` (string): Path to the PDF file.
- `base_model` (string): Model name (e.g., `"Meta-Llama-3.1-8B-Instruct"`).
- `gpu_utilization` (float, default: 0.9): GPU memory usage percentage.
- `num_threads` (int, default: 4-16): Number of threads for parallel processing.
- `max_context_length` (int, default: 8196): Maximum token length.
- `output_dir` (string, default: `"data"`): Output directory for QA pairs.

#### Example JSON Configuration
```json
{
    "pdf_file": "synctalk.pdf",
    "base_model": "Meta-Llama-3.1-8B-Instruct",
    "gpu_utilization": 0.8,
    "num_threads": 4,
    "max_context_length": 8196,
    "output_dir": "data"
}
