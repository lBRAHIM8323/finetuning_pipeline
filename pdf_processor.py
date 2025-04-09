
# pdf_processor.py

import json
import os
import sys
import time
import gc
import math
import re
import psutil
import torch
import requests
import subprocess
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from pdf_extractor import PDFExtractor, PDFContent, setup_logging
from token_processor import TokenProcessor
from qa_generator import QAGenerator

logger = setup_logging()

class ProcessorConfig:
    """Configuration manager for PDF processing."""
    
    def __init__(self):
        self.pdf_file: Optional[str] = None
        self.base_model: Optional[str] = None  # This will already include "base_model/" prefix
        self.gpu_utilization: float = 0.9
        self.num_threads: int = max(4, min(os.cpu_count(), 16))
        self.max_context_length: int = 8196
        self.output_dir: str = "data"
        
    def validate(self) -> bool:
        """Validate the configuration."""
        if not self.pdf_file:
            logger.error("PDF file path is required")
            return False
        
        if not os.path.exists(self.pdf_file):
            logger.error(f"PDF file not found: {self.pdf_file}")
            return False
            
        if self.base_model and not os.path.exists(self.base_model):
            logger.error(f"Model not found: {self.base_model}")
            return False
                
        if not 0.0 <= self.gpu_utilization <= 1.0:
            logger.error(f"GPU utilization must be between 0.0 and 1.0, got {self.gpu_utilization}")
            return False
            
        if self.num_threads <= 0:
            logger.error(f"Thread count must be positive, got {self.num_threads}")
            return False
            
        return True
            
    def from_json(self, json_str: str) -> bool:
        """Load configuration from JSON string."""
        try:
            input_data = json.loads(json_str)
            
            # Required field
            if 'pdf_file' not in input_data:
                logger.error("Missing required field: pdf_file")
                return False
            self.pdf_file = input_data['pdf_file']
            
            # Base model - it already includes the base_model/ prefix
            if 'base_model' in input_data:
                self.base_model = f"base_model/{input_data['base_model']}"
                
            # Optional fields
            if 'gpu_utilization' in input_data:
                try:
                    gpu_util = float(input_data['gpu_utilization'])
                    if 0.0 <= gpu_util <= 1.0:
                        self.gpu_utilization = gpu_util
                    else:
                        logger.warning(
                            f"Invalid GPU utilization value: {gpu_util}. "
                            f"Must be between 0.0 and 1.0. Using default: {self.gpu_utilization}"
                        )
                except ValueError:
                    logger.warning(
                        f"Invalid GPU utilization format: {input_data['gpu_utilization']}. "
                        f"Using default: {self.gpu_utilization}"
                    )
                    
            if 'num_threads' in input_data:
                try:
                    threads = int(input_data['num_threads'])
                    if threads > 0:
                        self.num_threads = threads
                    else:
                        logger.warning(
                            f"Invalid thread count: {threads}. "
                            f"Must be greater than 0. Using default: {self.num_threads}"
                        )
                except ValueError:
                    logger.warning(
                        f"Invalid thread count format: {input_data['num_threads']}. "
                        f"Using default: {self.num_threads}"
                    )
                    
            if 'output_dir' in input_data:
                self.output_dir = input_data['output_dir']
                
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {str(e)}")
            return False
    
    def get_output_path(self) -> str:
        """Generate output file path based on input PDF."""
        if not self.pdf_file:
            raise ValueError("PDF file not set")
            
        pdf_name = Path(self.pdf_file).stem
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, f"{pdf_name}_qa.json")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        # Strip the 'base_model/' prefix for user-facing output
        display_base_model = None
        if self.base_model and self.base_model.startswith("base_model/"):
            display_base_model = self.base_model[len("base_model/"):]
            
        return {
            "pdf_file": self.pdf_file,
            "base_model": display_base_model,
            "gpu_utilization": self.gpu_utilization,
            "num_threads": self.num_threads,
            "max_context_length": self.max_context_length,
            "output_dir": self.output_dir,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }


class ServerManager:
    """Manages vLLM server lifecycle."""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.server_process = None
        
    def calculate_max_batched_tokens(self, avg_tokens_per_page: int = 1000) -> int:
        """Calculate optimal token batch size based on available resources."""
        # Default if calculations fail
        DEFAULT_TOKEN_BATCH_SIZE = 16384
        
        # Get available GPU memory in GB
        available_gpu_memory = 0
        try:
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                available_memory = total_memory * self.config.gpu_utilization
                available_gpu_memory = available_memory
                logger.info(f"Available GPU memory: {available_gpu_memory:.2f} GB")
            else:
                logger.warning("CUDA not available, using default token batch size")
                return DEFAULT_TOKEN_BATCH_SIZE
        except Exception as e:
            logger.warning(f"Error getting GPU memory: {str(e)}")
            return DEFAULT_TOKEN_BATCH_SIZE
            
        # Constants for memory calculations
        bytes_per_token = 4 * 2  
        
        # Calculate based on available memory
        concurrent_requests = self.config.num_threads * 2
        available_memory_bytes = available_gpu_memory * (1024**3)
        safety_factor = 0.8
        memory_based_tokens = int((available_memory_bytes * safety_factor) / (bytes_per_token * concurrent_requests))
        
        # Apply constraints and scaling
        max_tokens = min(memory_based_tokens, self.config.max_context_length)
        scaling_factor = min(2.0, max(0.5, avg_tokens_per_page / 1000))
        scaled_tokens = int(max_tokens * scaling_factor)
        
        # Ensure reasonable bounds
        min_tokens = 4096
        max_tokens = 32768
        final_tokens = max(min_tokens, min(scaled_tokens, max_tokens))
        
        # Round to nearest multiple of 1024 for efficiency
        final_tokens = (final_tokens // 1024) * 1024
        
        logger.info(f"Dynamically calculated max_num_batched_tokens: {final_tokens}")
        return final_tokens
        
    def wait_for_server(self, max_attempts: int = 150) -> bool:
        """Wait for vLLM server to be ready."""
        url = "http://localhost:8000/health"
        for attempt in range(max_attempts):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info("vLLM server is ready!")
                    return True
            except requests.exceptions.RequestException:
                time.sleep(2)
        logger.error("vLLM server failed to start")
        return False
        
    def start_server(self, avg_tokens_per_page: int = 1000, max_retries: int = 5) -> bool:
        """Start the vLLM server."""
        if not self.config.base_model:
            logger.error("No model specified for vLLM server")
            return False
            
        max_batched_tokens = self.calculate_max_batched_tokens(avg_tokens_per_page)
        
        cmd = [
            "vllm",
            "serve",
            self.config.base_model,  # Already includes base_model/ prefix
            f"--gpu_memory_utilization={self.config.gpu_utilization}",
            "--max_model_len=8196",
            "--enable-chunked-prefill",
            f"--max_num_batched_tokens={max_batched_tokens}",
            f"--max_num_seqs={self.config.num_threads * 2}"
        ]

        for attempt in range(max_retries):
            logger.info(f"Starting vLLM server: {' '.join(cmd)} (Attempt {attempt + 1})")
            try:
                server_process = subprocess.Popen(cmd)
                if self.wait_for_server():
                    self.server_process = server_process
                    return True
            except Exception as e:
                logger.error(f"Error starting server: {str(e)}")
                if server_process:
                    self._terminate_process(server_process)
                time.sleep(5)

        logger.error("Server failed to start after multiple attempts")
        return False
        
    def _terminate_process(self, process: subprocess.Popen) -> None:
        """Safely terminate a process."""
        try:
            process.terminate()
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"Error terminating process: {str(e)}")
            try:
                process.kill()
            except Exception:
                pass
                
    def stop_server(self) -> None:
        """Stop the vLLM server."""
        if self.server_process:
            self._terminate_process(self.server_process)
            self.server_process = None
            
    def kill_existing_vllm_processes(self) -> None:
        """Kill any existing vLLM processes."""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if 'vllm' in proc.info['name'].lower():
                    self._terminate_process(proc)
                    logger.info(f"Terminated vLLM process: {proc.info['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
                logger.error(f"Error killing process: {str(e)}")


class PageProcessor:
    """Processes PDF pages and generates QA pairs."""
    
    def __init__(self, token_processor: TokenProcessor, qa_generator: QAGenerator, num_threads: int):
        self.token_processor = token_processor
        self.qa_generator = qa_generator
        self.num_threads = num_threads
        
    def analyze_token_statistics(self, pdf_content: List[PDFContent]) -> Dict[str, Any]:
        """Analyze token statistics for the PDF content."""
        return self.token_processor.analyze_token_statistics(pdf_content)
        
    def partition_pages(self, pages: List[PDFContent]) -> List[List[Tuple[PDFContent, int]]]:
        """Divide pages into balanced partitions for parallel processing."""
        total_pages = len(pages)
        partition_size = math.ceil(total_pages / self.num_threads)
        
        batches = []
        for i in range(0, total_pages, partition_size):
            batch = [(pages[j], pages[j].page_number) for j in range(i, min(i + partition_size, total_pages))]
            batches.append(batch)
            
        # Log batch distribution
        logger.info(f"Divided {total_pages} pages into {len(batches)} batches")
        for i, batch in enumerate(batches):
            page_nums = [page_num for _, page_num in batch]
            logger.info(f"Batch {i+1}: Processing {len(batch)} pages - {page_nums}")
            
        return batches
        
    def process_page_batch(self, pages_batch: List[Tuple[PDFContent, int]]) -> List[Dict]:
        """Process a batch of pages to generate QA pairs."""
        batch_qa_pairs = []
        
        for page_content, page_num in pages_batch:
            try:
                logger.info(f"Processing page {page_num} in batch")
                page_qa_pairs = self.qa_generator.process_page_content(page_content, page_num)
                if page_qa_pairs:
                    batch_qa_pairs.extend(page_qa_pairs)
                else:
                    logger.warning(f"No QA pairs generated for page {page_num}")
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {str(e)}")
                
        return batch_qa_pairs
        
    def process_pages_parallel(self, pdf_content_pages: List[PDFContent]) -> List[Dict]:
        """Process PDF pages in parallel and generate QA pairs."""
        # Prepare page batches
        page_batches = self.partition_pages(pdf_content_pages)
        all_qa_pairs = []
        
        # Process batches in parallel
        logger.info(f"Processing page batches in parallel with {self.num_threads} threads")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all batches for processing
            future_to_batch = {
                executor.submit(self.process_page_batch, batch): i 
                for i, batch in enumerate(page_batches)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_index = future_to_batch[future]
                try:
                    batch_qa_pairs = future.result()
                    if batch_qa_pairs:
                        logger.info(f"Batch {batch_index+1} completed with {len(batch_qa_pairs)} QA pairs")
                        all_qa_pairs.extend(batch_qa_pairs)
                    else:
                        logger.warning(f"Batch {batch_index+1} produced no QA pairs")
                except Exception as e:
                    logger.error(f"Error processing batch {batch_index+1}: {str(e)}")
                    
        return all_qa_pairs
        
    def sort_qa_pairs(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Sort QA pairs by page number."""
        def get_page_number(qa_pair):
            # Try to extract page number from source field
            if qa_pair.get('source') and (match := re.search(r'Page (\d+)', qa_pair.get('source', ''))):
                return int(match.group(1))
            # Try to extract page number from response field
            elif qa_pair.get('response') and (match := re.search(r'-Page (\d+)$', qa_pair.get('response', ''))):
                return int(match.group(1))
            # Default to 0 if no page number found
            else:
                return 0
                
        return sorted(qa_pairs, key=get_page_number)


class EnhancedPDFProcessor:
    """Main class for processing PDFs, extracting content, and generating QA pairs."""
    
    def __init__(self):
        self.config = ProcessorConfig()
        self.pdf_extractor = PDFExtractor()
        self.token_processor = TokenProcessor()
        self.qa_generator = None
        self.server_manager = None
        self.page_processor = None
        
    def initialize_components(self) -> bool:
        """Initialize components based on configuration."""
        if not self.config.base_model:
            logger.error("No base model specified")
            return False
            
        try:
            self.token_processor.set_model_path(self.config.base_model)
            self.qa_generator = QAGenerator(self.token_processor, self.config.base_model)
            self.server_manager = ServerManager(self.config)
            self.page_processor = PageProcessor(
                self.token_processor,
                self.qa_generator,
                self.config.num_threads
            )
            return True
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            return False
        
    def process_input(self, args: List[str]) -> bool:
        """Process command line arguments."""
        if len(args) < 2:
            logger.error("No input JSON provided")
            return False
            
        try:
            if not self.config.from_json(args[1]):
                return False
                
            if not self.config.validate():
                return False
                
            return self.initialize_components()
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            return False
            
    def process_pdf(self) -> Dict[str, Any]:
        """Process PDF and generate QA pairs."""
        start_time = time.time()
        
        try:
            # Extract content
            logger.info(f"Extracting content from PDF: {self.config.pdf_file}")
            pdf_content = self.pdf_extractor.extract_pdf_content(self.config.pdf_file)
            
            if not pdf_content or not pdf_content.pages:
                return {'status': 'error', 'message': 'No content extracted from PDF'}
                
            # Analyze token statistics
            logger.info("Analyzing token statistics")
            token_stats = self.page_processor.analyze_token_statistics(pdf_content.pages)
            
            # Start the server
            logger.info("Starting vLLM server")
            self.server_manager.kill_existing_vllm_processes()
            avg_tokens_per_page = token_stats.get("avg_tokens_per_page", 1000)
            server_started = self.server_manager.start_server(avg_tokens_per_page)
            
            if not server_started:
                return {'status': 'error', 'message': 'Failed to start vLLM server'}
                
            # Process PDF pages
            logger.info("Processing PDF pages")
            all_qa_pairs = self.page_processor.process_pages_parallel(pdf_content.pages)
            
            # Post-process QA pairs
            logger.info("Removing duplicate QA pairs")
            unique_qa_pairs = self.qa_generator.remove_duplicate_qa_pairs(all_qa_pairs)
            
            if not unique_qa_pairs:
                return {'status': 'error', 'message': 'No QA pairs generated'}
                
            # Prepare output path
            output_path = self.config.get_output_path()
            
            # Sort QA pairs by page number
            sorted_qa_pairs = self.page_processor.sort_qa_pairs(unique_qa_pairs)
            
            # Save results
            logger.info(f"Saving results to {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_qa_pairs, f, ensure_ascii=False, indent=4)
                
            processing_time = time.time() - start_time
            logger.info(f"Generated {len(sorted_qa_pairs)} unique QA pairs from {len(pdf_content.pages)} pages in {processing_time:.2f} seconds")
            
            return {
                'status': 'success', 
                'output_file': output_path, 
                'qa_pairs_count': len(sorted_qa_pairs),
                'processing_time': processing_time,
                'config': self.config.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {'status': 'error', 'message': str(e)}
        finally:
            if self.server_manager:
                self.server_manager.stop_server()

    def cleanup(self):
        """Clean up resources."""
        if self.server_manager:
            self.server_manager.kill_existing_vllm_processes()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Main entry point for the application."""
    start_time = time.time()
    processor = None
    
    try:
        # Initialize processor
        processor = EnhancedPDFProcessor()
        
        # Process input arguments
        if not processor.process_input(sys.argv):
            sys.exit(1)
            
        # Process PDF
        result = processor.process_pdf()
        
        # Handle result
        if result['status'] == 'success':
            total_time = time.time() - start_time
            logger.info(f"Processing completed in {total_time:.2f} seconds. Output: {result['output_file']} with {result.get('qa_pairs_count', 0)} QA pairs")
            print(json.dumps(result, indent=2))
            sys.exit(0)
        else:
            logger.error(f"Processing failed: {result['message']}")
            print(json.dumps(result, indent=2))
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(json.dumps({'status': 'error', 'message': str(e)}, indent=2))
        sys.exit(1)
    finally:
        # Clean up resources
        if processor:
            processor.cleanup()

if __name__ == "__main__":
    main()