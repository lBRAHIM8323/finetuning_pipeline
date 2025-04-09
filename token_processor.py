import os
import logging
from typing import List, Dict
from transformers import AutoTokenizer
from pdf_extractor import PDFContent, setup_logging

logger = setup_logging()

class TokenProcessor:
    def __init__(self, base_model: str = None):
        self.setup_tokenizer(base_model)
        self.logger = logger
        self.system_message_tokens = len(self.tokenizer.encode("You are a helpful assistant."))
        self.max_context_length = 8196

    def setup_tokenizer(self, base_model):
        if base_model:
            model_path = base_model
            if os.path.exists(model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")
        else:
            available_models = os.listdir("base_model")
            if available_models:
                default_model = os.path.join("base_model", available_models[0])
                self.tokenizer = AutoTokenizer.from_pretrained(
                    default_model,
                    local_files_only=True,
                    trust_remote_code=True
                )
            else:
                raise ValueError("No models found")

    def set_model_path(self, base_model: str):
        """Update the model path and reinitialize the tokenizer."""
        self.setup_tokenizer(base_model)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        return len(self.tokenizer.encode(text))

    def calculate_available_tokens(self, prompt_template: str) -> int:
        """Calculate the number of tokens available for response."""
        template_tokens = self.count_tokens(prompt_template)
        reserved_tokens = self.system_message_tokens + 100
        return self.max_context_length - template_tokens - reserved_tokens

    def analyze_token_statistics(self, pdf_content: List[PDFContent]) -> Dict:
        """Analyze token statistics for PDF content."""
        if not pdf_content:
            return {"avg_tokens_per_page": 1000}  # Default fallback
            
        # Take a representative sample of pages (up to 10 pages)
        sample_size = min(10, len(pdf_content))
        sample_indices = [int(i * len(pdf_content) / sample_size) for i in range(sample_size)]
        sample_pages = [pdf_content[i] for i in sample_indices]
        
        # Analyze token counts for the sample
        token_counts = []
        for page in sample_pages:
            if hasattr(page, 'text') and page.text:
                token_count = self.count_tokens(page.text)
                token_counts.append(token_count)
        
        if not token_counts:
            return {"avg_tokens_per_page": 1000}  # Default fallback
            
        # Calculate statistics
        avg_tokens = sum(token_counts) / len(token_counts)
        max_tokens = max(token_counts)
        min_tokens = min(token_counts)
        median_tokens = sorted(token_counts)[len(token_counts) // 2]
        
        self.logger.info(f"Token statistics - Avg: {avg_tokens:.1f}, Min: {min_tokens}, Max: {max_tokens}, Median: {median_tokens}")
        
        return {
            "avg_tokens_per_page": avg_tokens,
            "max_tokens_per_page": max_tokens,
            "min_tokens_per_page": min_tokens,
            "median_tokens_per_page": median_tokens
        }
        
    def split_chunk_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """Split text into chunks that fit within token limit."""
        import re
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk += sentence + " "
                current_tokens += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                current_tokens = sentence_tokens

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks