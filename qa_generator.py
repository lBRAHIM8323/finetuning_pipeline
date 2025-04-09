import json
import logging
import re
import requests
import time
import pandas as pd
from typing import List, Dict
from pdf_extractor import PDFContent, setup_logging

logger = setup_logging()

class QAGenerator:
    def __init__(self, token_processor, base_model):
        self.token_processor = token_processor
        self.logger = logger
        self.base_model= base_model

    def _generate_qa_with_prompt(self, prompt_template: str, retries: int = 2) -> List[Dict]:
        available_tokens = self.token_processor.calculate_available_tokens(prompt_template)
        
        for attempt in range(retries):
            try:
                response = requests.post(
                    "http://localhost:8000/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": self.base_model,  # Using the base model set in vLLM server
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt_template}
                        ],
                        "max_tokens": available_tokens
                    },
                    timeout=600
                )
                response.raise_for_status()
                
                response_data = response.json()
                if 'choices' in response_data and response_data['choices']:
                    generated_text = response_data['choices'][0]['message']['content'].strip()
                    try:
                        qa_pairs = json.loads(generated_text)
                        if isinstance(qa_pairs, list):
                            valid_pairs = [
                                pair for pair in qa_pairs 
                                if isinstance(pair, dict) and 
                                "prompt" in pair and 
                                "response" in pair and 
                                pair["prompt"] and 
                                pair["response"]
                            ]
                            if valid_pairs:
                                self.logger.info(f"Generated {len(valid_pairs)} valid QA pairs")
                                return valid_pairs
                    except json.JSONDecodeError:
                        self.logger.warning("Failed to parse JSON response")

            except Exception as e:
                self.logger.error(f"Error in attempt {attempt + 1}: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(5)

        return []

    def generate_qa_from_text(self, text_chunk: str, retries: int = 2) -> List[Dict]:
        if not text_chunk or not text_chunk.strip():
            return []

        prompt_template = f"""You are an AI assistant tasked with generating informative question-answer pairs from text-based documents.

            INPUT CONTEXT:
            {text_chunk}

            TASK:
            Generate well-structured, informative, and in-depth question-answer pairs based on the provided text. Ensure that:
            1. Each question is clear, precise, and designed to extract valuable information.
            2. Each answer is detailed, well-explained, and provides comprehensive insights.
            3. Answers should include supporting details, examples, or elaborations where applicable.
            4. Responses should not introduce external or made-up information but should maximize the depth of the provided content.

            OUTPUT FORMAT REQUIREMENTS:
            1. Respond ONLY with a JSON array
            2. Each object must contain exactly two fields:
            - "prompt": A well-formed question.
            - "response": A detailed, structured, and comprehensive answer.
            3. Include no text outside the JSON array
            4. Follow this exact structure:

            [
                {{
                    "prompt": "What is the significance of XYZ in the given text?,
                    "response": "XYZ is important because... (detailed explanation with context, examples, or reasoning)."
                }},
                {{
                    "prompt": "How does ABC relate to the overall topic?",
                    "response": "ABC connects to the topic as it... (in-depth discussion with logical connections)."
                }}
            ]"""

        return self._generate_qa_with_prompt(prompt_template, retries)

    def table_to_markdown(self, df: pd.DataFrame) -> str:
        try:
            columns = df.columns
            column_headers = []
            for i, col in enumerate(columns):
                col_str = str(col).strip()
                if not col_str or col_str == 'nan' or col_str == '':
                    col_str = f"Column_{i+1}"
                # Escape pipe characters in column names
                col_str = col_str.replace("|", "\\|")
                column_headers.append(col_str)
            
            header = "| " + " | ".join(column_headers) + " |"
            separator = "| " + " | ".join(["---" for _ in columns]) + " |"
            
            rows = []
            for _, row in df.iterrows():
                row_values = []
                for col in columns:
                    cell_value = str(row[col]).strip()
                    # Escape pipe characters and handle multiline content
                    cell_value = cell_value.replace("|", "\\|").replace("\n", "<br>")
                    row_values.append(cell_value)
                rows.append("| " + " | ".join(row_values) + " |")
            
            # Combine all parts into a well-formed markdown table
            markdown_table = header + "\n" + separator + "\n" + "\n".join(rows)
            return markdown_table
        except Exception as e:
            self.logger.warning(f"Error converting table to Markdown: {str(e)}")
            # Fallback to simple string representation
            return df.to_string(index=False)

    def generate_qa_from_table(self, table: pd.DataFrame, retries: int = 2) -> List[Dict]:
        """Generate QA pairs from a table."""
        if table.empty:
            return []
            
        # Convert table to string representation that preserves structure
        table_str = table.to_string(index=False)
        
        prompt_template = f"""You are an AI assistant tasked with generating informative question-answer pairs from tabular data.

            INPUT TABLE:
            {table_str}

            TASK:
            Generate well-structured, informative, and in-depth question-answer pairs based on the provided table. Ensure that:
            1. Each question focuses on extracting insights, patterns, or notable information from the table.
            2. Questions should address key metrics, trends, comparisons or significant values in the data.
            3. Each answer should be detailed, precise, and reference specific values from the table.
            4. Responses should not introduce external information but focus on analyzing the table data.
            5. Make sure to maintain accuracy when referencing numeric values or relationships in the table.

            OUTPUT FORMAT REQUIREMENTS:
            1. Respond ONLY with a JSON array
            2. Each object must contain exactly two fields:
            - "prompt": A well-formed question about the table data.
            - "response": A detailed, structured, and data-focused answer.
            3. Include no text outside the JSON array
            4. Follow this exact structure:

            [
                {{"prompt": "What are the  values in the table and what do they represent?","response": "The values in the table are X in column Y, representing... (detailed explanation with specific data points)."}}
            ]"""
            
        return self._generate_qa_with_prompt(prompt_template, retries)
        
    def _generate_table_qa_pairs(self, table: pd.DataFrame, table_title: str, page_num: int, table_idx: int) -> List[Dict]:
        qa_pairs = []
        
        try:
            md_table = self.table_to_markdown(table)
            
            # Table overview prompt
            table_overview_qa = {
                'prompt': f"What is the content of {table_title} table on page {page_num}?",
                'response': f"Content of {table_title} on page {page_num}:\n\n{md_table}",
                'source': f'Page {page_num}'
            }
            qa_pairs.append(table_overview_qa)
            
            table_qa = self.generate_qa_from_table(table)
            for qa in table_qa:
                qa['response'] = f"{qa['response']}-Table ({table_title}),-Page {page_num}"
                qa_pairs.append(qa)
            
        except Exception as e:
            self.logger.warning(f"Error generating table QA pairs: {str(e)}")
            error_qa = {
                'prompt': f"What is the content of table on page {page_num}?",
                'response': f"Error processing table: {str(e)}",
                'source': f'Page {page_num}'
            }
            qa_pairs.append(error_qa)
        
        return qa_pairs

    def _extract_table_context(self, page_text: str, table_idx: int, total_tables: int) -> str:
        try:
            # Split page text into lines
            lines = page_text.split('\n')
            context_window = 5
            estimated_table_line = int(len(lines) * (table_idx + 1) / (total_tables + 1))
            
            # Extract context lines
            start_line = max(0, estimated_table_line - context_window)
            end_line = min(len(lines), estimated_table_line + context_window)
            
            context_lines = lines[start_line:end_line]
            return '\n'.join(context_lines)
        
        except Exception as e:
            self.logger.warning(f"Error extracting table context: {str(e)}")
            return ""

    def _extract_table_title_with_llm(self, context: str, table: pd.DataFrame) -> str:
        try:
            # Prepare a prompt for title extraction
            prompt_template = f"""Given the following contextual text and table data, 
            please extract or infer the most likely title for this table:

            CONTEXT:
            {context}

            TABLE PREVIEW:
            {table.head().to_string()}

            Instructions:
            1. Identify the most probable title based on context and table preview
            2. Provide a concise, descriptive title
            3. If no clear title is found, return "Untitled Table"
            4. Respond ONLY with the table title
            5. Do not modify the table title in any way return as it is in context 
            """
            
            response = requests.post(
                "http://localhost:8000/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.base_model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant at extracting table titles."},
                        {"role": "user", "content": prompt_template}
                    ],
                    "max_tokens": 50,  # Limit title length
                    "temperature": 0.3  # Low randomness for consistency
                },
                timeout=30
            )
            response.raise_for_status()
            response_data = response.json()
            
            if 'choices' in response_data and response_data['choices']:
                title = response_data['choices'][0]['message']['content'].strip()

                if not title or title.lower() in ['untitled table', 'none']:
                    return "Untitled Table"
                
                return title
            
        except Exception as e:
            self.logger.warning(f"LLM title extraction failed: {str(e)}")
        return None


    def process_page_content(self, page_content, page_num: int) -> List[Dict]:
        qa_pairs = []
        
        # Process tables
        if page_content.tables and len(page_content.tables) > 0:
            # Perform a first pass to detect table titles and contexts
            table_contexts = []
            for table_idx, table in enumerate(page_content.tables):
                if table.empty:
                    continue
                table_context = self._extract_table_context(page_content.text, table_idx, len(page_content.tables))
                
                table_contexts.append({
                    'table': table,
                    'index': table_idx,
                    'context': table_context
                })
            
            # Process each table with its context
            for table_info in table_contexts:
                table = table_info['table']
                table_idx = table_info['index']
                table_context = table_info['context']
                
                # Detect table title using LLM
                table_title = self._extract_table_title_with_llm(page_content.text, table)
                table_qa_pairs = self._generate_table_qa_pairs(
                    table, 
                    table_title, 
                    page_num, 
                    table_idx
                )
                
                qa_pairs.extend(table_qa_pairs)

        if page_content.tables and len(page_content.tables) > 0:
            cleaned_page_text = self._remove_table_content_from_text(
                page_content.text, 
                page_content.tables
            )
        else:
            cleaned_page_text = page_content.text
            
        # Only generate QA pairs if there's meaningful text content
        if cleaned_page_text and cleaned_page_text.strip():
            # Pass the cleaned text instead of the original text
            text_qa = self.generate_qa_from_text(cleaned_page_text)
            for qa in text_qa:
                qa['response'] = f"{qa['response']}-Page {page_num}"
                qa_pairs.append(qa)
                
        return qa_pairs

    def remove_duplicate_qa_pairs(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Remove duplicate QA pairs based on prompt similarity."""
        if not qa_pairs:
            return []
            
        unique_pairs = []
        prompts = set()
        
        for qa in qa_pairs:
            prompt = qa.get("prompt", "").strip().lower()
            if prompt and prompt not in prompts:
                prompts.add(prompt)
                unique_pairs.append(qa)
                
        return unique_pairs


    def _remove_table_content_from_text(self, page_text: str, tables: List[pd.DataFrame]) -> str:
        try:
            # Filter out empty tables
            tables = [table for table in tables if not table.empty]
            if not tables:
                return page_text
                
            # Convert tables to string representations
            table_texts = []
            for table in tables:
                table_str = table.to_string(index=False)
                table_lines = table_str.strip().split('\n')
                if len(table_lines) > 1:
                    header_pattern = re.escape(table_lines[0].strip())
                    
                    data_patterns = []
                    if len(table_lines) > 2:  
                        data_patterns.append(re.escape(table_lines[1].strip()))
                    if len(table_lines) > 3: 
                        data_patterns.append(re.escape(table_lines[-1].strip()))
                    
                    table_texts.append(header_pattern)
                    table_texts.extend(data_patterns)
            if table_texts:
                cleaned_text = page_text
                for pattern in table_texts:
                    try:
                        cleaned_text = re.sub(f"(?:.{{0,100}}{pattern}.{{0,100}})", " ", cleaned_text, 
                                            flags=re.MULTILINE | re.DOTALL)
                    except Exception as e:
                        self.logger.warning(f"Error removing specific table pattern: {str(e)}")
                
                cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
                return cleaned_text.strip()
        except Exception as e:
            self.logger.warning(f"Error removing table content from text: {str(e)}")
            return page_text