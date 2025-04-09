import logging
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, NamedTuple
import re
import pandas as pd
from dataclasses import dataclass
import pdfplumber
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTPage
import fitz
from gmft.pdf_bindings import PyPDFium2Document
from gmft.auto import CroppedTable, AutoTableDetector, AutoTableFormatter
import subprocess

class PageContent(NamedTuple):
    """Structure to hold extracted page content"""
    page_number: int
    text: str
    tables: List[pd.DataFrame]

class PDFContent(NamedTuple):
    """Structure to hold complete PDF content"""
    pages: List[PageContent]
    total_pages: int

def setup_logging():
    """Configure logging with proper formatting and handlers"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("pdf_processing.log", mode='a')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class PDFExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.detector = AutoTableDetector()
        # Increase overlap threshold to handle cases with higher overlap
        self.detector.max_area_ratio_overlap = 1.5  # Increased from default 0.9 (90%)
        self.formatter = AutoTableFormatter()

    def get_cache_filename(self, pdf_path: str) -> str:
        """Generate a cache filename based on the PDF path and modification time."""
        pdf_stat = os.stat(pdf_path)
        pdf_modified_time = pdf_stat.st_mtime
        base_name = Path(pdf_path).stem
        cache_filename = f"{base_name}_{pdf_modified_time}.json"
        cache_dir = "pdf_cache"
        return os.path.join(cache_dir, cache_filename)

    def extract_all_text(self, pdf_doc: fitz.Document) -> Dict[int, str]:
        """Extract all text from PDF in a single pass."""
        text_by_page = {}
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            text_by_page[page_num] = page.get_text()
        return text_by_page

    def get_table_bbox(self, table: CroppedTable) -> tuple:
        """
        Safely get the bounding box from a CroppedTable object.
        Handles both crop_box and alternative attribute names.
        """
        # Try different possible attribute names 
        if hasattr(table, 'crop_box'):
            return table.crop_box
        elif hasattr(table, 'bbox'):
            return table.bbox
        elif hasattr(table, 'box'):
            return table.box
        
        # If no standard attribute found, try to extract from table object
        try:
            # Try to get the table dimensions if they exist as properties
            if all(hasattr(table, attr) for attr in ['x', 'y', 'width', 'height']):
                return (table.x, table.y, table.width, table.height)
        except Exception:
            pass
            
        # Return a default value as fallback
        self.logger.warning(f"Could not determine bounding box for table, using default")
        return (0, 0, 100, 100)  # Default fallback dimensions
        
    def merge_overlapping_tables(self, tables: List[CroppedTable]) -> List[CroppedTable]:
        """Merge tables that have significant overlap"""
        if not tables:
            return []
        
        result = []
        tables_to_process = tables.copy()
        
        while tables_to_process:
            current = tables_to_process.pop(0)
            merged = False
            
            i = 0
            while i < len(tables_to_process):
                try:
                    overlap = self._calculate_overlap(current, tables_to_process[i])
                    if overlap > 0.9:  # 90% overlap threshold for merging
                        # Create a new table with combined boundaries
                        current = self._merge_table_boundaries(current, tables_to_process[i])
                        tables_to_process.pop(i)
                        merged = True
                    else:
                        i += 1
                except Exception as e:
                    self.logger.warning(f"Error calculating overlap: {str(e)}")
                    i += 1
            
            if merged:
                # Put back into processing list if we merged something
                tables_to_process.insert(0, current)
            else:
                result.append(current)
        
        return result

    def _calculate_overlap(self, table1, table2):
        """Calculate overlap ratio between two tables"""
        try:
            # Extract bounding box coordinates
            x1, y1, w1, h1 = self.get_table_bbox(table1)
            x2, y2, w2, h2 = self.get_table_bbox(table2)
            
            # Calculate intersection
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            intersection = x_overlap * y_overlap
            
            # Calculate areas
            area1 = w1 * h1
            area2 = w2 * h2
            
            # Prevent division by zero
            if min(area1, area2) == 0:
                return 0
                
            # Return overlap ratio
            return intersection / min(area1, area2)
        except Exception as e:
            self.logger.warning(f"Error calculating table overlap: {str(e)}")
            return 0  # Assume no overlap if calculation fails

    def _merge_table_boundaries(self, table1, table2):
        """Create a new table with combined boundaries"""
        try:
            x1, y1, w1, h1 = self.get_table_bbox(table1)
            x2, y2, w2, h2 = self.get_table_bbox(table2)
            
            # Calculate new bounding box
            new_x = min(x1, x2)
            new_y = min(y1, y2)
            new_w = max(x1 + w1, x2 + w2) - new_x
            new_h = max(y1 + h1, y2 + h2) - new_y
            
            # Create a new table with merged boundaries
            return CroppedTable(table1.page, (new_x, new_y, new_w, new_h))
        except Exception as e:
            self.logger.warning(f"Error merging table boundaries: {str(e)}")
            return table1  # Return first table if merge fails

    def extract_all_tables(self, pdf_path: str, total_pages: int) -> Dict[int, List[pd.DataFrame]]:
        """Extract all tables from PDF in a single pass."""
        tables_by_page = {}
        doc = None
        try:
            doc = PyPDFium2Document(pdf_path)
            
            for page_num in range(total_pages):
                try:
                    # Try primary extraction method
                    page = doc[page_num]
                    detected_tables = self.detector.extract(page)
                    
                    # Apply merging to handle overlapping tables
                    try:
                        detected_tables = self.merge_overlapping_tables(detected_tables)
                    except Exception as merge_error:
                        self.logger.warning(f"Error merging tables on page {page_num}: {str(merge_error)}")
                    
                    processed_tables = []
                    for table in detected_tables:
                        try:
                            df = self.formatter.extract(table).df()
                            if df is not None and not df.empty:
                                df = self.clean_table_data(df)
                                df = self.clean_column_names(df)
                                processed_tables.append(df)
                        except Exception as table_error:
                            self.logger.warning(f"Error processing table on page {page_num}: {str(table_error)}")
                    
                    tables_by_page[page_num] = processed_tables
                except Exception as e:
                    self.logger.warning(f"Error extracting tables from page {page_num}, trying fallback method: {str(e)}")
                    # Fallback method using pdfplumber
                    tables_by_page[page_num] = self.extract_tables_with_fallback(pdf_path, page_num)
            
            self.logger.info(f"Found {len(doc)} pages in PDF")
            return tables_by_page
        except Exception as e:
            self.logger.error(f"Critical error in table extraction: {str(e)}")
            return {page_num: [] for page_num in range(total_pages)}
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

    def extract_tables_with_fallback(self, pdf_path: str, page_num: int) -> List[pd.DataFrame]:
        """Fallback method for table extraction using pdfplumber"""
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num >= len(pdf.pages):
                    self.logger.warning(f"Page {page_num} out of range in fallback method")
                    return []
                    
                page = pdf.pages[page_num]
                extracted_tables = page.extract_tables()
                for table_data in extracted_tables:
                    if table_data and len(table_data) > 1:  # Need at least headers + one row
                        try:
                            # Use first row as headers
                            headers = [str(col).strip() if col else f"col_{i}" 
                                      for i, col in enumerate(table_data[0])]
                            
                            # Handle missing or empty headers
                            headers = [f"col_{i}" if not h else h 
                                      for i, h in enumerate(headers)]
                            
                            # Create dataframe with remaining rows
                            df = pd.DataFrame(table_data[1:], columns=headers)
                            df = self.clean_table_data(df)
                            df = self.clean_column_names(df)
                            if not df.empty:
                                tables.append(df)
                        except Exception as df_error:
                            self.logger.warning(f"Error creating dataframe in fallback: {str(df_error)}")
        except Exception as e:
            self.logger.error(f"Fallback table extraction failed: {str(e)}")
        
        return tables

    def clean_table_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean table data."""
        if df.empty:
            return df
        
        for col in df.columns:
            try:
                df[col] = df[col].astype(str).map(
                    lambda x: re.sub(r"[^a-zA-Z0-9.,%()/-]", " ", x).strip()
                )
            except Exception as e:
                self.logger.warning(f"Error cleaning column {col}: {str(e)}")
        return df

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and deduplicate column names."""
        if df.empty:
            return df
        
        try:
            df.columns = df.columns.map(lambda x: re.sub(r'[^a-zA-Z0-9_]', '_', str(x)))
            cols = pd.Series(df.columns)
            for dup in cols[cols.duplicated()].unique():
                cols[cols == dup] = [f'{dup}_{i}' for i in range(sum(cols == dup))]
            df.columns = cols
        except Exception as e:
            self.logger.warning(f"Error cleaning column names: {str(e)}")
            # Fallback to generic column names if cleaning fails
            df.columns = [f"col_{i}" for i in range(len(df.columns))]
        
        return df
    
    def extract_pdf_content(self, pdf_path: str) -> PDFContent:
        """Extract all content from PDF in a single pass."""
        pdf_doc = None
        try:
            cache_path = self.get_cache_filename(pdf_path)
            
            # Try to load from cache first
            cached_content = self.load_from_cache(cache_path)
            if cached_content:
                return cached_content

            self.logger.info(f"Extracting content from PDF: {pdf_path}")
            
            # Apply OCR if needed
            pdf_path = self.apply_ocr(pdf_path)
            
            # Open PDF document
            pdf_doc = fitz.open(pdf_path)
            total_pages = len(pdf_doc)
            
            # Extract all text and tables in single passes
            text_content = self.extract_all_text(pdf_doc)
            tables_content = self.extract_all_tables(pdf_path, total_pages)
            
            # Combine into page content objects
            pages = []
            for page_num in range(total_pages):
                page_content = PageContent(
                    page_number=page_num + 1,
                    text=text_content.get(page_num, ""),
                    tables=tables_content.get(page_num, [])
                )
                pages.append(page_content)
            
            pdf_content = PDFContent(pages=pages, total_pages=total_pages)
            
            # Save to cache
            self.save_to_cache(cache_path, pdf_content)
            
            return pdf_content
            
        except Exception as e:
            self.logger.error(f"Error extracting PDF content: {str(e)}")
            raise
        finally:
            if pdf_doc is not None:
                try:
                    pdf_doc.close()
                except Exception:
                    pass

    def apply_ocr(self, pdf_path: str) -> str:
        """Apply OCR to PDF if needed."""
        try:
            # First check if OCR is needed by looking for text in a sample of pages
            needs_ocr = self.check_if_ocr_needed(pdf_path)
            if not needs_ocr:
                self.logger.info(f"PDF already contains text, skipping OCR")
                return pdf_path
                
            # Apply OCR if needed
            ocr_output = Path(pdf_path).with_name(f"ocr_{Path(pdf_path).name}")
            
            # Use more robust OCR settings
            ocr_cmd = [
                "ocrmypdf",
                "--clean",               # Clean before OCR
                "--optimize", "3",       # Optimize PDF
                "--skip-text",           # Skip pages that already have text
                "--output-type", "pdf",  # Ensure PDF output
                "--quiet",               # Reduce console output
                pdf_path,
                str(ocr_output)
            ]
            
            process = subprocess.run(ocr_cmd, capture_output=True, text=True, check=False)
            
            if process.returncode == 0:
                self.logger.info(f"OCR applied successfully: {ocr_output}")
                return str(ocr_output)
            else:
                # Log detailed error
                self.logger.warning(f"OCR failed with code {process.returncode}: {process.stderr}")
                # Try fallback method with simpler options
                return self.apply_fallback_ocr(pdf_path)
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"OCR process error: {e}")
            return self.apply_fallback_ocr(pdf_path)
        except Exception as e:
            self.logger.warning(f"OCR exception: {str(e)}")
            return pdf_path

    def check_if_ocr_needed(self, pdf_path: str) -> bool:
        """Check if OCR is needed by sampling pages for text content"""
        try:
            doc = fitz.open(pdf_path)
            sample_size = min(5, len(doc))  # Sample first 5 pages or all if fewer
            
            total_text = 0
            for i in range(sample_size):
                text = doc[i].get_text().strip()
                total_text += len(text)
            
            doc.close()
            
            # If average text per page is very low, OCR is needed
            avg_text_per_page = total_text / sample_size if sample_size > 0 else 0
            return avg_text_per_page < 100  # Threshold: less than 100 chars per page
        except Exception as e:
            self.logger.warning(f"Error checking if OCR needed: {str(e)}")
            return True  # Default to applying OCR if check fails

    def apply_fallback_ocr(self, pdf_path: str) -> str:
        """Apply OCR with minimal options as fallback"""
        try:
            ocr_output = Path(pdf_path).with_name(f"ocr_simple_{Path(pdf_path).name}")
            
            # Simpler OCR command with fewer options
            ocr_cmd = [
                "ocrmypdf",
                "--skip-text",
                pdf_path,
                str(ocr_output)
            ]
            
            process = subprocess.run(ocr_cmd, capture_output=True, text=True, check=False)
            
            if process.returncode == 0:
                self.logger.info(f"Fallback OCR applied successfully: {ocr_output}")
                return str(ocr_output)
            else:
                self.logger.warning(f"Fallback OCR also failed: {process.stderr}")
                return pdf_path
        except Exception as e:
            self.logger.warning(f"Fallback OCR exception: {str(e)}")
            return pdf_path

    def save_to_cache(self, cache_path: str, content: PDFContent) -> None:
        """Save extracted content to cache."""
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            serializable_content = {
                'total_pages': content.total_pages,
                'pages': [
                    {
                        'page_number': page.page_number,
                        'text': page.text,
                        'tables': [df.to_dict('records') for df in page.tables]
                    }
                    for page in content.pages
                ]
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_content, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Content saved to cache: {cache_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving to cache: {str(e)}")

    def load_from_cache(self, cache_path: str) -> Optional[PDFContent]:
        """Load content from cache if available."""
        try:
            if not os.path.exists(cache_path):
                return None
                
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            pages = []
            for page_data in data['pages']:
                tables = [pd.DataFrame.from_records(table) for table in page_data['tables']]
                page = PageContent(
                    page_number=page_data['page_number'],
                    text=page_data['text'],
                    tables=tables
                )
                pages.append(page)
                
            return PDFContent(pages=pages, total_pages=data['total_pages'])
            
        except Exception as e:
            self.logger.warning(f"Could not load cache: {str(e)}")
            return None