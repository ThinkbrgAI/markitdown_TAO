# type: ignore
import base64
import binascii
import copy
import html
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import warnings
from typing import Any, Dict, List, Optional, Union
from urllib.parse import parse_qs, quote, unquote, urlparse, urlunparse
from enum import Enum
import ast

# Suppress pydub warning about ffmpeg
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub.utils")

# Core imports
import mammoth
import markdownify
import pandas as pd
import pdfminer
import pdfminer.high_level
import pdfplumber
import pptx
import puremagic
import requests
from bs4 import BeautifulSoup

# PDF processing
import camelot
import tabula
from pdf2image import convert_from_path

# Optional imports with fallbacks
try:
    import extract_msg
    IS_OUTLOOK_CAPABLE = True
except ModuleNotFoundError:
    IS_OUTLOOK_CAPABLE = False

try:
    from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor
    import torch
    HAS_LAYOUTLM = True
except ImportError:
    HAS_LAYOUTLM = False
    warnings.warn("LayoutLMv3 not available. Install with 'pip install transformers torch'")

class TableExtractionStrategy(Enum):
    """Strategies for table extraction from documents"""
    HEURISTIC = "heuristic"  # Default strategy
    LAYOUTLM = "layoutlm"    # Uses LayoutLMv3
    GPT4V = "gpt4v"         # Uses GPT-4V
    AUTO = "auto"           # Tries heuristic first, falls back to ML if needed

class FileConversionException(Exception):
    """Exception raised for errors during file conversion."""
    pass

class UnsupportedFormatException(Exception):
    """Exception raised for unsupported file formats."""
    pass

class ConversionResult:
    """Container for conversion results"""
    def __init__(self, text_content: str = "", metadata: Dict = None, images: List[str] = None):
        self.text_content = text_content
        self.metadata = metadata or {}
        self.images = images or []

    def __str__(self):
        return f"ConversionResult(text_length={len(self.text_content)}, images={len(self.images)})"

class TableCell:
    """Represents a cell in a table with merge and style information"""
    def __init__(self, content: str, rowspan: int = 1, colspan: int = 1):
        self.content = content
        self.rowspan = rowspan
        self.colspan = colspan
        self.is_merged_cell = False
        self.styles = []  # List of styles: 'bold', 'italic', 'code', etc.
        
        # Detect and strip markdown styling
        self._detect_styles()
    
    def _detect_styles(self):
        """Detect markdown styling in content"""
        # Bold detection (both ** and __)
        if (self.content.startswith('**') and self.content.endswith('**')) or \
           (self.content.startswith('__') and self.content.endswith('__')):
            self.styles.append('bold')
            self.content = self.content[2:-2]
        
        # Italic detection (both * and _)
        elif (self.content.startswith('*') and self.content.endswith('*')) or \
             (self.content.startswith('_') and self.content.endswith('_')):
            self.styles.append('italic')
            self.content = self.content[1:-1]
        
        # Code detection
        elif self.content.startswith('`') and self.content.endswith('`'):
            self.styles.append('code')
            self.content = self.content[1:-1]
        
        # Mixed bold-italic
        elif (self.content.startswith('***') and self.content.endswith('***')) or \
             (self.content.startswith('___') and self.content.endswith('___')):
            self.styles.extend(['bold', 'italic'])
            self.content = self.content[3:-3]

    def apply_styles(self, content: str) -> str:
        """Apply stored styles to content"""
        styled_content = content
        
        # Apply styles in reverse order (inside out)
        for style in reversed(self.styles):
            if style == 'bold':
                styled_content = f"**{styled_content}**"
            elif style == 'italic':
                styled_content = f"_{styled_content}_"
            elif style == 'code':
                styled_content = f"`{styled_content}`"
        
        return styled_content

class PdfTableExtractor:
    """Advanced PDF table extraction using multiple strategies"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.last_cost = 0.0
        self.seen_tables = set()
        self.strategies = [
            self._extract_with_camelot,
            self._extract_with_tabula,
            self._extract_with_pdfplumber,
            self._extract_with_layoutlm,
            self._extract_with_gpt4v,
            self._extract_with_heuristics
        ]

    def reset_state(self):
        """Reset state for new document"""
        self.seen_tables.clear()
        self.last_cost = 0.0

    def extract_tables(self, pdf_path: str) -> List[List[List[str]]]:
        """
        Extract tables using multiple strategies and combine results
        Returns list of tables, where each table is a list of rows
        """
        all_tables = []
        tables_confidence = {}  # Store confidence scores for each table

        # Try each strategy
        for strategy in self.strategies:
            try:
                tables = strategy(pdf_path)
                for table in tables:
                    confidence = self._calculate_table_confidence(table)
                    table_hash = self._hash_table(table)
                    if table_hash not in tables_confidence or confidence > tables_confidence[table_hash][0]:
                        tables_confidence[table_hash] = (confidence, table)
            except Exception as e:
                print(f"Strategy {strategy.__name__} failed: {str(e)}", file=sys.stderr)

        # Return unique tables with highest confidence
        return [table for _, table in sorted(tables_confidence.values(), reverse=True)]

    def _extract_with_camelot(self, pdf_path: str) -> List[List[List[str]]]:
        """Extract tables using Camelot - best for bordered tables"""
        tables = []
        try:
            # Check if PDF is image-based
            with pdfplumber.open(pdf_path) as pdf:
                if not pdf.pages[0].extract_text().strip():
                    print("PDF appears to be image-based, skipping Camelot", file=sys.stderr)
                    return []

            # Try lattice mode
            camelot_tables = camelot.read_pdf(
                pdf_path, 
                flavor='lattice',
                flag_size=True,
                line_scale=40
            )
            for table in camelot_tables:
                if table.parsing_report['accuracy'] > 80:
                    tables.append(table.data)
                    
            # Try stream mode if no tables found
            if not tables:
                camelot_tables = camelot.read_pdf(
                    pdf_path, 
                    flavor='stream',
                    flag_size=True
                )
                for table in camelot_tables:
                    if table.parsing_report['accuracy'] > 80:
                        tables.append(table.data)
                    
        except Exception as e:
            print(f"Camelot extraction failed: {str(e)}", file=sys.stderr)
        return tables

    def _extract_with_tabula(self, pdf_path: str) -> List[List[List[str]]]:
        """Extract tables using Tabula - good for spreadsheet-like tables"""
        tables = []
        try:
            # Use lattice mode
            df_list = tabula.read_pdf(
                pdf_path,
                pages='all',
                multiple_tables=True,
                lattice=True  # Changed from method='lattice'
            )
            if df_list:
                for df in df_list:
                    if not df.empty:
                        tables.append(df.values.tolist())
                    
            # Try stream mode
            df_list = tabula.read_pdf(
                pdf_path,
                pages='all',
                multiple_tables=True,
                stream=True  # Changed from method='stream'
            )
            if df_list:
                for df in df_list:
                    if not df.empty:
                        tables.append(df.values.tolist())
                    
        except Exception as e:
            print(f"Tabula extraction failed: {str(e)}", file=sys.stderr)
        return tables

    def _extract_with_pdfplumber(self, pdf_path: str) -> List[List[List[str]]]:
        """Extract tables using pdfplumber - good for simple tables"""
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table and any(any(cell for cell in row) for row in table):
                            # Clean empty cells and normalize
                            cleaned_table = [
                                [str(cell).strip() if cell else '' for cell in row]
                                for row in table
                            ]
                            tables.append(cleaned_table)
        except Exception as e:
            print(f"pdfplumber extraction failed: {str(e)}", file=sys.stderr)
        return tables

    def _extract_with_layoutlm(self, pdf_path: str) -> List[List[List[str]]]:
        """Extract tables using LayoutLM - good for complex layouts"""
        if not HAS_LAYOUTLM:
            return []
            
        try:
            # Check for Poppler in common locations
            poppler_paths = [
                r"C:\Program Files\poppler\Library\bin",
                r"C:\Program Files\poppler\bin",
                os.path.join(os.environ.get("PROGRAMFILES", ""), "poppler", "Library", "bin"),
                os.path.join(os.environ.get("PROGRAMFILES", ""), "poppler", "bin")
            ]
            
            # Add Poppler to PATH if found
            for path in poppler_paths:
                if os.path.exists(path):
                    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                    break
            else:
                print("Poppler not found in common locations", file=sys.stderr)
                return []
            
            # Convert PDF pages to images
            images = convert_from_path(pdf_path)
            
            for image in images:
                # Process with LayoutLM
                # This is a placeholder - implement actual LayoutLM processing
                pass
                
        except Exception as e:
            print(f"LayoutLM extraction failed: {str(e)}", file=sys.stderr)
        return []

    def _extract_with_heuristics(self, pdf_path: str) -> List[List[List[str]]]:
        """Extract tables using text-based heuristics - fallback method"""
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        # Use existing heuristic detection
                        current_table = []
                        for line in text.split('\n'):
                            cells = self._detect_table_row(line)
                            if cells:
                                current_table.append(cells)
                            elif current_table:
                                if len(current_table) >= 2:
                                    tables.append(current_table)
                                current_table = []
                        
                        if current_table and len(current_table) >= 2:
                            tables.append(current_table)
        except Exception as e:
            print(f"Heuristic extraction failed: {str(e)}", file=sys.stderr)
        return tables

    def _detect_table_row(self, line: str) -> Optional[List[str]]:
        """Detect if a line contains table cells"""
        # Check for common delimiters
        delimiters = ["|", "\t", "    ", ";"]
        for delimiter in delimiters:
            if delimiter in line:
                cells = [cell.strip() for cell in line.split(delimiter)]
                if len(cells) >= 2 and all(cells):
                    return cells
        return None

    def _calculate_table_confidence(self, table: List[List[str]]) -> float:
        """Calculate confidence score for a table based on various metrics"""
        if not table or not table[0]:
            return 0.0

        score = 0.0
        num_rows = len(table)
        num_cols = len(table[0])
        
        # Check for minimum size
        if num_rows < 2 or num_cols < 2:
            return 0.0

        # Score based on consistency
        col_lengths = [len(row) for row in table]
        score += 0.3 if len(set(col_lengths)) == 1 else 0.0  # Consistent columns
        
        # Score based on content
        non_empty_cells = sum(1 for row in table for cell in row if cell.strip())
        content_ratio = non_empty_cells / (num_rows * num_cols)
        score += 0.3 * content_ratio
        
        # Score based on structure
        has_header = any(cell.isupper() for cell in table[0])
        score += 0.2 if has_header else 0.0
        
        return score

    def _hash_table(self, table: List[List[str]]) -> str:
        """Create a hash of table content to identify duplicates"""
        return hash(str(table))

    def _extract_with_gpt4v(self, pdf_path: str) -> List[List[List[str]]]:
        """Extract tables using GPT-4V"""
        if not self.llm_client:
            print("OpenAI key not provided, skipping GPT-4V", file=sys.stderr)
            return []
        
        try:
            import openai
            from PIL import Image
            
            # Initialize cost tracking
            total_tokens = 0
            total_images = 0
            estimated_cost = 0.0
            
            print("\nStarting GPT-4V table extraction...", file=sys.stderr)
            images = convert_from_path(pdf_path)
            all_tables = []
            
            # Process each page
            total_pages = len(images)
            for page_num, image in enumerate(images, 1):
                print(f"\nPage {page_num}/{total_pages}:", file=sys.stderr)
                
                # Save temporary image
                temp_path = f"{pdf_path}_temp_page_{page_num}.png"
                image.save(temp_path)
                
                try:
                    with open(temp_path, 'rb') as img_file:
                        print("  Sending to OpenAI...", file=sys.stderr)
                        
                        # Create OpenAI client
                        client = openai.OpenAI(api_key=self.llm_client)
                        
                        # Call GPT-4V with the exact prompt
                        response = client.chat.completions.create(
                            model="gpt-4-turbo-2024-04-09",
                            messages=[
                                {
                                    "role": "system",
                                    "content": """You are an expert data extraction model. I will provide you with an image that may contain one or more tables. Your goal is to identify and extract **all** tables present in the image and format them according to the rules below.

**What to return:**
- A Python list of tables.
- Each table is a list of rows.
- Each row is a list of cell strings (text from each cell).

**Detailed Requirements:**
1. **Identify All Tables**:
   - Scan the image for any tabular data (grids, rows, columns).
   - Each detected table must be treated as a separate item in the overall Python list.
   - Include any relevant headers above the table (e.g., "Schedule A", "Ineligible Items").
   - Include any relevant footers or totals below the table.

2. **Handle Rows and Cells**:
   - For each table, preserve the logical order of rows (top to bottom) and cells (left to right).
   - If you encounter merged cells (e.g., a cell that spans multiple rows or columns):
     - **Option A**: Duplicate the merged cell's text in each cell's position if you can confidently parse it.
     - **Option B**: Leave cells blank or put "N/A" if the correct content cannot be identified.
   - For section headers (like "Schedule A"), create a row with empty cells except for the header text.
   - For totals or summary rows, include them as regular rows at the bottom of the table.

3. **Clean Up Text**:
   - Remove any extraneous whitespace, line breaks, or non-table text.
   - Keep section headers, subtotals, and grand totals that are part of the table structure.
   - Preserve exact currency formatting (e.g., "$1,234.56").
   - Maintain original number formatting (commas, decimals, etc.).

4. **Edge Cases**:
   - If a table is malformed or partially visible, extract only the recognizable cells.
   - If no tables are found, return an empty list (e.g., `[]`).
   - For split tables that are clearly one logical table, combine them.
   - For tables with subtotals and grand totals, keep them with their parent table.

5. **Strict Output Format**:
   - Return only a **Python list** data structure with no additional explanatory text, comments, or markdown formatting.
   - Do not wrap your result in quotes or backticks.
   - Do not include any labels, headings, or bullet points. Just the raw Python list.

Remember: 
- Return only the list. 
- Do not include the word "Answer:" or any markdown in your output.
- If no tables are found, return `[]`."""
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"
                                            }
                                        }
                                    ]
                                }
                            ],
                            max_tokens=4096,
                            temperature=0.1,
                            response_format={"type": "text"}
                        )
                        
                        # Track usage
                        total_tokens += response.usage.total_tokens
                        total_images += 1
                        estimated_cost += (response.usage.total_tokens / 1000 * 0.01) + 0.00765
                        
                        # Parse the response
                        try:
                            response_text = response.choices[0].message.content.strip()
                            print(f"  Raw response: {response_text[:200]}...", file=sys.stderr)
                            
                            # Evaluate the response as Python code
                            tables_data = eval(response_text)
                            
                            # Validate and add tables
                            if isinstance(tables_data, list):
                                if tables_data and isinstance(tables_data[0], list):
                                    if isinstance(tables_data[0][0], list):  # Multiple tables
                                        for table in tables_data:
                                            if len(table) > 1:  # At least headers and one row
                                                all_tables.append(table)
                                        print(f"  Found {len(tables_data)} tables", file=sys.stderr)
                                    else:  # Single table
                                        if len(tables_data) > 1:  # At least headers and one row
                                            all_tables.append(tables_data)
                                            print(f"  Found 1 table with {len(tables_data)} rows", file=sys.stderr)
                                            
                        except Exception as e:
                            print(f"  Could not parse response: {str(e)}", file=sys.stderr)
                        
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            print(f"\nGPT-4V Processing Complete:", file=sys.stderr)
            print(f"- Pages processed: {total_pages}", file=sys.stderr)
            print(f"- Tables found: {len(all_tables)}", file=sys.stderr)
            print(f"- Total cost: ${estimated_cost:.4f}", file=sys.stderr)
            
            return all_tables
            
        except Exception as e:
            print(f"GPT-4V extraction failed: {str(e)}", file=sys.stderr)
            return []

    def _calculate_openai_cost(self, model: str, tokens: int, images: int = 0) -> float:
        """Calculate OpenAI API cost"""
        costs = {
            "gpt-4-turbo-2024-04-09": {
                "input": 0.01,   # per 1K tokens
                "output": 0.03,  # per 1K tokens
                "image": 0.00765  # per image
            }
        }
        
        if model not in costs:
            return 0.0
        
        model_costs = costs[model]
        token_cost = (tokens / 1000) * (model_costs["input"] + model_costs["output"])
        image_cost = images * model_costs["image"]
        
        return token_cost + image_cost

    def extract_with_gpt4v(self, image_path: str, extract_tables_only: bool = False) -> Union[List[List[List[str]]], Dict]:
        """Extract content from an image using GPT-4V"""
        if not self.llm_client:
            print("OpenAI key not provided, skipping GPT-4V", file=sys.stderr)
            return [] if extract_tables_only else {}
        
        try:
            import openai
            from PIL import Image
            
            print("  Sending to OpenAI...", file=sys.stderr)
            
            # Create OpenAI client
            client = openai.OpenAI(api_key=self.llm_client)
            
            # Read and encode image
            with open(image_path, 'rb') as img_file:
                base64_image = base64.b64encode(img_file.read()).decode()
            
            # Use appropriate prompt based on mode
            if extract_tables_only:
                system_prompt = """You are an expert document parser specialized in extracting structured content from images. Extract ALL content from this image and format it as a JSON object.

Format the response as:
{
    "text": "All text content from the image, excluding any content that appears in tables",
    "tables": [
        {
            "title": null,
            "data": [
                ["Column1", "Column2", "Column3", "Column4", "Column5", "Column6"],  # Original header row
                ["", "", "", "Category", "", ""],  # Category/section row with empty cells
                ["Data", "Data", "Data", "Data", "Data", "Data"]  # Data row
            ],
            "structure": {
                "header_rows": [0],
                "section_rows": [
                    {"index": 1, "text": "Category", "column": 3}  # Preserve exact text from document
                ],
                "total_rows": [],
                "column_types": {
                    "numeric": [],
                    "text": [],
                    "currency": []
                }
            }
        }
    ]
}

Critical Rules:
1. NEVER generate, modify, or enhance any content
2. NEVER add descriptive headers or labels
3. NEVER change the text of section headers
4. EXACTLY copy all text as it appears in the document
5. For section/category rows (like "Schedule A" or "Ineligible Items"):
   - Create a row with empty cells
   - Place the EXACT text in its original column
   - Keep all other cells empty
6. Preserve original:
   - Column headers
   - Section names
   - Numbers and currency formatting
   - Cell content exactly as shown
7. Do not combine or split cells
8. Do not add explanatory text
9. Do not reformat or standardize data
"""
                response_format = {"type": "json_object"}
            else:
                system_prompt = """You are an expert document parser specialized in extracting structured content from images. Extract ALL content from this image and format it as a JSON object.

Format the response as:
{
    "text": "All text content from the image, excluding any content that appears in tables",
    "tables": [
        {
            "title": null,
            "data": [
                ["Column1", "Column2", "Column3", "Column4", "Column5", "Column6"],  # Original header row
                ["", "", "", "Category", "", ""],  # Category/section row with empty cells
                ["Data", "Data", "Data", "Data", "Data", "Data"]  # Data row
            ],
            "structure": {
                "header_rows": [0],
                "section_rows": [
                    {"index": 1, "text": "Category", "column": 3}  # Preserve exact text from document
                ],
                "total_rows": [],
                "column_types": {
                    "numeric": [],
                    "text": [],
                    "currency": []
                }
            }
        }
    ]
}

Critical Rules:
1. NEVER generate, modify, or enhance any content
2. NEVER add descriptive headers or labels
3. NEVER change the text of section headers
4. EXACTLY copy all text as it appears in the document
5. For section/category rows (like "Schedule A" or "Ineligible Items"):
   - Create a row with empty cells
   - Place the EXACT text in its original column
   - Keep all other cells empty
6. Preserve original:
   - Column headers
   - Section names
   - Numbers and currency formatting
   - Cell content exactly as shown
7. Do not combine or split cells
8. Do not add explanatory text
9. Do not reformat or standardize data
"""
                response_format = {"type": "json_object"}
            
            # Call GPT-4V with updated model name
            model = "gpt-4-turbo-2024-04-09"
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all content from this image, including both text and tables."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=0.1,
                response_format=response_format
            )
            
            # Calculate and log cost
            self.last_cost = self._calculate_openai_cost(model, response.usage.total_tokens, images=1)
            print(f"  Cost: ${self.last_cost:.4f} (Tokens: {response.usage.total_tokens}, Images: 1)", file=sys.stderr)
            
            # Parse response based on mode
            if extract_tables_only:
                try:
                    content = response.choices[0].message.content.strip()
                    # Clean up the response to ensure valid Python syntax
                    content = content.replace("'", '"')  # Replace single quotes with double quotes
                    content = re.sub(r'(\d),(\d)', r'\1\2', content)  # Remove commas in numbers
                    content = re.sub(r'\$\s*(\d+)', r'$ \1', content)  # Format currency consistently
                    
                    # Safely evaluate the string as Python code
                    tables_data = ast.literal_eval(content)
                    
                    if not isinstance(tables_data, list):
                        print("  Warning: Response was not a list", file=sys.stderr)
                        return []
                        
                    # Validate table structure
                    valid_tables = []
                    for table in tables_data:
                        if isinstance(table, list) and all(isinstance(row, list) for row in table):
                            # Convert all values to strings
                            valid_table = [[str(cell).strip() for cell in row] for row in table]
                            valid_tables.append(valid_table)
                    
                    return valid_tables
                    
                except Exception as e:
                    print(f"  Could not parse tables: {str(e)}", file=sys.stderr)
                    print(f"  Raw response: {content[:200]}...", file=sys.stderr)
                    return []
            else:
                try:
                    data = json.loads(response.choices[0].message.content)
                    markdown_content = []
                    
                    if data.get("text"):
                        text = data["text"].strip()
                        if text:
                            markdown_content.append(text + "\n")
                    
                    # Use class-level seen_tables
                    for table in data.get("tables", []):
                        table_data = table.get("data", [])
                        table_hash = hash(str(table_data))
                        
                        if table_hash not in self.seen_tables:
                            self.seen_tables.add(table_hash)
                            
                            if table.get("title"):
                                markdown_content.append(f"\n### {table['title']}\n")
                            
                            if table_data:
                                # Format table with section headers
                                table_md = self._format_table(table_data, structure=table.get("structure", {}))
                                markdown_content.extend(table_md)
                                markdown_content.append("\n")
                    
                    return {
                        "text": "\n".join(markdown_content),
                        "tables": data.get("tables", [])
                    }
                    
                except Exception as e:
                    print(f"  Could not parse JSON: {str(e)}", file=sys.stderr)
                    return {"text": "", "tables": []}
                
        except Exception as e:
            print(f"GPT-4V extraction failed: {str(e)}", file=sys.stderr)
            return [] if extract_tables_only else {"text": "", "tables": []}

    def _format_table(self, table: List[List[str]], structure: Dict = None) -> List[str]:
        """Format table with improved structure handling"""
        if not table or not structure:
            return []

        # Clean and standardize table
        cleaned_table = []
        max_cols = max(len(row) for row in table)
        
        # Process header row first
        header_row = table[0]
        header_row = [str(cell).strip() for cell in header_row] + [''] * (max_cols - len(header_row))
        cleaned_table.append(header_row)
        
        # Determine alignments based on column types
        alignments = []
        col_types = structure.get('column_types', {})
        numeric_cols = set(col_types.get('numeric', []))
        text_cols = set(col_types.get('text', []))
        
        for i in range(max_cols):
            if i in numeric_cols:
                alignments.append('---:')  # Right align
            elif i in text_cols:
                alignments.append(':---')  # Left align
            else:
                alignments.append(':---:')  # Center align
        
        # Format markdown table
        markdown_lines = []
        markdown_lines.append('| ' + ' | '.join(header_row) + ' |')
        markdown_lines.append('|' + '|'.join(alignments) + '|')
        
        # Process data rows with special handling for section headers
        section_headers = {row['row']: (row['text'], row['column']) 
                          for row in structure.get('section_headers', [])}
        
        for row_idx, row in enumerate(table[1:], 1):
            if row_idx in section_headers:
                # Create section header row
                text, col = section_headers[row_idx]
                cells = [''] * max_cols
                cells[col] = f"**{text}**"
                markdown_lines.append('| ' + ' | '.join(cells) + ' |')
            else:
                # Regular data row
                cells = [str(cell).strip() for cell in row] + [''] * (max_cols - len(row))
                markdown_lines.append('| ' + ' | '.join(cells) + ' |')
        
        return markdown_lines

    def _post_process_table(self, markdown_lines: List[str], num_columns: int) -> List[str]:
        """Post-process table to fix special rows and formatting"""
        fixed_lines = []
        header_pattern = re.compile(r'^\s*(Schedule [A-Z]|Ineligible Items|Total|Subtotal|Grand Total)\s*$', re.IGNORECASE)
        
        for line in markdown_lines:
            # Skip alignment row
            if line.count('---') > 0:
                fixed_lines.append(line)
                continue
                
            # Extract cells
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            
            # Check if any cell matches header pattern
            for i, cell in enumerate(cells):
                if header_pattern.match(cell):
                    # Create new row with the header in a centered position
                    new_cells = [''] * num_columns
                    center_pos = num_columns // 2
                    new_cells[center_pos] = f"**{cell}**"  # Make it bold
                    line = '| ' + ' | '.join(new_cells) + ' |'
                    break
                    
            # Handle currency formatting
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            formatted_cells = []
            for cell in cells:
                # Standardize currency format
                if re.match(r'^\$?\s*[\d,.]+$', cell.strip()):
                    # Remove existing $ and spaces
                    num = cell.replace('$', '').replace(' ', '')
                    # Format with $ and proper spacing
                    cell = f"$ {num}"
                formatted_cells.append(cell)
            
            line = '| ' + ' | '.join(formatted_cells) + ' |'
            fixed_lines.append(line)
            
        return fixed_lines

    def _clean_table_text(self, text: str) -> str:
        """Clean table text for better matching"""
        if not text:
            return ""
            
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Standardize currency formatting
        text = re.sub(r'\$\s+', '$', text)
        
        # Keep thousands separators for readability
        # text = re.sub(r'(?<=\d),(?=\d{3})', '', text)  # Removed this line
        
        # Standardize number formatting
        text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)  # Remove space between numbers
        
        # Clean up special characters
        text = text.replace('–', '-')  # Standardize dashes
        text = text.replace('—', '-')
        
        return text.strip()

    def _validate_table(self, table: List[List[str]]) -> bool:
        """Validate table structure and content"""
        if not table or not table[0]:
            return False
        
        try:
            # Check row lengths are consistent
            row_length = len(table[0])
            if not all(len(row) == row_length for row in table):
                return False
            
            # Check for minimum content
            content_cells = sum(1 for row in table for cell in row if str(cell).strip())
            if content_cells < (row_length * 2):  # At least header and one data row
                return False
            
            # Check for reasonable cell content
            for row in table:
                for cell in row:
                    cell_text = str(cell).strip()
                    if cell_text and len(cell_text) > 1000:  # Cell too long
                        return False
                    
                    # Check for invalid characters or patterns
                    if any(char in cell_text for char in ['<', '>', '{', '}']):
                        return False
            
            # Check for header-like first row
            header_row = [str(cell).strip() for cell in table[0]]
            if not any(header_row):  # Empty header
                return False
            
            return True
            
        except Exception as e:
            print(f"Table validation error: {str(e)}", file=sys.stderr)
            return False

class MarkItDown:
    """Main class for converting documents to markdown."""
    
    def __init__(self, 
                 llm_client=None, 
                 llm_model=None,
                 table_strategy: Union[TableExtractionStrategy, str] = TableExtractionStrategy.HEURISTIC):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.pdf_table_extractor = PdfTableExtractor(llm_client=llm_client)
        
        # Set table extraction strategy
        if isinstance(table_strategy, str):
            table_strategy = TableExtractionStrategy(table_strategy.lower())
        self.table_strategy = table_strategy
        
        # Initialize ML models if needed
        self.layout_model = None
        if table_strategy in [TableExtractionStrategy.LAYOUTLM, TableExtractionStrategy.AUTO]:
            if HAS_LAYOUTLM:
                self.layout_model = self._init_layoutlm()
            else:
                warnings.warn("LayoutLMv3 requested but not available")

        # Add support for different file handlers
        self.handlers = {
            '.pdf': self._handle_pdf,
            '.docx': self._handle_docx,
            '.xlsx': self._handle_excel,
            '.pptx': self._handle_powerpoint,
            '.jpg': self._handle_image,
            '.png': self._handle_image,
            '.html': self._handle_html,
            '.json': self._handle_json,
            '.xml': self._handle_xml,
            '.zip': self._handle_zip,
            '.msg': self._handle_msg
        }

    def convert(self, file_path: str) -> ConversionResult:
        """Convert with progress reporting"""
        print(f"\nProcessing: {os.path.basename(file_path)}")
        
        try:
            # Get file extension
            ext = os.path.splitext(file_path)[1].lower()
            print(f"File type: {ext}")
            
            # Get appropriate handler
            handler = self.handlers.get(ext)
            if not handler:
                raise UnsupportedFormatException(f"Unsupported file type: {ext}")
            
            # Convert file
            print("Converting...")
            result = handler(file_path)
            print("Conversion complete")
            
            # Save result
            output_dir = result.get('output_dir') or os.path.join(os.path.dirname(file_path), "output")
            os.makedirs(output_dir, exist_ok=True)
            
            markdown_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + '.md')
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(result['text'])
            print(f"Saved to: {markdown_path}")
            
            return ConversionResult(
                text_content=result['text'],
                metadata=result.get('metadata', {}),
                images=result.get('images', [])
            )
            
        except Exception as e:
            print(f"Error: {str(e)}")
            raise FileConversionException(f"Error converting {file_path}: {str(e)}")

    def convert_batch(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """Convert all files in a directory with progress bar"""
        try:
            from tqdm import tqdm
        except ImportError:
            print("For progress bars, install tqdm: pip install tqdm")
            tqdm = lambda x: x  # Fallback if tqdm not installed
        
        results = {
            'successful': [],
            'failed': [],
            'skipped': []
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of all files first
        all_files = []
        for root, _, files in os.walk(input_dir):
            for filename in files:
                all_files.append((root, filename))
        
        # Process files with progress bar
        for root, filename in tqdm(all_files, desc="Converting files", unit="file"):
            input_path = os.path.join(root, filename)
            
            # Create relative path for output
            rel_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            output_path = os.path.splitext(output_path)[0] + '.md'
            
            try:
                # Convert file
                result = self.convert(input_path)
                
                # Create output directory if needed
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Write markdown output
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result.text_content)
                
                results['successful'].append({
                    'input': input_path,
                    'output': output_path,
                    'metadata': result.metadata,
                    'images': result.images
                })
                
            except UnsupportedFormatException:
                results['skipped'].append({
                    'input': input_path,
                    'reason': 'Unsupported file type'
                })
            except Exception as e:
                results['failed'].append({
                    'input': input_path,
                    'error': str(e)
                })
                
        return results

    def _handle_pdf(self, file_path: str) -> Dict[str, Any]:
        """Handle PDF conversion with both text and table extraction"""
        try:
            # Setup directories
            base_dir = os.path.dirname(file_path)
            output_dir = os.path.join(base_dir, "output")
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Reset table extractor state
            self.pdf_table_extractor.reset_state()
            
            with pdfplumber.open(file_path) as pdf:
                text_content = []
                images = []
                
                # Extract content page by page
                print("\nExtracting content...", file=sys.stderr)
                for page_num, page in enumerate(pdf.pages, 1):
                    text_content.append(f"\n## Page {page_num}\n\n")
                    
                    # Convert page to image for GPT-4V
                    images_from_path = convert_from_path(file_path, first_page=page_num, last_page=page_num)
                    if images_from_path:
                        img_path = os.path.join(images_dir, f"page_{page_num}.png")
                        images_from_path[0].save(img_path)
                        images.append(img_path)
                        
                        if self.llm_client:  # Use GPT-4V if available
                            gpt4v_result = self.pdf_table_extractor.extract_with_gpt4v(img_path, extract_tables_only=False)
                            if gpt4v_result and isinstance(gpt4v_result, dict):
                                # Add extracted text
                                if gpt4v_result.get("text"):
                                    text_content.append(gpt4v_result["text"] + "\n\n")
                                
                                # Add tables
                                for table in gpt4v_result.get("tables", []):
                                    table_data = table.get("data", [])
                                    table_hash = hash(str(table_data))
                                    
                                    if table_hash not in self.pdf_table_extractor.seen_tables:
                                        self.pdf_table_extractor.seen_tables.add(table_hash)
                                        if table.get("title"):
                                            text_content.append(f"\n### {table['title']}\n")
                                        if table_data:
                                            table_md = self.pdf_table_extractor._format_table(table_data)
                                            if table_md:
                                                text_content.extend(table_md)
                                                text_content.append("\n")
                            else:  # Use traditional methods if GPT-4V not available
                                page_text = page.extract_text()
                                if not page_text:
                                    page_text = self._extract_text_with_ocr(page, file_path, page_num)
                                if page_text:
                                    text_content.append(page_text + "\n\n")
                                
                                # Extract tables using default method
                                tables = self.pdf_table_extractor.extract_tables(file_path)
                                for table in tables:
                                    if self._table_belongs_to_page(table, page_text):
                                        table_md = self.pdf_table_extractor._format_table(table)
                                        if table_md:
                                            text_content.extend(table_md)
                                            text_content.append("\n")
                    
                    # Add image reference
                    rel_img_path = os.path.relpath(img_path, output_dir).replace('\\', '/')
                    text_content.append(f"\n![Page {page_num}]({rel_img_path})\n")
                
                # Add file identifier at the end
                text_content.append(f"\n{os.path.splitext(os.path.basename(file_path))[0]}\n")
                
                return {
                    'text': '\n'.join(text_content),
                    'images': images,
                    'output_dir': output_dir
                }
                
        except Exception as e:
            print(f"Error processing PDF {os.path.basename(file_path)}: {str(e)}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {
                'text': f"Error processing file: {str(e)}",
                'images': [],
                'output_dir': os.path.join(os.path.dirname(file_path), "output")
            }

    def _handle_docx(self, file_path: str) -> Dict[str, Any]:
        """Handle DOCX conversion"""
        try:
            # Create output directory structure
            base_dir = os.path.dirname(file_path)
            output_dir = os.path.join(base_dir, "output")
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Convert docx to html
            with open(file_path, 'rb') as docx_file:
                result = mammoth.convert_to_html(docx_file)
            
            # Extract and save images
            images = []
            for message in result.messages:
                if message.type == "image":
                    try:
                        base_name = os.path.splitext(os.path.basename(file_path))[0]
                        img_path = os.path.join(images_dir, f"{base_name}_image_{len(images)}.png")
                        with open(img_path, 'wb') as f:
                            f.write(message.value)
                        images.append(img_path)
                    except Exception as e:
                        print(f"Warning: Could not save image: {e}", file=sys.stderr)
            
            # Convert HTML to markdown
            markdown = markdownify.markdownify(result.value, heading_style="ATX")
            
            return {
                'text': markdown,
                'metadata': {'image_count': len(images)},
                'images': images,
                'output_dir': output_dir
            }
            
        except Exception as e:
            raise FileConversionException(f"Error converting DOCX: {str(e)}")

    def _extract_text_with_ocr(self, page, file_path: str, page_num: int) -> str:
        """Extract text from a page using OCR"""
        try:
            # Convert page to image for OCR
            images_from_path = convert_from_path(file_path, first_page=page_num, last_page=page_num)
            if not images_from_path:
                return ""
            
            # Save temporary image
            temp_path = f"{file_path}_temp_page_{page_num}.png"
            images_from_path[0].save(temp_path)
            
            try:
                # Perform OCR
                text = self._perform_ocr(temp_path)
                if text:
                    print(f"  OCR successful for page {page_num}", file=sys.stderr)
                    return text
                else:
                    print(f"  No text found by OCR on page {page_num}", file=sys.stderr)
                    return ""
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
        except Exception as e:
            print(f"Warning: OCR failed for page {page_num}: {e}", file=sys.stderr)
            return ""

    def _perform_ocr(self, image_path: str) -> str:
        """Perform OCR on an image"""
        try:
            import pytesseract
            from PIL import Image
            
            # Set explicit Tesseract path for Windows
            if sys.platform == "win32":
                pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            
            # Perform OCR
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
            
        except Exception as e:
            print(f"Warning: OCR failed: {str(e)}", file=sys.stderr)
            return ""

    def _table_belongs_to_page(self, table: List[List[str]], page_text: str) -> bool:
        """Check if a table belongs to the current page"""
        if not table or not page_text:
            return False
            
        # Convert table to text for comparison
        table_text = ' '.join(' '.join(str(cell) for cell in row) for row in table)
        
        # Look for significant matches
        header_row = table[0] if table else []
        data_row = table[1] if len(table) > 1 else []
        
        # Check if header or first data row appears in page
        header_match = all(str(cell) in page_text for cell in header_row if cell)
        data_match = all(str(cell) in page_text for cell in data_row if cell)
        
        return header_match or data_match

    # Add stubs for other handlers
    def _handle_excel(self, file_path: str) -> Dict[str, Any]:
        raise NotImplementedError("Excel handling not implemented yet")

    def _handle_powerpoint(self, file_path: str) -> Dict[str, Any]:
        raise NotImplementedError("PowerPoint handling not implemented yet")

    def _handle_image(self, file_path: str) -> Dict[str, Any]:
        raise NotImplementedError("Image handling not implemented yet")

    def _handle_html(self, file_path: str) -> Dict[str, Any]:
        raise NotImplementedError("HTML handling not implemented yet")

    def _handle_json(self, file_path: str) -> Dict[str, Any]:
        raise NotImplementedError("JSON handling not implemented yet")

    def _handle_xml(self, file_path: str) -> Dict[str, Any]:
        raise NotImplementedError("XML handling not implemented yet")

    def _handle_zip(self, file_path: str) -> Dict[str, Any]:
        raise NotImplementedError("ZIP handling not implemented yet")

    def _handle_msg(self, file_path: str) -> Dict[str, Any]:
        raise NotImplementedError("MSG handling not implemented yet")