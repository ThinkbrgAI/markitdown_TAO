# type: ignore
import base64
import binascii
import copy
import html
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse, urlunparse
from enum import Enum
import ast

import numpy as np
import pytesseract
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn("OpenCV not available. Install with 'pip install opencv-python'")

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub.utils")

import mammoth
import markdownify
import pandas as pd
import pdfplumber
import puremagic
import requests
from bs4 import BeautifulSoup

import camelot
import tabula
from pdf2image import convert_from_path

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

# Configure basic logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TableExtractionStrategy(Enum):
    HEURISTIC = "heuristic"
    LAYOUTLM = "layoutlm"
    GPT4V = "gpt4v"
    AUTO = "auto"

class FileConversionException(Exception):
    pass

class UnsupportedFormatException(Exception):
    pass

class ConversionResult:
    """Container for conversion results."""
    def __init__(self, text_content: str = "", metadata: Dict = None, images: List[str] = None):
        self.text_content = text_content
        self.metadata = metadata or {}
        self.images = images or []

    def __str__(self):
        return f"ConversionResult(text_length={len(self.text_content)}, images={len(self.images)})"

# --------------------------------------------------
# PDF Table Extractor Class
# --------------------------------------------------

class PdfTableExtractor:
    """
    Advanced PDF table extraction class with multiple strategies.
    Includes orientation correction, GPT-4V usage with cost control,
    basic no-hallucination verification, and partial fallback logic.
    """

    # Default prompt that can be restored
    DEFAULT_PROMPT = (
        "You are an expert data extraction model. Your task is to detect and extract ONLY tabular data that clearly appears in the provided image.\n\n"
        "**What to Return**:\n"
        "- A SINGLE Python list named `tables`.\n"
        "- Each item in `tables` is a dictionary with two keys:\n"
        "  1. \"title\": a string containing the table's visible title (or \"\" if none).\n"
        "  2. \"rows\": a list of rows, each row being a list of cell strings.\n"
        "- If there is no clearly visible table in the image, return `tables = []`.\n\n"
        "**Critical Instructions**:\n"
        "1. **Extract Only VISIBLE Table Data**:\n"
        "   - ONLY extract data that you can actually see in the image\n"
        "   - DO NOT invent, generate, or hallucinate ANY data\n"
        "   - If you can't read a cell clearly, use \"\" (empty string)\n"
        "   - If a table is partially visible, only extract the visible parts\n"
        "2. **No Additional Explanations**:\n"
        "   - Return ONLY the Python object named `tables`\n"
        "   - Do NOT add commentary, JSON wrappers, or Markdown\n"
        "   - Do NOT wrap your output in quotes or code blocks\n"
        "3. **Exact Cell Preservation**:\n"
        "   - Copy cell contents EXACTLY as they appear\n"
        "   - Keep multi-word cells in a single string\n"
        "   - For merged cells, pad with \"\" to maintain column count\n"
        "4. **Multiple Tables**:\n"
        "   - If multiple tables exist, extract each one\n"
        "   - Each table becomes a separate dictionary in `tables`\n"
        "5. **No Table**:\n"
        "   - If no table is visible, return `tables = []`\n\n"
        "**Important**:\n"
        "- NEVER invent or generate data - only extract what you actually see\n"
        "- Provide your final result in valid Python syntax\n"
        "- Example format (DO NOT COPY THIS DATA):\n\n"
        "tables = [\n"
        "  {\n"
        "    \"title\": \"\",\n"
        "    \"rows\": [\n"
        "      [\"Header1\", \"Header2\"],\n"
        "      [\"Cell1\", \"Cell2\"]\n"
        "    ]\n"
        "  }\n"
        "]\n\n"
        "Nothing else."
    )

    def __init__(self, llm_client: Optional[Any] = None, max_gpt4v_cost: float = 2.0):
        """
        :param llm_client: OpenAI-like client or string (API key)
        :param max_gpt4v_cost: Max cost in USD for GPT-4V usage
        """
        if isinstance(llm_client, str):
            try:
                import openai
                openai.api_key = llm_client
                self.llm_client = openai
            except ImportError:
                logger.warning("OpenAI not installed; GPT-4V calls will fail.")
                self.llm_client = None
        else:
            self.llm_client = llm_client

        self.total_cost = 0.0  # Add total cost tracking
        self.last_cost = 0.0
        self.seen_tables = set()
        self.max_gpt4v_cost = max_gpt4v_cost
        self.PROMPT_SYSTEM_GPT4V = self.DEFAULT_PROMPT  # Initialize with new default

        self.strategies = [
            self._extract_with_camelot,
            self._extract_with_tabula,
            self._extract_with_pdfplumber,
            self._extract_with_layoutlm,
            self._extract_with_gpt4v,
            self._extract_with_heuristics
        ]

    def reset_state(self) -> None:
        """Reset internal state for a new document."""
        self.seen_tables.clear()
        self.last_cost = 0.0

    def extract_tables(self, pdf_path: str) -> List[List[List[str]]]:
        """
        Attempt to extract tables from the PDF using multiple strategies in sequence.
        Returns a list of unique tables (list-of-rows), sorted by highest 'confidence.'
        For brevity, left as a placeholder.
        """
        return []

    # ---------------------------------------------------------------------
    # Strategy 1: Camelot
    # ---------------------------------------------------------------------
    def _extract_with_camelot(self, pdf_path: str) -> List[List[List[str]]]:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if not pdf.pages or not pdf.pages[0].extract_text():
                    logger.info("PDF appears image-based; Camelot won't be effective.")
                    return []
            tables = camelot.read_pdf(pdf_path, pages="1-end")
            return [t.data for t in tables]
        except Exception as e:
            logger.warning(f"Camelot extraction failed: {e}")
            return []

    # ---------------------------------------------------------------------
    # Strategy 2: Tabula
    # ---------------------------------------------------------------------
    def _extract_with_tabula(self, pdf_path: str) -> List[List[List[str]]]:
        """Extract tables using Tabula."""
        try:
            df_list = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            result = []
            for df in df_list:
                if not df.empty:
                    result.append(df.values.tolist())
            return result
        except Exception as e:
            logger.warning(f"Tabula extraction failed: {e}")
            return []

    # ---------------------------------------------------------------------
    # Strategy 3: pdfplumber
    # ---------------------------------------------------------------------
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[List[List[str]]]:
        try:
            all_tables = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    for t in page_tables:
                        all_tables.append(t)
            return all_tables
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
            return []

    # ---------------------------------------------------------------------
    # Strategy 4: LayoutLM
    # ---------------------------------------------------------------------
    def _extract_with_layoutlm(self, pdf_path: str) -> List[List[List[str]]]:
        if not HAS_LAYOUTLM:
            return []
        try:
            # Placeholder for an actual LayoutLM-based table extraction pipeline
            return []
        except Exception as e:
            logger.warning(f"LayoutLM extraction failed: {e}")
        return []

    # ---------------------------------------------------------------------
    # Strategy 5: GPT-4V
    # ---------------------------------------------------------------------
    def _extract_with_gpt4v(self, image_path: str, extract_tables_only: bool = False) -> Union[Dict, List]:
        """Use GPT-4V to extract tables from an image."""
        if not self.llm_client:
            logger.info("No LLM client available; skipping GPT-4V extraction.")
            return [] if extract_tables_only else {"text": "", "tables": []}

        try:
            # Read and encode image
            with open(image_path, "rb") as img_file:
                b64_image = base64.b64encode(img_file.read()).decode()

            # Send to OpenAI
            logger.info("Sending to OpenAI...")
            response = self.llm_client.chat.completions.create(
                model="gpt-4-turbo-2024-04-09",
                messages=[
                    {"role": "system", "content": self.PROMPT_SYSTEM_GPT4V},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all tables from this image. Return only a Python list of table dictionaries with 'title' and 'rows' keys."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=0
            )

            # Log the truly raw response first
            logger.info("Raw GPT response before any processing:")
            logger.info("-" * 40)
            logger.info(response.choices[0].message.content)
            logger.info("-" * 40)

            # Calculate cost
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            self.last_cost = (input_tokens * 0.01 + output_tokens * 0.03) / 1000 + 0.01  # $0.01 per image
            self.total_cost += self.last_cost
            logger.info(f"Cost: ${self.last_cost:.4f} (Tokens: {input_tokens + output_tokens}, Images: 1)")

            # Parse response
            content = response.choices[0].message.content
            return self._parse_gpt4v_response(content)

        except Exception as e:
            logger.error(f"GPT-4V API call failed: {e}")
            self.last_cost = 0.0
            return []

    def _verify_gpt4v_tables(self, extracted_tables: Any, pil_img: Image.Image) -> List[List[List[str]]]:
        """Cross-check extracted words with OCR to reduce hallucination risk."""
        try:
            # First try to parse the content as Python literal
            try:
                if isinstance(extracted_tables, str):
                    tables = ast.literal_eval(extracted_tables)
                else:
                    tables = extracted_tables
            except Exception:
                logger.warning("Could not parse GPT-4V response as Python literal")
                return []

            if not isinstance(tables, list):
                return []

            # Get OCR text for verification
            ocr_text = pytesseract.image_to_string(pil_img).lower()
            recognized_words = set(re.findall(r"\w+", ocr_text))

            verified_tables = []
            for table in tables:
                if not isinstance(table, list) or not table:
                    continue

                # Extract words from table for verification
                table_words = set()
                for row in table:
                    if not isinstance(row, list):
                        continue
                    for cell in row:
                        if cell:
                            table_words.update(re.findall(r"\w+", str(cell).lower()))

                if not table_words:
                    # Empty table or no text to verify - keep it
                    verified_tables.append(table)
                    continue

                # Calculate match ratio
                matches = len(table_words.intersection(recognized_words))
                ratio = matches / len(table_words) if table_words else 0

                if ratio > 0.3:  # At least 30% of words should match
                    verified_tables.append(table)
                    logger.info(f"Table verified with match ratio {ratio:.2f}")
                else:
                    logger.warning(f"Table discarded due to low match ratio {ratio:.2f}")

            return verified_tables

        except Exception as e:
            logger.error(f"Table verification failed: {e}")
            return []

    def _format_table_markdown(self, table: List[List[str]]) -> List[str]:
        """Format a table as markdown with proper alignment and spacing"""
        if not table or not isinstance(table, list):
            return []

        # Format each row as markdown
        md_lines = []
        md_lines.append('')  # Blank line before table

        # Add header row
        if table:
            md_lines.append('| ' + ' | '.join(str(cell) for cell in table[0]) + ' |')
            
            # Add separator with proper column alignment
            separators = []
            for cell in table[0]:
                if isinstance(cell, (int, float)):
                    separators.append(':---:')  # Center align numbers
                else:
                    separators.append('---')    # Left align text
            md_lines.append('| ' + ' | '.join(separators) + ' |')

            # Add data rows
            for row in table[1:]:
                md_lines.append('| ' + ' | '.join(str(cell) for cell in row) + ' |')

        md_lines.append('')  # Blank line after table
        return md_lines

    # ---------------------------------------------------------------------
    # Strategy 6: Heuristics
    # ---------------------------------------------------------------------
    def _extract_with_heuristics(self, pdf_path: str) -> List[List[List[str]]]:
        """A last fallback approach: parse raw text lines for table-like delimiters."""
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    raw_text = page.extract_text() or ""
                    current_table = []
                    for line in raw_text.split("\n"):
                        row_cells = self._detect_table_row(line)
                        if row_cells:
                            current_table.append(row_cells)
                        else:
                            if current_table and len(current_table) > 1:
                                tables.append(current_table)
                            current_table = []
                    if current_table and len(current_table) > 1:
                        tables.append(current_table)
        except Exception as e:
            logger.warning(f"Heuristic extraction failed: {e}")
        return tables

    def _detect_table_row(self, line: str) -> Optional[List[str]]:
        """Detect table-like row by common delimiters."""
        delimiters = ["|", "\t", "    ", ";"]
        for delim in delimiters:
            if delim in line:
                cells = [c.strip() for c in line.split(delim)]
                if len(cells) >= 2 and any(cells):
                    return cells
        return None

    # ---------------------------------------------------------------------
    # Orientation Correction
    # ---------------------------------------------------------------------
    def _detect_and_correct_orientation(self, image_path: str) -> str:
        """Use OCR confidence and visual features to detect best orientation."""
        if not HAS_CV2:
            return image_path
        try:
            img = cv2.imread(image_path)
            if img is None:
                return image_path

            angles = [0, 90, 180, 270]
            max_conf = 0.0
            best_angle = 0
            
            # Convert to RGB for consistent processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            for angle in angles:
                if angle == 0:
                    rotated = img_rgb
                else:
                    height, width = img_rgb.shape[:2]
                    center = (width/2, height/2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(
                        img_rgb, rotation_matrix, (width, height),
                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                    )

                # Convert to PIL Image for OCR
                pil_img = Image.fromarray(rotated)
                
                try:
                    # Try OCR with different PSM modes for better detection
                    for psm in [6, 3]:  # Most reliable modes for orientation
                        ocr_data = pytesseract.image_to_data(
                            pil_img, 
                            output_type=pytesseract.Output.DICT,
                            config=f'--psm {psm} --oem 3'
                        )
                        conf_scores = [float(conf) for conf in ocr_data['conf'] if conf != '-1']
                        text_len = sum(len(word) for word in ocr_data['text'] if word.strip())
                        
                        # Weight confidence by amount of text found
                        avg_conf = float(np.mean(conf_scores)) * text_len if conf_scores else 0.0
                        
                        if avg_conf > max_conf:
                            max_conf = avg_conf
                            best_angle = angle
                            logger.debug(f"New best orientation: {angle}° (confidence: {avg_conf:.2f})")
                except Exception as e:
                    logger.warning(f"OCR failed for angle {angle}: {e}")
                    continue

            if best_angle != 0:
                logger.info(f"Rotating image by {best_angle}°")
                height, width = img.shape[:2]
                center = (width/2, height/2)
                rot_matrix = cv2.getRotationMatrix2D(center, best_angle, 1.0)
                rotated = cv2.warpAffine(img, rot_matrix, (width, height),
                                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                base, ext = os.path.splitext(image_path)
                rotated_path = f"{base}_rotated{ext}"
                cv2.imwrite(rotated_path, rotated)
                return rotated_path

            return image_path
        except Exception as e:
            logger.warning(f"Orientation detection failed: {e}")
            return image_path

    # ---------------------------------------------------------------------
    # Cost & Utility Helpers
    # ---------------------------------------------------------------------
    def _estimate_gpt4v_cost(self, num_pages: int = 1) -> float:
        """A rough cost estimate for GPT-4V usage."""
        tokens_per_page = 500
        images_per_page = 1
        cost_per_page = self._calculate_openai_cost(
            "gpt-4-turbo-2024-04-09", tokens_per_page, images=images_per_page
        )
        return cost_per_page * num_pages

    def _calculate_openai_cost(self, model: str, tokens: int, images: int = 0) -> float:
        costs = {
            "gpt-4-turbo-2024-04-09": {
                "input": 0.01,   # per 1K tokens
                "output": 0.03,  # per 1K tokens
                "image": 0.00765 # per image
            }
        }
        if model not in costs:
            return 0.0
        c = costs[model]
        token_cost = (tokens / 1000.0) * (c["input"] + c["output"])
        image_cost = images * c["image"]
        return token_cost + image_cost

    def _parse_gpt4v_response(self, content: str) -> List[Dict[str, Any]]:
        """Parse and validate GPT-4V response with better error handling."""
        if not content:
            return []
        
        # Always log the raw response for review
        logger.info("GPT-4V raw response:")
        logger.info("-" * 40)
        logger.info(content)
        logger.info("-" * 40)
        
        # Clean up the response
        content = content.strip()
        
        # Try to find the tables list using regex with more flexible patterns
        patterns = [
            r'tables\s*=\s*(\[[\s\S]*\])\s*$',  # Standard format
            r'(\[[\s\S]*\])\s*$',               # Just a list
            r'(\[{[\s\S]*}])\s*$'               # List of objects
        ]
        
        cleaned_content = None
        for pattern in patterns:
            match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
            if match:
                cleaned_content = match.group(1)
                break
        
        if not cleaned_content:
            logger.error("Could not find valid table structure in response")
            # Still try to use the full response as a fallback
            cleaned_content = content
        
        # Additional cleanup
        cleaned_content = re.sub(r'```\w*\n?', '', cleaned_content)  # Remove code fences
        cleaned_content = re.sub(r'tables\s*=\s*', '', cleaned_content)  # Remove variable assignment
        cleaned_content = re.sub(r'[\n\r]+', ' ', cleaned_content)  # Normalize newlines
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)      # Normalize spaces
        cleaned_content = re.sub(r',\s*([}\]])', r'\1', cleaned_content)  # Remove trailing commas
        cleaned_content = re.sub(r'None', '""', cleaned_content)    # Replace None with empty string
        
        try:
            # Parse the Python literal
            tables = ast.literal_eval(cleaned_content)
            if not isinstance(tables, list):
                logger.error(f"GPT response not a list: {type(tables)}")
                return []
            
            # Validate table structure with more lenient rules
            valid_tables = []
            for table in tables:
                if not isinstance(table, dict):
                    logger.warning(f"Invalid table type: {type(table)}")
                    # Try to convert non-dict table to dict format
                    if isinstance(table, list):
                        table = {
                            'title': '',
                            'rows': table
                        }
                
                # Allow missing title
                if 'title' not in table:
                    table['title'] = ''
                
                # Check for rows
                if 'rows' not in table and isinstance(table, dict):
                    # Try to find any list that could be rows
                    for key, value in table.items():
                        if isinstance(value, list) and value:
                            table['rows'] = value
                            break
                
                if not isinstance(table.get('rows', []), list):
                    logger.warning("'rows' is not a list")
                    continue
                
                # Allow empty tables but require at least one row
                if not table.get('rows', []):
                    logger.warning("Empty rows list")
                    continue
                
                # Validate and clean up rows
                cleaned_rows = []
                for row in table['rows']:
                    if not isinstance(row, list):
                        # Try to convert single items to a one-cell row
                        cleaned_rows.append([str(row)])
                    else:
                        # Convert all cells to strings
                        cleaned_rows.append([str(cell) if cell is not None else '' for cell in row])
                
                if cleaned_rows:
                    table['rows'] = cleaned_rows
                    valid_tables.append(table)
                    logger.debug(f"Valid table found with {len(cleaned_rows)} rows")
                else:
                    logger.warning("No valid rows found in table")
            
            if valid_tables:
                logger.info(f"Successfully parsed {len(valid_tables)} valid tables")
                # Log the first few rows of each table for debugging
                for i, table in enumerate(valid_tables):
                    logger.debug(f"Table {i+1} first row: {table['rows'][0]}")
            else:
                logger.warning("No valid tables found in response")
            
            return valid_tables
            
        except (SyntaxError, ValueError) as e:
            logger.error(f"Failed to parse GPT response: {e}")
            return []

    # Add these methods to PdfTableExtractor class
    
    def _correct_skew(self, image_path: str) -> str:
        """Correct image skew using OpenCV"""
        if not HAS_CV2:
            return image_path
        
        try:
            # Read image
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
            
            if lines is not None:
                # Calculate skew angle
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = np.degrees(theta)
                    if angle < 45:
                        angles.append(angle)
                    elif angle > 135:
                        angles.append(angle - 180)
                
                if angles:
                    skew_angle = np.median(angles)
                    
                    # Rotate if skew is significant
                    if abs(skew_angle) > 0.5:
                        height, width = img.shape[:2]
                        center = (width//2, height//2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                        rotated = cv2.warpAffine(img, rotation_matrix, (width, height), 
                                               flags=cv2.INTER_CUBIC, 
                                               borderMode=cv2.BORDER_REPLICATE)
                        
                        # Save rotated image
                        output_path = image_path.replace('.png', '_deskewed.png')
                        cv2.imwrite(output_path, rotated)
                        return output_path
            
            return image_path
            
        except Exception as e:
            logger.warning(f"Skew correction failed: {e}")
            return image_path

    def _preprocess_image_for_ocr(self, image_path: str) -> str:
        """Enhance image for better OCR results"""
        if not HAS_CV2:
            return image_path
        
        try:
            # Read image
            img = cv2.imread(image_path)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations
            kernel = np.ones((1, 1), np.uint8)
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Save enhanced image
            output_path = image_path.replace('.png', '_enhanced.png')
            cv2.imwrite(output_path, morph)
            return output_path
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image_path

    def _perform_enhanced_ocr(self, image_path: str) -> str:
        """Enhanced OCR with multiple attempts and better error handling"""
        best_text = ""
        best_confidence = 0
        
        logger.info(f"Attempting OCR with multiple PSM modes on {image_path}")
        
        # Try different PSM modes and OCR configurations
        for psm_mode in [6, 3, 4, 11, 1]:  # Added PSM mode 1 for auto page segmentation
            try:
                # Only use LSTM engines, remove legacy engine option
                configs = [
                    f'--psm {psm_mode} --oem 3',  # LSTM only
                    f'--psm {psm_mode} --oem 1'   # Neural nets LSTM only
                ]
                
                for config in configs:
                    logger.info(f"Trying OCR with config: {config}")
                    
                    # Open and preprocess image
                    img = Image.open(image_path)
                    
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    ocr_data = pytesseract.image_to_data(
                        img,
                        output_type=pytesseract.Output.DICT,
                        config=config
                    )
                    
                    # Get confidences and text
                    confidences = [float(conf) for conf, text in 
                                 zip(ocr_data['conf'], ocr_data['text']) 
                                 if str(conf).isdigit() and text.strip()]
                    
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    text = ' '.join([word for word in ocr_data['text'] if word.strip()])
                    
                    logger.info(f"OCR result - Length: {len(text)}, Confidence: {avg_confidence:.2f}")
                    
                    if text and avg_confidence > best_confidence:
                        best_text = text
                        best_confidence = avg_confidence
                        logger.info(f"New best text found (confidence: {best_confidence:.2f})")
                    
                    if best_confidence > 80:
                        logger.info("Found high-confidence text, stopping early")
                        break
                
            except Exception as e:
                logger.warning(f"OCR attempt failed with PSM {psm_mode}: {e}")
                continue
        
        if best_text:
            logger.info(f"OCR completed - Final text length: {len(best_text)}, confidence: {best_confidence:.2f}")
        else:
            logger.warning("OCR failed to extract any text")
        
        return best_text if best_confidence > 30 else ""

    def _pdf_to_image(self, file_path: str, page_num: int) -> Optional[Image.Image]:
        """
        Convert a single PDF page to a PIL Image. Try pdf2image, fallback to PyMuPDF.
        """
        try:
            images = convert_from_path(file_path, first_page=page_num, last_page=page_num)
            if images:
                return images[0]
        except Exception as e:
            logger.warning(f"pdf2image failed on page {page_num}: {e}")
        # Fallback to PyMuPDF
        try:
            import fitz
            doc = fitz.open(file_path)
            page = doc[page_num - 1]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return img
        except Exception as e:
            logger.warning(f"PyMuPDF fallback failed on page {page_num}: {e}")
        return None

    # ------------------ Other Handlers ------------------

    def _handle_docx(self, file_path: str) -> Dict[str, Any]:
        try:
            output_dir = os.path.join(os.path.dirname(file_path), "output")
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            with open(file_path, "rb") as docx_file:
                image_handler = mammoth.images.inline(lambda image: {
                    "src": self._save_docx_image(image, images_dir, os.path.basename(file_path))
                })
                result = mammoth.convert_to_markdown(
                    docx_file,
                    convert_image=image_handler
                )
                return {
                    'text': result.value,
                    'images': [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))],
                    'output_dir': output_dir
                }
        except Exception as e:
            logger.exception(f"Error processing DOCX: {e}")
            return {
                'text': f"Error processing file: {str(e)}",
                'images': [],
                'output_dir': os.path.join(os.path.dirname(file_path), "output")
            }

    def _save_docx_image(self, image, images_dir: str, doc_name: str) -> str:
        try:
            ext = '.png'
            image_filename = f"{os.path.splitext(doc_name)[0]}_{len(os.listdir(images_dir))}{ext}"
            image_path = os.path.join(images_dir, image_filename)
            with open(image_path, 'wb') as f:
                f.write(image.open().read())
            return os.path.join('images', image_filename).replace('\\', '/')
        except Exception as e:
            logger.warning(f"Failed to save DOCX image: {e}")
            return ""

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

    def _format_gpt4v_tables(self, tables: List[Dict[str, Any]]) -> List[str]:
        """Format tables from GPT-4V response into proper markdown."""
        if not tables:
            return []
        
        formatted_lines = []
        
        for table in tables:
            if not isinstance(table, dict) or 'rows' not in table:
                continue
            
            rows = table.get('rows', [])
            if not rows or not any(row for row in rows if any(cell for cell in row)):
                continue
            
            # Add title if present with proper spacing
            title = table.get('title', '').strip()
            if title:
                formatted_lines.extend(['', '', f"### {title}", ''])
            
            # Calculate column widths and normalize row lengths
            max_cols = max(len(row) for row in rows)
            normalized_rows = [row + [''] * (max_cols - len(row)) for row in rows]
            
            # Calculate optimal column widths
            col_widths = [0] * max_cols
            for row in normalized_rows:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(cell).strip()))
            
            # Determine column alignments
            alignments = []
            for i in range(max_cols):
                # Check if column appears to be numeric/currency
                is_numeric = all(
                    str(row[i]).strip().replace('.', '').replace('-', '').replace(',', '').replace('$', '').isdigit() 
                    for row in normalized_rows[1:]  # Skip header
                    if i < len(row) and row[i] and str(row[i]).strip()
                )
                # Check if column is a short code/unit
                is_short = all(
                    len(str(row[i]).strip()) <= 4 
                    for row in normalized_rows[1:]  # Skip header
                    if i < len(row) and row[i]
                )
                # Long text columns should be left-aligned
                is_long_text = any(
                    len(str(row[i]).strip()) > 20
                    for row in normalized_rows
                    if i < len(row) and row[i]
                )
                
                if is_numeric or is_short:
                    alignments.append('right')
                elif is_long_text:
                    alignments.append('left')
                else:
                    alignments.append('center')
            
            # Format header row
            header_cells = []
            for i, cell in enumerate(normalized_rows[0]):
                cell_str = str(cell).strip()
                if alignments[i] == 'right':
                    header_cells.append(cell_str.rjust(col_widths[i]))
                elif alignments[i] == 'left':
                    header_cells.append(cell_str.ljust(col_widths[i]))
                else:
                    header_cells.append(cell_str.center(col_widths[i]))
            formatted_lines.append('| ' + ' | '.join(header_cells) + ' |')
            
            # Add separator with alignment
            sep_cells = []
            for i, width in enumerate(col_widths):
                width = max(width, 3)  # Minimum width
                if alignments[i] == 'right':
                    sep_cells.append('-' * (width - 1) + ':')
                elif alignments[i] == 'left':
                    sep_cells.append(':' + '-' * (width - 1))
                else:
                    sep_cells.append(':' + '-' * (width - 2) + ':')
            formatted_lines.append('| ' + ' | '.join(sep_cells) + ' |')
            
            # Format data rows
            for row in normalized_rows[1:]:
                cells = []
                for i, cell in enumerate(row):
                    cell_str = str(cell).strip()
                    if alignments[i] == 'right':
                        cells.append(cell_str.rjust(col_widths[i]))
                    elif alignments[i] == 'left':
                        cells.append(cell_str.ljust(col_widths[i]))
                    else:
                        cells.append(cell_str.center(col_widths[i]))
                formatted_lines.append('| ' + ' | '.join(cells) + ' |')
            
            # Add spacing after table
            formatted_lines.extend(['', ''])
        
        return formatted_lines

    def _check_image_quality(self, image_path: str) -> float:
        """Check image quality using various metrics. Returns score between 0-1."""
        if not HAS_CV2:
            return 1.0  # Default to assuming good quality if OpenCV not available
            
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return 0.0
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            scores = []
            
            # 1. Check resolution
            height, width = img.shape[:2]
            resolution_score = min(1.0, (width * height) / (800 * 600))
            scores.append(resolution_score)
            
            # 2. Check contrast
            contrast = gray.std()
            contrast_score = min(1.0, contrast / 50)  # Normalize to 0-1
            scores.append(contrast_score)
            
            # 3. Check blur
            laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(1.0, laplacian / 500)  # Normalize to 0-1
            scores.append(blur_score)
            
            # 4. Check noise
            noise = cv2.fastNlMeansDenoising(gray)
            noise_diff = cv2.absdiff(gray, noise).mean()
            noise_score = 1.0 - min(1.0, noise_diff / 50)  # Normalize to 0-1
            scores.append(noise_score)
            
            # Calculate final score
            final_score = sum(scores) / len(scores)
            logger.debug(f"Image quality metrics - Resolution: {resolution_score:.2f}, "
                        f"Contrast: {contrast_score:.2f}, Blur: {blur_score:.2f}, "
                        f"Noise: {noise_score:.2f}, Final: {final_score:.2f}")
            
            return final_score
            
        except Exception as e:
            logger.warning(f"Image quality check failed: {e}")
            return 1.0  # Default to assuming good quality on error

# --------------------------------------------------
# MarkItDown Class (Main)
# --------------------------------------------------

class MarkItDown:
    """Main class for converting documents to markdown."""

    def __init__(self, llm_client: Optional[Any] = None, max_gpt4v_cost: float = 2.0, 
                 table_strategy: Union[str, TableExtractionStrategy] = TableExtractionStrategy.AUTO):
        """Initialize with optional LLM client for advanced features.
        
        Args:
            llm_client: Optional OpenAI client or API key
            max_gpt4v_cost: Maximum cost allowed for GPT-4V calls
            table_strategy: Strategy for table extraction (auto, heuristic, layoutlm, or gpt4v)
        """
        # Convert string strategy to enum if needed
        if isinstance(table_strategy, str):
            try:
                table_strategy = TableExtractionStrategy(table_strategy.lower())
            except ValueError:
                logger.warning(f"Invalid table strategy '{table_strategy}', falling back to AUTO")
                table_strategy = TableExtractionStrategy.AUTO

        self.table_strategy = table_strategy
        self.pdf_table_extractor = PdfTableExtractor(
            llm_client=llm_client, 
            max_gpt4v_cost=max_gpt4v_cost
        )
        
    def convert_file(self, file_path: str, output_dir: Optional[str] = None) -> str:
        """Convert a file to markdown format."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Create output directory if needed
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(file_path), "output")
        os.makedirs(output_dir, exist_ok=True)

        # Create images directory
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Detect file type
        file_type = self._detect_file_type(file_path)
        logger.info(f"File type: {file_type}")

        # Convert based on file type
        try:
            if file_type == '.pdf':
                return self._handle_pdf(file_path, output_dir)
            elif file_type == '.docx':
                return self._handle_docx(file_path, output_dir)
            else:
                raise UnsupportedFormatException(f"Unsupported file type: {file_type}")
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise FileConversionException(f"Failed to convert {file_path}: {str(e)}")

    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type using extension and magic numbers."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext:
            return ext
        
        # Fallback to magic numbers
        try:
            mime_type = puremagic.from_file(file_path)
            return mimetypes.guess_extension(mime_type) or ''
        except:
            return ''

    def _handle_pdf(self, pdf_path: str, output_dir: str) -> str:
        """Process PDF files."""
        logger.info("Converting...")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_text = []
                
                for page_index, page in enumerate(pdf.pages, 1):
                    try:
                        page_output, is_incomplete, page_img_path = self._process_single_pdf_page(
                            pdf_path, page_index, page, 
                            os.path.join(output_dir, "images"),
                            output_dir
                        )
                        all_text.append(page_output)
                        
                    except Exception as e:
                        logger.error(f"Processing PDF {pdf_path} failed: {e}")
                        logger.error(traceback.format_exc())
                        all_text.append(f"*Error processing page {page_index}*\n")
                
                # Combine all text
                combined_text = '\n'.join(all_text)
                
                # Write output
                output_path = os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(pdf_path))[0] + '.md'
                )
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(combined_text)
                
                logger.info(f"Conversion complete")
                logger.info(f"Saved to: {output_path}")
                
                # Return the content instead of the path
                return combined_text
                
        except Exception as e:
            logger.error(f"Failed to process PDF: {e}")
            raise

    def _process_single_pdf_page(self, pdf_path: str, page_index: int, page: pdfplumber.page.Page,
                               images_dir: str, output_dir: str) -> Tuple[str, bool, str]:
        """Process a single PDF page with enhanced formatting."""
        
        # 1. Extract all content first
        raw_text = page.extract_text() or ""
        logger.info(f"Page {page_index} raw text length: {len(raw_text)}")
        
        content = {
            'text': raw_text,
            'tables': [],
            'images': []
        }
        
        # 2. Process images and try OCR if needed
        enhanced_path = ""
        pil_img = self.pdf_table_extractor._pdf_to_image(pdf_path, page_index)
        if pil_img:
            raw_path = os.path.join(images_dir, f"page_{page_index}.png")
            pil_img.save(raw_path, dpi=(300, 300))
            logger.info(f"Page {page_index}: Saved initial image at {raw_path}")
            
            # ALWAYS check and correct orientation first
            oriented_path = self.pdf_table_extractor._detect_and_correct_orientation(raw_path)
            if oriented_path != raw_path:
                logger.info(f"Page {page_index}: Corrected page orientation")
                deskewed_path = oriented_path
            else:
                deskewed_path = self.pdf_table_extractor._correct_skew(raw_path)
                if deskewed_path != raw_path:
                    logger.info(f"Page {page_index}: Corrected page skew")
            
            # Then enhance for OCR and GPT
            enhanced_path = self.pdf_table_extractor._preprocess_image_for_ocr(deskewed_path)
            logger.info(f"Page {page_index}: Enhanced image for processing")
            
            if enhanced_path:
                rel_path = os.path.relpath(enhanced_path, output_dir).replace('\\', '/')
                content['images'].append({
                    'path': rel_path,
                    'page': page_index
                })
                
                # Try OCR if text is missing
                if not content['text'].strip():
                    ocr_text = self.pdf_table_extractor._perform_enhanced_ocr(enhanced_path)
                    if ocr_text:
                        content['text'] = ocr_text
                
                # Check for tables in the properly oriented and enhanced image
                logger.info(f"Page {page_index}: Checking for tables in enhanced image")
                try:
                    gpt4v_result = self.pdf_table_extractor._extract_with_gpt4v(
                        enhanced_path,
                        extract_tables_only=True
                    )
                    # Log the cost after GPT-4V call
                    logger.info(f"Page cost: ${self.pdf_table_extractor.last_cost:.4f}")
                    
                    if isinstance(gpt4v_result, list):
                        content['tables'] = gpt4v_result
                        logger.info(f"Page {page_index}: Successfully extracted {len(gpt4v_result)} tables from GPT-4V")
                except Exception as e:
                    logger.error(f"GPT-4V table extraction failed on page {page_index}: {e}")
        
        # 3. Format everything as markdown
        lines = []
        
        # Header
        lines.extend([
            '---',
            f"## Page {page_index}",
            ''
        ])
        
        # Text content
        if content['text'].strip():
            formatted_text = content['text']  # Basic formatting for now
            if formatted_text:
                lines.append(formatted_text)
                lines.append('')
        
        # Tables
        if content['tables']:
            for table in content['tables']:
                table_lines = self.pdf_table_extractor._format_gpt4v_tables([table])
                lines.extend(table_lines)
        
        # Images
        for img in content['images']:
            lines.extend([
                f"![Page {img['page']} - Enhanced scan]({img['path']})",
                f"*Page {img['page']} of document*",
                ''
            ])
        
        # Return with completion status
        text_content = '\n'.join(lines)
        is_incomplete = not content['text'] or len(content['text'].strip()) < 10
        return text_content, is_incomplete, enhanced_path

    def _handle_docx(self, file_path: str) -> Dict[str, Any]:
        try:
            output_dir = os.path.join(os.path.dirname(file_path), "output")
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            with open(file_path, "rb") as docx_file:
                image_handler = mammoth.images.inline(lambda image: {
                    "src": self._save_docx_image(image, images_dir, os.path.basename(file_path))
                })
                result = mammoth.convert_to_markdown(
                    docx_file,
                    convert_image=image_handler
                )
                return {
                    'text': result.value,
                    'images': [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))],
                    'output_dir': output_dir
                }
        except Exception as e:
            logger.exception(f"Error processing DOCX: {e}")
            return {
                'text': f"Error processing file: {str(e)}",
                'images': [],
                'output_dir': os.path.join(os.path.dirname(file_path), "output")
            }

    def _save_docx_image(self, image, images_dir: str, doc_name: str) -> str:
        try:
            ext = '.png'
            image_filename = f"{os.path.splitext(doc_name)[0]}_{len(os.listdir(images_dir))}{ext}"
            image_path = os.path.join(images_dir, image_filename)
            with open(image_path, 'wb') as f:
                f.write(image.open().read())
            return os.path.join('images', image_filename).replace('\\', '/')
        except Exception as e:
            logger.warning(f"Failed to save DOCX image: {e}")
            return ""

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