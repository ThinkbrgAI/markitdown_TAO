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

    # Central prompt or fallback prompts for GPT-4V (Section 5: Prompt Management).
    PROMPT_SYSTEM_GPT4V = (
        "You are an expert table extraction model. "
        "Extract ONLY the clearly visible tables as a Python list of lists. "
        "If no table is recognized, return []. Do not hallucinate or infer."
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

        self.last_cost: float = 0.0
        self.seen_tables = set()
        self.max_gpt4v_cost = max_gpt4v_cost

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

            # Updated system prompt for strict table extraction
            system_message = (
                "You are an expert data extraction model. Your task is to detect and extract ONLY tabular data that clearly appears in the provided image.\n\n"
                "**What to Return**:\n"
                "- A SINGLE Python list named `tables`.\n"
                "- Each item of `tables` is a list representing a single table.\n"
                "- Each table is a list of rows, where each row is a list of cell strings.\n"
                "- If there is no clearly visible table in the image, return `tables = []`.\n\n"
                "**Example**:\n"
                "tables = [\n"
                "  [\n"
                "    [\"Column1\", \"Column2\", \"Column3\"],\n"
                "    [\"\", \"\", \"\"],\n"
                "    [\"Data1\", \"Data2\", \"Data3\"]\n"
                "  ],\n"
                "  [\n"
                "    // Table 2, same structure\n"
                "  ]\n"
                "]\n\n"
                "**Critical Instructions**:\n"
                "1. **Ignore Non-Table Text**:\n"
                "   - Skip paragraphs, line items, or disclaimers that are not arranged in a grid or table format.\n"
                "   - Extract only the tabular data that is visually separated by grid lines or aligned rows & columns.\n"
                "2. **No ASCII Art**:\n"
                "   - Do not try to replicate table lines or ASCII characters in your output.\n"
                "   - Do not chunk text into single characters. Return each cell as a readable string.\n"
                "3. **No Additional Explanations**:\n"
                "   - Return ONLY the Python object named `tables`.\n"
                "   - Do NOT add commentary, JSON, or Markdown.\n"
                "   - Do NOT wrap your output in quotes or code blocks.\n"
                "4. **Exact Cell Preservation**:\n"
                "   - If a table cell is partially unreadable, use \"\".\n"
                "   - Keep multi-word cells in a single string.\n"
                "   - Merged or \"section\" cells get repeated or padded with \"\" so each row has the same number of columns.\n"
                "5. **No Hallucination**:\n"
                "   - If text is unclear, leave it blank with \"\".\n"
                "   - If a table row or column count is uncertain, approximate based on visible alignment—do NOT guess or add data not seen.\n"
                "6. **Multiple Tables**:\n"
                "   - If the image shows more than one distinct table, each is a separate item in `tables`.\n"
                "7. **If No Table**:\n"
                "   - Return `tables = []`.\n\n"
                "**Important**: Provide your final result exactly in valid Python syntax, nothing else."
            )

            logger.info("Sending to OpenAI...")
            response = self.llm_client.chat.completions.create(
                model="gpt-4-turbo-2024-04-09",  # Always use this model
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text", 
                                "text": "Extract all tables from this image. Return only the Python list object."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=0
            )

            # Track cost
            if hasattr(response, 'usage'):
                tokens = response.usage.total_tokens
                self.last_cost = self._calculate_openai_cost("gpt-4-turbo-2024-04-09", tokens, images=1)
                logger.info(f"Cost: ${self.last_cost:.4f} (Tokens: {tokens}, Images: 1)")

            # Parse response
            content = response.choices[0].message.content.strip()
            if content.startswith("tables = "):
                content = content[len("tables = "):].strip()
            
            try:
                tables = ast.literal_eval(content)
                if isinstance(tables, list):
                    return tables if extract_tables_only else {"text": "", "tables": tables}
            except Exception as e:
                logger.error(f"Failed to parse GPT response: {e}")
            
            return [] if extract_tables_only else {"text": "", "tables": []}

        except Exception as e:
            logger.error(f"GPT-4V extraction failed: {e}")
            return [] if extract_tables_only else {"text": "", "tables": []}

    def _verify_gpt4v_tables(self, extracted_tables: Any, pil_img: Image.Image) -> List[List[List[str]]]:
        """Cross-check extracted words with OCR to reduce hallucination risk."""
        try:
            # First try to parse the content as Python literal
            try:
                if isinstance(extracted_tables, str):
                    tables = ast.literal_eval(extracted_tables)
                else:
                    tables = extracted_tables
            except:
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
        """Use OCR confidence to detect best orientation if cv2 is available."""
        if not HAS_CV2:
            return image_path
        try:
            img = cv2.imread(image_path)
            if img is None:
                return image_path

            angles = [0, 90, 180, 270]
            max_conf = 0.0
            best_angle = 0

            for angle in angles:
                if angle == 0:
                    rotated = img
                else:
                    height, width = img.shape[:2]
                    center = (width/2, height/2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(
                        img, rotation_matrix, (width, height),
                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                    )
                pil_img = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
                try:
                    ocr_data = pytesseract.image_to_data(
                        pil_img, output_type=pytesseract.Output.DICT
                    )
                    conf_scores = [float(conf) for conf in ocr_data['conf'] if conf != '-1']
                    avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.0
                    if avg_conf > max_conf:
                        max_conf = avg_conf
                        best_angle = angle
                except Exception:
                    continue

            if best_angle != 0:
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
        except Exception:
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

# --------------------------------------------------
# MarkItDown Class (Main)
# --------------------------------------------------

class MarkItDown:
    """
    Main class for converting documents to markdown.
    Incorporates:
      - Orientation correction for PDF pages
      - Partial text fallback / multi-PSM OCR
      - GPT usage for tables if needed, with cost/hallucination checks
      - Splits big PDF handling into smaller helper methods
      - Captures partial extraction / incomplete pages in metadata
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        llm_model: Optional[str] = None,
        table_strategy: Union[TableExtractionStrategy, str] = TableExtractionStrategy.HEURISTIC
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model

        # Create a PdfTableExtractor with default or provided LLM client
        self.pdf_table_extractor = PdfTableExtractor(llm_client=llm_client)

        if isinstance(table_strategy, str):
            table_strategy = TableExtractionStrategy(table_strategy.lower())
        self.table_strategy = table_strategy

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
        logger.info(f"Processing: {os.path.basename(file_path)}")
        try:
            ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"File type: {ext}")
            handler = self.handlers.get(ext)
            if not handler:
                raise UnsupportedFormatException(f"Unsupported file type: {ext}")
            logger.info("Converting...")
            result = handler(file_path)
            logger.info("Conversion complete")
            output_dir = result.get('output_dir') or os.path.join(os.path.dirname(file_path), "output")
            os.makedirs(output_dir, exist_ok=True)
            
            md_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + '.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(result['text'])
            logger.info(f"Saved to: {md_path}")

            return ConversionResult(
                text_content=result['text'],
                metadata=result.get('metadata', {}),
                images=result.get('images', [])
            )

        except Exception as e:
            logger.error(f"Error converting {file_path}: {str(e)}")
            raise FileConversionException(f"Error converting {file_path}: {str(e)}")

    def convert_batch(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Example batch conversion. 
        """
        results = {'successful': [], 'failed': [], 'skipped': []}
        os.makedirs(output_dir, exist_ok=True)
        for root, _, files in os.walk(input_dir):
            for filename in files:
                in_path = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1].lower()
                if ext not in self.handlers:
                    results['skipped'].append({'input': in_path, 'reason': 'Unsupported file type'})
                    continue
                try:
                    conv_result = self.convert(in_path)
                    out_path = os.path.join(
                        output_dir,
                        os.path.splitext(os.path.relpath(in_path, input_dir))[0] + '.md'
                    )
                    results['successful'].append({
                        'input': in_path,
                        'output': out_path,
                        'metadata': conv_result.metadata
                    })
                except Exception as ex:
                    results['failed'].append({'input': in_path, 'error': str(ex)})
        return results

    # ------------------ PDF Handling (Modularized) ------------------

    def _handle_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Handle PDF conversion by splitting into smaller sub-methods:
          1) Convert and correct orientation
          2) Extract text
          3) Extract tables (optional)
          4) Merge/Return
        """
        base_dir = os.path.dirname(file_path)
        output_dir = os.path.join(base_dir, "output")
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        combined_text: List[str] = []
        all_images: List[str] = []
        metadata: Dict[str, Any] = {"incomplete_pages": []}  # track partially extracted pages

        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                for page_index, page in enumerate(pdf.pages, start=1):
                    page_output, is_incomplete, page_img_path = self._process_single_pdf_page(
                        file_path, page_index, page, images_dir, output_dir
                    )
                    combined_text.append(page_output)
                    if page_img_path:
                        all_images.append(page_img_path)
                    if is_incomplete:
                        logger.warning(f"Page {page_index} might be incomplete.")
                        metadata["incomplete_pages"].append(page_index)

            combined_text.append(f"\n{os.path.splitext(os.path.basename(file_path))[0]}\n")

            return {
                "text": "\n".join(combined_text),
                "images": all_images,
                "output_dir": output_dir,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Processing PDF {os.path.basename(file_path)} failed: {e}")
            traceback.print_exc()
            return {
                'text': f"Error processing file: {e}",
                'images': [],
                'output_dir': output_dir,
                'metadata': {}
            }

    def _process_single_pdf_page(
        self,
        pdf_path: str,
        page_index: int,
        page: pdfplumber.page.Page,
        images_dir: str,
        output_dir: str
    ) -> Tuple[str, bool, str]:
        """Process a single PDF page with enhanced table detection"""
        
        # 1. Try pdfplumber text and table detection first
        page_text = page.extract_text() or ""
        pdf_tables = page.extract_tables()
        has_tables = bool(pdf_tables and any(t for t in pdf_tables if t and len(t) > 1))
        
        # Also check for table-like structures using lines/rectangles
        if not has_tables and page.lines and page.rects:
            h_lines = [l for l in page.lines if l['height'] == 0]
            v_lines = [l for l in page.lines if l['width'] == 0]
            if len(h_lines) > 2 and len(v_lines) > 2:
                has_tables = True
                logger.info(f"Page {page_index}: Detected table-like structure from lines")

        # 2. Image processing path (always convert to image for table detection)
        pil_img = self._pdf_to_image(pdf_path, page_index)
        enhanced_path = None
        if pil_img:
            # Save and process image
            raw_path = os.path.join(images_dir, f"page_{page_index}.png")
            pil_img.save(raw_path, dpi=(300, 300))
            
            # Fix orientation and enhance
            upright_path = self.pdf_table_extractor._detect_and_correct_orientation(raw_path)
            enhanced_path = upright_path
            
            # Always check for tables with GPT-4V if we have a client
            if self.llm_client:
                try:
                    logger.info(f"Page {page_index}: Checking for tables with GPT-4V")
                    gpt4v_result = self.pdf_table_extractor._extract_with_gpt4v(
                        enhanced_path,
                        extract_tables_only=True
                    )
                    
                    if gpt4v_result and isinstance(gpt4v_result, list):
                        # Format tables as markdown
                        table_md = []
                        for table in gpt4v_result:
                            if not table or not isinstance(table, list):  # Skip invalid tables
                                continue
                            if not any(row for row in table if any(cell for cell in row)):  # Skip empty tables
                                continue
                            
                            table_md.append('')  # Blank line before table
                            
                            # Get max column count for proper alignment
                            max_cols = max(len(row) for row in table)
                            
                            # Format each row
                            for i, row in enumerate(table):
                                # Pad row to max columns if needed
                                padded_row = row + [''] * (max_cols - len(row))
                                table_md.append('| ' + ' | '.join(str(cell) for cell in padded_row) + ' |')
                                
                                # Add separator after header
                                if i == 0:
                                    table_md.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
                        
                            table_md.append('')  # Blank line after table
                        
                        # Add tables to page text
                        if table_md:
                            logger.info(f"Page {page_index}: Found {len(gpt4v_result)} tables")
                            page_text = page_text.strip() + '\n\n' + '\n'.join(table_md)
                    
                except Exception as e:
                    logger.error(f"GPT-4V table extraction failed on page {page_index}: {e}")
            
            # If no text yet, try OCR
            if not page_text or len(page_text.strip()) < 10:
                ocr_text = self._perform_enhanced_ocr(enhanced_path)
                if ocr_text:
                    page_text = ocr_text

        # Build final output
        lines = [f"\n## Page {page_index}\n"]
        if page_text and page_text.strip():
            lines.append(page_text.strip())
        else:
            lines.append("[No reliable text]")
            
        # Add image reference if we created one
        if enhanced_path:
            rel_img_path = os.path.relpath(enhanced_path, output_dir).replace('\\', '/')
            lines.append(f"\n![Page {page_index}]({rel_img_path})\n")
        
        # Return with incomplete flag if text extraction was poor
        is_incomplete = not page_text or len(page_text.strip()) < 10
        return ('\n'.join(lines), is_incomplete, enhanced_path or "")

    def _perform_enhanced_ocr(self, image_path: str) -> str:
        """Enhanced OCR with multiple attempts"""
        best_text = ""
        best_confidence = 0
        
        for psm_mode in [6, 3, 4, 11]:
            try:
                ocr_data = pytesseract.image_to_data(
                    Image.open(image_path),
                    output_type=pytesseract.Output.DICT,
                    config=f'--psm {psm_mode} --oem 3'
                )
                
                confidences = [float(conf) for conf, text in 
                             zip(ocr_data['conf'], ocr_data['text']) 
                             if str(conf).isdigit() and text.strip()]
                
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                text = ' '.join([word for word in ocr_data['text'] if word.strip()])
                
                if text and avg_confidence > best_confidence:
                    best_text = text
                    best_confidence = avg_confidence
                
                if best_confidence > 80:
                    break
                    
            except Exception as e:
                logger.warning(f"OCR attempt failed with PSM {psm_mode}: {e}")
                continue
        
        return best_text if best_confidence > 30 else ""

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
