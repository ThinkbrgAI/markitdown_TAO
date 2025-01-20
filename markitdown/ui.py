from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QProgressBar, QCheckBox,
    QFileDialog, QComboBox, QTextEdit, QMessageBox, QGroupBox, QDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import os
import keyring
from typing import List, Dict, Tuple, Optional
from ._markitdown import MarkItDown, TableExtractionStrategy
import pdfplumber.page

class ConversionWorker(QThread):
    """Worker thread for folder conversion"""
    overall_progress = pyqtSignal(int)
    file_progress = pyqtSignal(int)
    current_file = pyqtSignal(str)
    status_update = pyqtSignal(str)
    cost_update = pyqtSignal(float)  # New signal for cost updates
    finished = pyqtSignal()
    
    def __init__(self, input_dir: str, output_dir: str, options: Dict):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.options = options
        self.total_cost = 0.0  # Track total cost
        
    def run(self):
        try:
            # Initialize MarkItDown with options
            md = MarkItDown(
                table_strategy=self.options.get('table_strategy', 'heuristic'),
                llm_client=self.options.get('openai_key')
            )
            
            # Get list of files to process
            files = []
            for root, _, filenames in os.walk(self.input_dir):
                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in ['.pdf', '.docx', '.xlsx', '.pptx', '.html', '.msg']:
                        files.append(os.path.join(root, filename))
            
            # Process each file
            for i, file_path in enumerate(files):
                try:
                    # Update current file
                    rel_path = os.path.relpath(file_path, self.input_dir)
                    self.current_file.emit(rel_path)
                    self.status_update.emit(f"Converting: {rel_path}")
                    
                    # Convert file - returns string now, not ConversionResult
                    result = md.convert_file(file_path, self.output_dir)
                    
                    # Update total cost if available
                    page_cost = md.pdf_table_extractor.last_cost
                    self.total_cost += page_cost
                    self.cost_update.emit(self.total_cost)
                    self.status_update.emit(f"Page cost: ${page_cost:.4f}")
                    
                    # Create relative output path
                    rel_output = os.path.relpath(file_path, self.input_dir)
                    output_path = os.path.join(self.output_dir, rel_output)
                    output_path = os.path.splitext(output_path)[0] + '.md'
                    
                    # Create output directory if needed
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Save result (now a string)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(result)
                    
                    self.status_update.emit(f"Completed: {rel_path}")
                    
                except Exception as e:
                    self.status_update.emit(f"Error converting {rel_path}: {str(e)}")
                
                # Update progress
                self.overall_progress.emit(int((i + 1) * 100 / len(files)))
                self.file_progress.emit(100)  # Reset file progress
            
            self.status_update.emit(f"Processed {len(files)} files")
            self.status_update.emit(f"Total OpenAI cost: ${self.total_cost:.4f}")
            self.finished.emit()
            
        except Exception as e:
            self.status_update.emit(f"Worker error: {str(e)}")

    def _process_single_pdf_page(self, pdf_path: str, page_index: int, page: pdfplumber.page.Page, images_dir: str, output_dir: str) -> Tuple[str, bool, str]:
        # ... existing code ...

        if gpt4v_result and isinstance(gpt4v_result, list):
            # Format tables as markdown
            table_md = []
            for table in gpt4v_result:
                if not isinstance(table, dict) or 'rows' not in table:  # Skip invalid tables
                    continue
                rows = table.get('rows', [])
                if not rows or not any(row for row in rows if any(cell for cell in row)):  # Skip empty tables
                    continue
                
                table_md.append('')  # Blank line before table
                
                # Add table title if present
                title = table.get('title', '').strip()
                if title:
                    table_md.append(f"**{title}**\n")
                
                # Get max column count for proper alignment
                max_cols = max(len(row) for row in rows)
                
                # Format each row
                for i, row in enumerate(rows):
                    # Pad row to max columns if needed
                    padded_row = row + [''] * (max_cols - len(row))
                    table_md.append('| ' + ' | '.join(str(cell) for cell in padded_row) + ' |')
                    
                    # Add separator after header
                    if i == 0:
                        table_md.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
                
                table_md.append('')  # Blank line after table

class MarkItDownUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MarkItDown")
        self.converter = MarkItDown()
        
        # Create main widget and layout
        main_widget = QWidget()
        layout = QVBoxLayout()
        
        # Add input folder selection
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input Folder:"))
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Select folder containing documents...")
        browse_input_button = QPushButton("Browse")
        browse_input_button.clicked.connect(self.browse_input)
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(browse_input_button)
        layout.addLayout(input_layout)
        
        # Add output folder selection
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder:"))
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Select output folder (default: input_folder/output)")
        browse_output_button = QPushButton("Browse")
        browse_output_button.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(browse_output_button)
        layout.addLayout(output_layout)
        
        # Add settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()
        
        # OpenAI settings
        openai_layout = QHBoxLayout()
        openai_layout.addWidget(QLabel("OpenAI Key:"))
        self.openai_key = QLineEdit()
        self.openai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.openai_key.setText(keyring.get_password("markitdown", "openai_key") or "")
        openai_layout.addWidget(self.openai_key)
        
        # Save key button
        save_key_button = QPushButton("Save Key")
        save_key_button.clicked.connect(self.save_openai_key)
        openai_layout.addWidget(save_key_button)
        settings_layout.addLayout(openai_layout)
        
        # Table extraction strategy
        table_layout = QHBoxLayout()
        table_layout.addWidget(QLabel("Table Extraction:"))
        self.table_strategy = QComboBox()
        self.table_strategy.addItems([
            "Heuristic (Default)",
            "LayoutLM (ML-based)",
            "GPT-4V (OpenAI)",
            "Auto (Try all)"
        ])
        # Connect strategy change to handle OpenAI dependency
        self.table_strategy.currentTextChanged.connect(self.on_strategy_change)
        table_layout.addWidget(self.table_strategy)
        settings_layout.addLayout(table_layout)
        
        # OCR settings
        ocr_layout = QHBoxLayout()
        self.enable_ocr = QCheckBox("Enable OCR")
        self.enable_ocr.setChecked(True)
        ocr_layout.addWidget(self.enable_ocr)
        settings_layout.addLayout(ocr_layout)
        
        # Add GPT Prompt Editor
        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel("GPT-4V Prompt:"))
        self.prompt_edit = QPushButton("Edit Prompt")
        self.prompt_edit.clicked.connect(self.edit_prompt)
        prompt_layout.addWidget(self.prompt_edit)
        settings_layout.addLayout(prompt_layout)
        
        # Add settings to main layout
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Add convert button
        convert_button = QPushButton("Convert All")
        convert_button.clicked.connect(self.convert_folder)
        layout.addWidget(convert_button)
        
        # Add progress bars
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        # Overall progress
        progress_layout.addWidget(QLabel("Overall Progress:"))
        self.overall_progress = QProgressBar()
        progress_layout.addWidget(self.overall_progress)
        
        # Current file progress
        progress_layout.addWidget(QLabel("Current File:"))
        self.current_file_label = QLabel("")
        progress_layout.addWidget(self.current_file_label)
        self.file_progress = QProgressBar()
        progress_layout.addWidget(self.file_progress)
        
        # Add cost tracking
        progress_layout.addWidget(QLabel("OpenAI Cost:"))
        self.cost_label = QLabel("$0.00")
        progress_layout.addWidget(self.cost_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Add status area
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)
        
        # Set up the main widget
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        
        # Set up drag and drop
        self.setAcceptDrops(True)
    
    def browse_input(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder_path:
            self.input_path.setText(folder_path)
            # Set default output directory
            default_output = os.path.join(folder_path, "output")
            self.output_path.setText(default_output)
    
    def browse_output(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder_path:
            self.output_path.setText(folder_path)
    
    def convert_folder(self):
        input_dir = self.input_path.text()
        if not input_dir:
            self.status_text.append("Please select an input folder first.")
            return
        
        try:
            # Get settings
            strategy_map = {
                "Heuristic (Default)": TableExtractionStrategy.HEURISTIC,
                "LayoutLM (ML-based)": TableExtractionStrategy.LAYOUTLM,
                "GPT-4V (OpenAI)": TableExtractionStrategy.GPT4V,
                "Auto (Try all)": TableExtractionStrategy.AUTO
            }
            table_strategy = strategy_map[self.table_strategy.currentText()]
            
            # Validate OpenAI key if needed
            if table_strategy in [TableExtractionStrategy.GPT4V, TableExtractionStrategy.AUTO]:
                if not self.openai_key.text():
                    self.status_text.append("Error: OpenAI key required for selected strategy")
                    return
            
            # Get output directory
            output_dir = self.output_path.text() or os.path.join(input_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Start conversion worker
            self.worker = ConversionWorker(
                input_dir=input_dir,
                output_dir=output_dir,
                options={
                    'table_strategy': table_strategy,
                    'enable_ocr': self.enable_ocr.isChecked(),
                    'openai_key': self.openai_key.text()
                }
            )
            
            # Connect signals
            self.worker.overall_progress.connect(self.overall_progress.setValue)
            self.worker.file_progress.connect(self.file_progress.setValue)
            self.worker.current_file.connect(self.current_file_label.setText)
            self.worker.status_update.connect(self.status_text.append)
            self.worker.finished.connect(self.conversion_finished)
            
            # Connect cost signal
            self.worker.cost_update.connect(self.update_cost)
            
            # Start conversion
            self.worker.start()
            self.status_text.append(f"Starting conversion of {input_dir}...")
            
        except Exception as e:
            self.status_text.append(f"Error: {str(e)}")
            self.status_text.append("---\n")
    
    def conversion_finished(self):
        self.status_text.append("Conversion completed!")
        self.status_text.append("---\n")
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        if files:
            self.file_path.setText(files[0])
            # Set default output directory
            default_output = os.path.join(os.path.dirname(files[0]), "output")
            self.output_path.setText(default_output) 
    
    def save_openai_key(self):
        """Save OpenAI key to system keyring"""
        key = self.openai_key.text()
        if key:
            keyring.set_password("markitdown", "openai_key", key)
            self.status_text.append("OpenAI key saved successfully")
        else:
            self.status_text.append("Please enter an OpenAI key") 
    
    def on_strategy_change(self, strategy: str):
        """Handle table strategy changes"""
        needs_openai = strategy in ["GPT-4V (OpenAI)", "Auto (Try all)"]
        if needs_openai and not self.openai_key.text():
            self.status_text.append("Warning: OpenAI key required for this strategy")
            # Optionally, revert to default if no key
            # self.table_strategy.setCurrentText("Heuristic (Default)") 
    
    def update_cost(self, cost: float):
        """Update the cost display"""
        self.cost_label.setText(f"${cost:.4f}") 
    
    def edit_prompt(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit GPT-4V Prompt")
        layout = QVBoxLayout()

        # Add text editor
        editor = QTextEdit()
        editor.setPlainText(self.converter.pdf_table_extractor.PROMPT_SYSTEM_GPT4V)
        layout.addWidget(editor)

        # Add buttons
        button_box = QHBoxLayout()
        reset_btn = QPushButton("Reset to Default")
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")

        reset_btn.clicked.connect(lambda: editor.setPlainText(self.converter.pdf_table_extractor.DEFAULT_PROMPT))
        save_btn.clicked.connect(lambda: self.save_prompt(editor.toPlainText(), dialog))
        cancel_btn.clicked.connect(dialog.reject)

        button_box.addWidget(reset_btn)
        button_box.addWidget(save_btn)
        button_box.addWidget(cancel_btn)
        layout.addLayout(button_box)

        dialog.setLayout(layout)
        dialog.resize(800, 600)
        dialog.exec()

    def save_prompt(self, new_prompt: str, dialog: QDialog):
        if hasattr(self.converter, 'pdf_table_extractor'):
            self.converter.pdf_table_extractor.PROMPT_SYSTEM_GPT4V = new_prompt
            self.status_text.append("GPT-4V prompt updated")
            dialog.accept() 

def process_files(input_path: str, output_dir: Optional[str] = None, 
                 table_strategy: str = "auto", max_gpt4v_cost: float = 2.0) -> None:
    """Process all PDF files in a directory or a single file."""
    
    # Initialize converter
    converter = MarkItDown(
        llm_client=os.getenv("OPENAI_API_KEY"),
        max_gpt4v_cost=max_gpt4v_cost,
        table_strategy=table_strategy
    )
    
    total_cost = 0.0
    files_processed = 0
    
    print(f"Starting conversion of {input_path}...")
    
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                if f.lower().endswith('.pdf')]
    
    for file_path in files:
        print(f"Converting: {os.path.basename(file_path)}")
        try:
            # Change this line from converter.convert to converter.convert_file
            output_path = converter.convert_file(file_path, output_dir)
            files_processed += 1
            # Add cost from table extractor
            total_cost += converter.pdf_table_extractor.last_cost
        except Exception as e:
            print(f"Error converting {os.path.basename(file_path)}: {str(e)}")
    
    print(f"Processed {files_processed} files")
    print(f"Total OpenAI cost: ${total_cost:.4f}")
    print("Conversion completed!") 