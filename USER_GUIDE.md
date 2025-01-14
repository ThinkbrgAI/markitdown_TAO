# Quick Start Guide for MarkItDown (Windows)

## Setup Steps (First Time Only)
1. Open PowerShell
2. Navigate to your markitdown folder:
   ```
   cd A:\markitdown
   ```
3. Activate the virtual environment:
   ```
   .\venv\Scripts\activate
   ```
   - You'll know it worked when you see `(venv)` at the start of your command line

## Converting Files
1. Make sure you're in the markitdown folder and the virtual environment is active (you should see `(venv)` at the start of the line)
2. Use this command to convert your file:
   ```
   python -m markitdown your_file.docx > output.md
   ```
   Replace `your_file.docx` with your actual file name
   Replace `output.md` with what you want to name the converted file

## Batch Converting Multiple Files
1. Put all your PDF files in one folder (for example: `A:\my_pdfs`)
2. Make sure you're in the markitdown folder and virtual environment is active
3. Copy and paste this entire command (it will convert all PDFs in the folder):
   ```
   Get-ChildItem "A:\my_pdfs\*.pdf" | ForEach-Object { python -m markitdown $_.FullName > "$($_.DirectoryName)\$($_.BaseName).md" }
   ```
   (Replace `A:\my_pdfs` with your actual folder path)

This will:
- Find all PDF files in the specified folder
- Convert each one to markdown
- Save each markdown file in the same folder with the same name (but .md extension)
- Work even if your filenames have spaces

## Examples
Convert a single PDF:
```
python -m markitdown example.pdf > example.md
```

Convert a Word document:
```
python -m markitdown mydoc.docx > mydoc.md
```

## Tips
- Your input file can be in a different folder. Just include the full path:
  ```
  python -m markitdown "C:\Users\YourName\Documents\file.pdf" > output.md
  ```
- If a file path has spaces, put it in quotes:
  ```
  python -m markitdown "A:\My Documents\my file.docx" > output.md
  ```
- To close the virtual environment when you're done:
  ```
  deactivate
  ```

## Supported File Types
- PDF (.pdf)
- Word Documents (.docx)
- Excel Spreadsheets (.xlsx)
- PowerPoint Presentations (.pptx)
- HTML files (.html, .htm)
- Text files (.txt)
- Images (.jpg, .jpeg, .png)
- Audio files (.mp3, .wav) - requires additional setup

## Batch Convert Different File Types
To convert different file types (like .docx or .xlsx), modify the batch command:
- For Word documents:
  ```
  Get-ChildItem "A:\my_docs\*.docx" | ForEach-Object { python -m markitdown $_.FullName > "$($_.DirectoryName)\$($_.BaseName).md" }
  ```
- For Excel files:
  ```
  Get-ChildItem "A:\my_excel\*.xlsx" | ForEach-Object { python -m markitdown $_.FullName > "$($_.DirectoryName)\$($_.BaseName).md" }
  ```