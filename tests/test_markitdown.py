import pytest
from markitdown import MarkItDown, ConversionError, TableExtractionStrategy
import os
from PIL import Image
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def test_basic_conversion():
    md = MarkItDown()
    result = md.convert('tests/fixtures/sample.txt')
    assert isinstance(result, ConversionResult)
    assert result.text_content

def test_unsupported_format():
    md = MarkItDown()
    with pytest.raises(ValueError, match="Unsupported file type"):
        md.convert('invalid.xyz')

def test_llm_integration():
    mock_llm = MockLLMClient()
    md = MarkItDown(llm_client=mock_llm, llm_model="test-model")
    result = md.convert('tests/fixtures/sample.jpg')
    assert "AI-Generated Description" in result.text_content 

class TestTableExtraction:
    @pytest.fixture
    def sample_tables(self):
        """Create sample table data for testing"""
        return [
            # Simple table
            [
                ["Header1", "Header2", "Header3"],
                ["Row1Col1", "Row1Col2", "Row1Col3"],
                ["Row2Col1", "Row2Col2", "Row2Col3"]
            ],
            # Financial table
            [
                ["Date", "Description", "Amount", "Balance"],
                ["2024-01-01", "Deposit", "$100.00", "$100.00"],
                ["2024-01-02", "Withdrawal", "-$50.00", "$50.00"]
            ]
        ]

    @pytest.fixture
    def sample_pdf(self, tmp_path, sample_tables):
        """Create a sample PDF with tables for testing"""
        pdf_path = tmp_path / "test.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Add a title
        elements.append(Paragraph("Test Document with Tables", styles['Heading1']))
        elements.append(Spacer(1, 12))

        # Add simple table
        simple_table = Table(sample_tables[0])
        simple_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(simple_table)
        elements.append(Spacer(1, 20))

        # Add financial table
        financial_table = Table(sample_tables[1])
        financial_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            # Right-align amount and balance columns
            ('ALIGN', (2, 1), (3, -1), 'RIGHT'),
        ]))
        elements.append(financial_table)

        # Build PDF
        doc.build(elements)
        return str(pdf_path)

    def test_pdf_table_extraction(self, sample_pdf):
        """Test table extraction from PDF"""
        md = MarkItDown(table_strategy=TableExtractionStrategy.AUTO)
        result = md.convert(sample_pdf)
        
        # Check if tables were extracted
        assert "Header1" in result.text_content
        assert "Header2" in result.text_content
        assert "Header3" in result.text_content
        
        # Check financial table
        assert "Date" in result.text_content
        assert "Description" in result.text_content
        assert "Amount" in result.text_content
        assert "Balance" in result.text_content
        
        # Check formatting
        assert "|" in result.text_content  # Table borders
        assert "---" in result.text_content  # Header separator
        
        # Check alignment
        table_lines = result.text_content.split('\n')
        amount_lines = [line for line in table_lines if "$" in line]
        for line in amount_lines:
            # Check if dollar amounts are right-aligned
            assert any(cell.strip().startswith('$') for cell in line.split('|'))

    def test_pdf_complex_layout(self, tmp_path):
        """Test extraction from PDF with complex layout"""
        pdf_path = tmp_path / "complex.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        # Add text before table
        elements.append(Paragraph("Monthly Report", styles['Heading1']))
        elements.append(Paragraph("This document contains important financial data.", styles['Normal']))
        elements.append(Spacer(1, 12))

        # Add nested tables
        inner_table = Table([
            ['Subtotal', '$500'],
            ['Tax', '$50'],
            ['Total', '$550']
        ])
        inner_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ]))

        outer_table = Table([
            ['Category', 'Details', 'Summary'],
            ['Electronics', 'Laptops and Phones', inner_table],
            ['Furniture', 'Chairs and Desks', '$300']
        ])
        outer_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ]))
        
        elements.append(outer_table)
        doc.build(elements)

        # Test extraction
        md = MarkItDown(table_strategy=TableExtractionStrategy.AUTO)
        result = md.convert(str(pdf_path))
        
        # Verify complex layout handling
        assert "Monthly Report" in result.text_content
        assert "Category" in result.text_content
        assert "Electronics" in result.text_content
        assert "Subtotal" in result.text_content

    def test_heuristic_table_detection(self):
        """Test basic heuristic table detection"""
        md = MarkItDown(table_strategy=TableExtractionStrategy.HEURISTIC)
        text = """
        Header1    Header2    Header3
        Value1     Value2     Value3
        Value4     Value5     Value6
        """
        tables = md._extract_tables_heuristic(text)
        assert len(tables) == 1
        assert len(tables[0]) == 3  # 3 rows
        assert len(tables[0][0]) == 3  # 3 columns

    @pytest.mark.skipif(not HAS_LAYOUTLM, reason="LayoutLM not installed")
    def test_layoutlm_table_detection(self):
        """Test LayoutLM table detection"""
        md = MarkItDown(table_strategy=TableExtractionStrategy.LAYOUTLM)
        
        # Create a simple test image with a table
        img = Image.new('RGB', (800, 600), color='white')
        # Add table-like content to image
        # ... image creation logic ...
        
        tables = md._extract_tables_layoutlm(None, img)
        assert len(tables) > 0

    def test_table_formatting(self):
        """Test table formatting to markdown"""
        md = MarkItDown()
        table = [
            ["Header1", "Header2", "Numbers"],
            ["Value1", "Value2", "123"],
            ["Value3", "Value4", "456"]
        ]
        formatted = md._format_table(table)
        assert len(formatted) > 0
        assert "Header1" in formatted[0]
        assert "---" in formatted[1]  # Separator row
        assert formatted[1].count("|") == formatted[0].count("|")  # Consistent columns

    def test_table_alignment(self):
        """Test table column alignment"""
        md = MarkItDown()
        table = [
            ["Text", "Amount", "Date"],
            ["Sample", "123.45", "2024-01-01"],
            ["Test", "67.89", "2024-01-02"]
        ]
        formatted = md._format_table(table)
        # Check right alignment for numeric column
        assert ":---:" in formatted[1] or "---:" in formatted[1]

    def test_complex_table_detection(self):
        """Test detection of complex table structures"""
        md = MarkItDown()
        text = """
        Financial Report 2024
        
        Revenue Breakdown
        Category    Q1        Q2        Q3        Q4
        Product A   $1,000    $1,200    $1,100    $1,300
        Product B   $800      $900      $950      $1,000
        Product C   $500      $600      $650      $700
        
        Total       $2,300    $2,700    $2,700    $3,000
        """
        tables = md._extract_tables_heuristic(text)
        assert len(tables) == 1
        assert len(tables[0]) == 5  # Header + 3 products + total
        assert len(tables[0][0]) == 5  # Category + 4 quarters

    def test_table_strategy_fallback(self):
        """Test automatic fallback between strategies"""
        md = MarkItDown(table_strategy=TableExtractionStrategy.AUTO)
        
        # Test with simple text that heuristic can handle
        simple_text = "Col1    Col2    Col3\nVal1    Val2    Val3"
        tables = md.extract_tables(simple_text)
        assert len(tables) > 0
        
        # Test with complex case that might need LayoutLM
        if HAS_LAYOUTLM:
            img = Image.new('RGB', (800, 600), color='white')
            tables = md.extract_tables(simple_text, image=img)
            assert len(tables) > 0

    @pytest.mark.parametrize("input_text,expected_rows", [
        ("A    B    C\n1    2    3", 2),
        ("No table here", 0),
        ("Col1|Col2|Col3\nVal1|Val2|Val3", 2),
    ])
    def test_various_table_formats(self, input_text, expected_rows):
        """Test detection of various table formats"""
        md = MarkItDown()
        tables = md._extract_tables_heuristic(input_text)
        if expected_rows == 0:
            assert len(tables) == 0
        else:
            assert len(tables) > 0
            assert len(tables[0]) == expected_rows 