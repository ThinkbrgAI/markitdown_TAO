# SPDX-FileCopyrightText: 2024-present Adam Fourney <adamfo@microsoft.com>
#
# SPDX-License-Identifier: MIT
import sys
import argparse
from PyQt6.QtWidgets import QApplication
from ._markitdown import MarkItDown
from .ui import MarkItDownUI

def main():
    # Check if we should show UI or use CLI
    if len(sys.argv) == 1:
        # Show UI
        app = QApplication(sys.argv)
        window = MarkItDownUI()
        window.show()
        sys.exit(app.exec())
    else:
        # Use CLI
        parser = argparse.ArgumentParser(description='Convert documents to Markdown')
        parser.add_argument('input', nargs='?', type=str, help='Input file path')
        parser.add_argument('-o', '--output', type=str, help='Output file path')
        parser.add_argument('--llm-model', type=str, help='LLM model to use for enhanced conversion')
        parser.add_argument('--table-strategy', 
                           choices=['heuristic', 'layoutlm', 'gpt4v', 'auto'],
                           default='heuristic',
                           help='Table extraction strategy to use')
        
        args = parser.parse_args()

        md = MarkItDown(
            llm_model=args.llm_model,
            table_strategy=args.table_strategy
        )
        
        # Handle piped input
        if not args.input and not sys.stdin.isatty():
            content = sys.stdin.buffer.read()
            result = md.convert_from_bytes(content)
        else:
            result = md.convert(args.input)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result.text_content)
        else:
            print(result.text_content)

if __name__ == '__main__':
    main()