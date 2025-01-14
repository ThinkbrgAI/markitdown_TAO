# SPDX-FileCopyrightText: 2024-present Adam Fourney <adamfo@microsoft.com>
#
# SPDX-License-Identifier: MIT
import argparse
import sys
from ._markitdown import MarkItDown


def main():
    parser = argparse.ArgumentParser(description="Convert documents to Markdown format")
    
    # Add arguments
    parser.add_argument("input", help="Input file or folder")
    parser.add_argument("-o", "--output", help="Output file or folder (default: print to stdout for single file, ./output for folder)")
    parser.add_argument("-p", "--pattern", default="*.msg", help="File pattern when processing folder (default: *.msg)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = MarkItDown()
    
    # Check if input is a file or directory
    import os
    if os.path.isfile(args.input):
        # Single file conversion
        try:
            result = converter.convert(args.input)
            
            # Output handling
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(result.text_content)
            else:
                print(result.text_content)
                
        except Exception as e:
            print(f"Error converting file: {str(e)}", file=sys.stderr)
            sys.exit(1)
            
    elif os.path.isdir(args.input):
        # Batch conversion
        output_dir = args.output if args.output else os.path.join(".", "output")
        
        try:
            results = converter.batch_convert_folder(
                args.input,
                output_dir,
                file_pattern=args.pattern,
                skip_existing=not args.force
            )
            
            # Print summary
            print("\nConversion Summary:")
            success_count = sum(1 for r in results.values() if r == "success")
            skip_count = sum(1 for r in results.values() if r == "skipped - already exists")
            error_count = len(results) - success_count - skip_count
            
            print(f"Total files: {len(results)}")
            print(f"Successful: {success_count}")
            print(f"Skipped: {skip_count}")
            print(f"Failed: {error_count}")
            
            # Print errors if any
            if error_count > 0:
                print("\nErrors:")
                for filename, result in results.items():
                    if isinstance(result, Exception):
                        print(f"{filename}: {str(result)}")
                        
        except Exception as e:
            print(f"Error during batch conversion: {str(e)}", file=sys.stderr)
            sys.exit(1)
            
    else:
        print(f"Error: Input path '{args.input}' does not exist", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()