#!/usr/bin/env python3
"""
Convenience script to run notebook exports from root directory
This script calls the actual export scripts in the utilities directory
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Main function to run notebook exports"""
    print("🚀 Quick Notebook Export (Python version)")
    print("=" * 50)
    
    # Check if utilities directory exists
    utilities_dir = Path("utilities")
    if not utilities_dir.exists():
        print("❌ utilities directory not found")
        print("This script must be run from the project root directory")
        sys.exit(1)
    
    # Change to utilities directory and run the export
    print("📁 Changing to utilities directory...")
    print("🔧 Running export_notebooks.py...")
    
    try:
        # Run the Python export script
        result = subprocess.run([
            sys.executable, "export_notebooks.py"
        ], cwd=utilities_dir, check=True)
        
        print("✅ Export completed successfully!")
        print("📄 PDFs saved in: resultsPdf/")
        
        # Show results
        results_dir = Path("resultsPdf")
        if results_dir.exists():
            print("")
            print("📊 Generated files:")
            pdf_files = list(results_dir.glob("*.pdf"))
            if pdf_files:
                for pdf_file in sorted(pdf_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:  # Show last 10
                    size_kb = pdf_file.stat().st_size / 1024
                    print(f"  {pdf_file.name} ({size_kb:.0f}KB)")
                if len(pdf_files) > 10:
                    print(f"  ... and {len(pdf_files) - 10} more files")
            else:
                print("  No PDF files found")
    
    except subprocess.CalledProcessError as e:
        print("❌ Export failed. Check the error messages above.")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: Python interpreter not found.")
        sys.exit(1)


if __name__ == "__main__":
    main() 