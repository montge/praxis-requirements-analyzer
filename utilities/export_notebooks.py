#!/usr/bin/env python3
"""
Export notebooks as PDFs with project name and timestamp
Usage: 
  python export_notebooks.py                              # Export default 3 notebooks (batch mode)
  python export_notebooks.py <notebook_path>              # Export single notebook  
  python export_notebooks.py <path1> <path2> ...          # Export multiple notebooks (batch mode)

Examples:
  python export_notebooks.py ../notebooks/11_Hypothesis_2_Tracability_SiFP.ipynb
  python export_notebooks.py ../notebooks/nb1.ipynb ../notebooks/nb2.ipynb
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

def load_project_name():
    """Load NEO4J_PROJECT_NAME from .env file"""
    # Load .env file from parent directory (project root), override existing env vars
    env_file = Path('../.env')
    if not env_file.exists():
        print("ERROR: .env file not found in parent directory")
        print(f"Current directory: {Path.cwd()}")
        print(f"Looking for: {env_file.absolute()}")
        sys.exit(1)
    
    # Force reload from .env file, override existing environment variables
    load_dotenv(env_file, override=True)
    project_name = os.getenv('NEO4J_PROJECT_NAME')
    
    if not project_name:
        print("ERROR: NEO4J_PROJECT_NAME not found in .env file")
        print("Please ensure your .env file contains:")
        print("NEO4J_PROJECT_NAME=your_project_name")
        sys.exit(1)
    
    print(f"✅ Loaded project name from .env: {project_name}")
    return project_name

def get_timestamp():
    """Get current timestamp in YYYYMMDD_HHMMSS format"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def run_and_export_notebook(notebook_path, output_dir, project_name, timestamp, is_single_mode=False):
    """
    Run a notebook and export it as PDF with only outputs
    
    Args:
        notebook_path: Path to the notebook file
        output_dir: Directory to save the PDF
        project_name: NEO4J project name
        timestamp: Timestamp string
        is_single_mode: If True, shows detailed single-notebook output
    """
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        print(f"ERROR: Notebook not found: {notebook_path}")
        return False
    
    # Extract notebook name without extension
    notebook_name = notebook_path.stem
    
    # Create output filename with project name and timestamp
    output_filename = f"{notebook_name}_{project_name}_{timestamp}.pdf"
    
    # Use absolute path to ensure it's created in the root directory
    output_path = Path.cwd() / output_dir / output_filename
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Output formatting based on mode
    if is_single_mode:
        print(f"🔄 Exporting: {notebook_path}")
        print(f"📄 Output: {output_path}")
    else:
        print(f"Running and exporting: {notebook_path}")
        print(f"Output: {output_path}")
    
    try:
        # Run nbconvert with execute and PDF export, hiding input cells
        # Using custom template for legal landscape format
        template_path = Path(__file__).parent / "legal_landscape.tex.j2"
        pandas_startup_path = Path(__file__).parent / "pandas_startup.py"
        
        cmd = [
            "jupyter", "nbconvert",
            "--to", "pdf",
            "--execute", 
            "--no-input",  # Hide code cells, show only outputs
            "--template-file", str(template_path),
            "--output", str(output_path),
            str(notebook_path)
        ]
        
        # Set environment to configure pandas for wide tables
        env = os.environ.copy()
        env['PYTHONSTARTUP'] = str(pandas_startup_path)
        
        # Run the command with pandas startup configuration
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        
        # Success output based on mode
        if is_single_mode:
            print(f"✅ Successfully exported: {notebook_name}")
            # Show file size for single mode
            if output_path.exists():
                file_size = output_path.stat().st_size
                size_kb = file_size / 1024
                print(f"📏 File size: {size_kb:.0f}KB")
        else:
            print(f"✅ Successfully exported: {output_filename}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running notebook {notebook_path}:")
        print(f"Command: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Error: jupyter command not found. Make sure Jupyter is installed and in PATH.")
        return False

def show_available_notebooks():
    """Show available notebooks in the notebooks directory"""
    print("Available notebooks in ../notebooks/:")
    notebooks_dir = Path("../notebooks")
    if notebooks_dir.exists():
        for nb in notebooks_dir.glob("*.ipynb"):
            print(f"  ../notebooks/{nb.name}")
    else:
        print("  ../notebooks/ directory not found")

def main():
    """Main function to export notebooks with auto-detection of single vs batch mode"""
    
    # Determine mode based on arguments
    if len(sys.argv) == 1:
        # No arguments - default batch mode
        mode = "default_batch"
        notebooks = [
            "../notebooks/10_Hypothesis_1_Sentence_Transformer_Meta_Judge_Comparison.ipynb",
            "../notebooks/11_Hypothesis_2_Tracability_SiFP.ipynb", 
            "../notebooks/12_Hypothesis_3_SiFP_COSMIC_Estimation.ipynb"
        ]
    elif len(sys.argv) == 2:
        # Single argument - could be single mode or help
        if sys.argv[1] in ['-h', '--help', 'help']:
            print(__doc__)
            sys.exit(0)
        mode = "single"
        notebooks = [sys.argv[1]]
    else:
        # Multiple arguments - custom batch mode
        mode = "custom_batch"
        notebooks = sys.argv[1:]
    
    # Mode-specific startup messages
    if mode == "single":
        print("🚀 Starting single notebook export...")
    elif mode == "default_batch":
        print("🚀 Starting default batch export...")
    else:
        print("🚀 Starting custom batch export...")
    
    # Load project name from .env
    project_name = load_project_name()
    
    # Get timestamp
    timestamp = get_timestamp()
    print(f"📅 Timestamp: {timestamp}")
    
    # Show what we're processing
    if mode == "default_batch":
        print("Using default notebook set")
    elif mode == "single":
        print(f"Notebook: {notebooks[0]}")
    else:
        print(f"Using provided notebook paths: {notebooks}")
    
    # Validate all notebooks exist before starting
    missing_notebooks = []
    for notebook in notebooks:
        if not Path(notebook).exists():
            missing_notebooks.append(notebook)
    
    if missing_notebooks:
        print(f"❌ Error: The following notebooks were not found:")
        for nb in missing_notebooks:
            print(f"  {nb}")
        print()
        show_available_notebooks()
        sys.exit(1)
    
    # Output directory (relative to parent directory)
    output_dir = "../resultsPdf"
    
    print(f"Project: {project_name}")
    print(f"Output directory: {output_dir}")
    
    # Single mode vs batch mode formatting
    if mode == "single":
        print("=" * 80)
        # Process single notebook
        success = run_and_export_notebook(notebooks[0], output_dir, project_name, timestamp, is_single_mode=True)
        
        if success:
            print("🎉 Export completed successfully!")
            sys.exit(0)
        else:
            print("⚠️ Export failed. Check the errors above.")
            sys.exit(1)
    else:
        # Batch mode
        print(f"\nExporting {len(notebooks)} notebooks...")
        print("=" * 80)
        
        # Process each notebook
        success_count = 0
        for i, notebook in enumerate(notebooks, 1):
            print(f"\n[{i}/{len(notebooks)}] Processing: {notebook}")
            success = run_and_export_notebook(notebook, output_dir, project_name, timestamp, is_single_mode=False)
            if success:
                success_count += 1
            print("-" * 40)
        
        # Summary
        print(f"\nExport Summary:")
        print(f"Total notebooks: {len(notebooks)}")
        print(f"Successfully exported: {success_count}")
        print(f"Failed: {len(notebooks) - success_count}")
        
        if success_count == len(notebooks):
            print("🎉 All notebooks exported successfully!")
        else:
            print("⚠️  Some notebooks failed to export. Check the errors above.")

if __name__ == "__main__":
    main() 