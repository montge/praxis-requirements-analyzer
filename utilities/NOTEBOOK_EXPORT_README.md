# Notebook Export Tools

This directory contains Python scripts for exporting Jupyter notebooks to PDF format with legal landscape layout (8.5" x 14"), including project name and timestamp in filenames.

## Features

- 📄 **PDF Export**: Convert Jupyter notebooks to PDF with output-only view (no code cells)
- 🏞️ **Legal Landscape Format**: Optimized 8.5" x 14" layout for wide tables and charts
- 🏷️ **Smart Naming**: Includes NEO4J_PROJECT_NAME from .env and timestamp in filenames
- 📊 **Wide Table Support**: Configured pandas display for wide DataFrames in PDFs
- 🎯 **Flexible Usage**: Auto-detects single vs batch mode based on arguments
- ✅ **Validation**: Pre-flight checks for all required files and dependencies

## Files

- `export_notebooks.py` - Unified export script (handles single notebooks and batch export)
- `legal_landscape.tex.j2` - LaTeX template for landscape format
- `pandas_startup.py` - Pandas configuration for wide tables
- `pandas_width_preprocessor.py` - Custom preprocessor for table formatting
- `NOTEBOOK_EXPORT_README.md` - This documentation

## Quick Start

### From Project Root (Recommended)
```bash
# Export all default notebooks
python export_notebooks_quick.py
```

### From Utilities Directory

**Export Default Notebooks (Batch Mode):**
```bash
cd utilities
python export_notebooks.py
```

**Export Single Notebook:**
```bash
cd utilities
python export_notebooks.py ../notebooks/11_Hypothesis_2_Tracability_SiFP.ipynb
```

**Export Multiple Custom Notebooks (Batch Mode):**
```bash
cd utilities
python export_notebooks.py ../notebooks/nb1.ipynb ../notebooks/nb2.ipynb ../notebooks/nb3.ipynb
```

## Requirements

### Environment Setup
1. **Python Environment**: Jupyter, nbconvert, pandas, python-dotenv
2. **LaTeX**: Full LaTeX distribution (for PDF generation)
3. **Environment Variables**: `.env` file in project root with `NEO4J_PROJECT_NAME`

### Installation
```bash
# Install required packages
pip install jupyter nbconvert pandas python-dotenv

# Install LaTeX (Ubuntu/Debian)
sudo apt-get install texlive-full

# Install LaTeX (macOS with Homebrew)
brew install mactex
```

### Environment File
Create `.env` in project root:
```
NEO4J_PROJECT_NAME=your_project_name
```

## Usage Examples

### Default Export (3 Notebooks - Batch Mode)
```bash
cd utilities
python export_notebooks.py
```
Exports:
- `../notebooks/10_Hypothesis_1_Sentence_Transformer_Meta_Judge_Comparison.ipynb`
- `../notebooks/11_Hypothesis_2_Tracability_SiFP.ipynb`
- `../notebooks/12_Hypothesis_3_SiFP_COSMIC_Estimation.ipynb`

### Single Notebook Export
```bash
cd utilities
# Single notebook with full path - shows file size and detailed output
python export_notebooks.py ../notebooks/custom_analysis.ipynb
```

### Custom Batch Export
```bash
cd utilities
# Multiple notebooks with full paths - shows progress and summary
python export_notebooks.py ../notebooks/report1.ipynb ../notebooks/report2.ipynb ../other_folder/notebook.ipynb
```

### Output Files
Files are saved to `../resultsPdf/` with format:
```
{notebook_name}_{project_name}_{timestamp}.pdf
```

Example:
```
11_Hypothesis_2_Tracability_SiFP_iTrust_20241215_143022.pdf
```

## Auto-Detection Features

The script automatically detects usage mode:

| Arguments | Mode | Behavior |
|-----------|------|----------|
| None | **Default Batch** | Exports 3 default notebooks, shows progress + summary |
| 1 notebook path | **Single Mode** | Exports one notebook, shows file size + detailed output |
| Multiple paths | **Custom Batch** | Exports specified notebooks, shows progress + summary |

**Single Mode Features:**
- 🔄 Detailed visual feedback with emojis
- 📏 File size reporting
- 🎯 Focused output for single notebook workflow

**Batch Mode Features:**
- 📊 Progress tracking ([1/3] Processing...)
- 📈 Summary statistics
- ⚡ Efficient processing of multiple notebooks

## Template Features

The `legal_landscape.tex.j2` template provides:
- **Legal size paper** (8.5" × 14")
- **Landscape orientation**
- **Optimized margins** for maximum content space
- **Wide table support** with automatic scaling
- **Professional formatting** with proper spacing

## Pandas Configuration

The export process includes:
- **Wide display options** for DataFrames
- **Increased column/row limits** for comprehensive tables
- **Custom preprocessor** for table formatting
- **Automatic width adjustment** for PDF layout

## Troubleshooting

### Common Issues

**1. "jupyter command not found"**
```bash
pip install jupyter nbconvert
```

**2. "LaTeX Error: File not found"**
```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS
brew install mactex
```

**3. "NEO4J_PROJECT_NAME not found"**
- Ensure `.env` file exists in project root
- Check `.env` file contains: `NEO4J_PROJECT_NAME=your_project_name`
- Verify no extra spaces or special characters

**4. "Notebook not found"**
- Use full relative paths: `../notebooks/filename.ipynb`
- Check the notebook exists at the specified path
- Ensure you're running from the utilities directory

**5. Template errors with nbconvert**
- Template uses basic LaTeX approach for maximum compatibility
- If issues persist, template will fall back to standard PDF export

### Help Command
```bash
cd utilities
python export_notebooks.py --help
# or
python export_notebooks.py -h
```

### Debug Commands
```bash
cd utilities

# Check environment loading
python -c "from dotenv import load_dotenv; import os; load_dotenv('../.env', override=True); print(f'Project: {os.getenv(\"NEO4J_PROJECT_NAME\")}')"

# List available notebooks
ls -la ../notebooks/*.ipynb

# Test basic nbconvert
jupyter nbconvert --to pdf --execute --no-input ../notebooks/test_notebook.ipynb
```

## Script Details

### export_notebooks.py (Unified Script)
The consolidated script provides all functionality in one place:

**Features:**
- **Auto-mode detection**: Single vs batch based on arguments
- **Default behavior**: Exports 3 hypothesis notebooks when no args provided
- **Flexible input**: Accepts any number of notebook paths
- **Smart output**: File size for single mode, progress/summary for batch
- **Comprehensive validation**: Checks all notebooks exist before starting
- **Help support**: Built-in help with `-h`, `--help`, or `help`

**Usage Patterns:**
```bash
# Default batch (3 notebooks)
python export_notebooks.py

# Single notebook (shows file size)
python export_notebooks.py ../notebooks/analysis.ipynb

# Custom batch (multiple notebooks)  
python export_notebooks.py ../notebooks/nb1.ipynb ../notebooks/nb2.ipynb

# Help
python export_notebooks.py --help
```

## Output Directory

All PDFs are saved to: `../resultsPdf/`
- Directory is created automatically if it doesn't exist
- Files are organized by timestamp
- Easy to identify with project name in filename

## Migration from Previous Version

If you were using the old `export_single_notebook.py` script:

**Old usage:**
```bash
python export_single_notebook.py ../notebooks/notebook.ipynb
```

**New usage (identical functionality):**
```bash
python export_notebooks.py ../notebooks/notebook.ipynb
```

The consolidated script provides all the same features with improved usability and no code duplication. 