#!/usr/bin/env python3
"""
Custom nbconvert preprocessor to configure pandas for wide table display
"""

from nbconvert.preprocessors import Preprocessor
import nbformat.v4 as nbf


class PandasWidthPreprocessor(Preprocessor):
    """
    Preprocessor that injects pandas configuration at the start of notebook execution
    to enable wide table display for legal landscape format (14" width)
    """
    
    def preprocess(self, nb, resources):
        """
        Add pandas configuration cell at the beginning of the notebook
        """
        # Create a code cell with pandas configuration
        pandas_config_code = """
# Configure pandas for wide landscape table display (injected by export script)
import pandas as pd
pd.set_option('display.width', 200)           # Very wide display (legal landscape)
pd.set_option('display.max_columns', 50)     # Show many columns
pd.set_option('display.max_colwidth', 50)    # Reasonable column width
pd.set_option('display.expand_frame_repr', False)  # Don't wrap DataFrame repr
print("✅ Pandas configured for wide landscape display (200 chars width)")
"""
        
        # Create the configuration cell
        config_cell = nbf.new_code_cell(source=pandas_config_code)
        
        # Insert at the beginning of the notebook
        nb.cells.insert(0, config_cell)
        
        return nb, resources 