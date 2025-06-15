# Pandas startup configuration for wide landscape table display
# This file is executed at Python startup to configure pandas display options

try:
    import pandas as pd
    
    # Configure pandas for legal landscape format with 9pt font requirement
    pd.set_option('display.width', 130)           # Set width threshold for legal landscape
    pd.set_option('display.max_columns', 25)     # Reasonable number of columns
    pd.set_option('display.max_colwidth', 25)    # Compact column width
    pd.set_option('display.precision', 2)        # Only 2 decimal places to save space
    pd.set_option('display.float_format', '{:.2f}'.format)  # Consistent float formatting
    # Note: Removed expand_frame_repr=False to allow natural wrapping at 130 chars
    
    print("✅ Pandas configured for wide landscape display (130 chars width, natural wrapping)")
    
except ImportError:
    # pandas not available, skip configuration
    pass 