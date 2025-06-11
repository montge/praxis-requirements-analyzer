"""
Utilities Package

This package provides common utility functions and classes used throughout the application.

Key Components:
- Logger utilities: Setup and configuration of application logging
- Timer utilities: Performance measurement and profiling tools

Features:
- Consistent logging setup across modules
- Exception handling decorators
- Performance timing with checkpoints
- Debug-level runtime statistics

Usage:
    # Logger setup
    from praxis_requirements_analyzer.utils import setup_logger, handle_exception
    
    logger = setup_logger(__name__)
    
    @handle_exception("Operation failed")
    def risky_operation():
        # Function code...
        
    # Performance timing
    from praxis_requirements_analyzer.utils import DebugTimer
    
    timer = DebugTimer("DataProcessing")
    # Some operations...
    timer.checkpoint("Data loaded")
    # More operations...
    total_time = timer.end()
"""

from .logger import (
    setup_logger,
    log_error,
    handle_exception
)
from .timer import DebugTimer

__all__ = [
    'setup_logger',
    'log_error',
    'handle_exception',
    'DebugTimer'
] 