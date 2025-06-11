"""
Timer Utility Module

This module provides a debug timer for performance monitoring and profiling.
It includes functionality for:
- Starting and stopping timers
- Recording checkpoints
- Measuring elapsed time between operations
"""

import time
from .logger import setup_logger

logger = setup_logger(__name__)

class DebugTimer:
    """
    Timer class for debugging and performance monitoring.
    
    This class provides utilities for timing operations and logging the results,
    useful for performance profiling and debugging.
    
    Attributes:
        name (str): Identifier for the timer
        start_time (float): Time when the timer was started
        last_checkpoint (float): Time of the last checkpoint
        logger (logging.Logger): Logger instance for output
        
    Example:
        timer = DebugTimer("DataProcessing")
        # Some operations...
        timer.checkpoint("Data loaded")
        # More operations...
        timer.end()
    """
    
    def __init__(self, name: str):
        """
        Initialize timer with name for identification.
        
        Args:
            name: Identifier for this timer instance
        """
        self.name = name
        self.start_time = None
        self.last_checkpoint = None
        self.logger = setup_logger(__name__)
        self.start()
    
    def start(self) -> None:
        """
        Start the timer.
        
        Initializes both the start time and first checkpoint.
        """
        self.start_time = time.time()
        self.last_checkpoint = self.start_time
        self.logger.debug(f"{self.name} timer started")
    
    def checkpoint(self, message: str) -> float:
        """
        Record a checkpoint with message and return duration since last checkpoint.
        
        Args:
            message: Description of the checkpoint
            
        Returns:
            float: Duration in seconds since the last checkpoint
            
        Example:
            duration = timer.checkpoint("Data processing complete")
        """
        current = time.time()
        duration = current - self.last_checkpoint
        self.last_checkpoint = current
        self.logger.debug(f"{self.name} - {message} ({duration:.2f}s)")
        return duration
    
    def end(self) -> float:
        """
        End timing and return total duration.
        
        Returns:
            float: Total duration in seconds from start to end
            
        Note:
            Returns 0 if the timer was never started
        """
        if self.start_time is None:
            return 0
        self.duration = time.time() - self.start_time
        self.logger.debug(f"{self.name} completed in {self.duration:.2f}s")
        return self.duration 