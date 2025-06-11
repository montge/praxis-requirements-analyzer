"""
Requirement Model Module

Defines the data structures for requirements and related entities.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class Requirement:
    """Data class to store requirement information.
    
    Attributes:
        id: Unique identifier for the requirement
        content: The actual text content of the requirement
        type: Type of requirement (SOURCE or TARGET)
        level: Optional hierarchical level of the requirement
    """
    id: str
    content: str
    type: str  # SOURCE or TARGET
    level: Optional[int] = None 