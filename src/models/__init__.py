"""
Models Package

This package contains data models used throughout the application for
representing requirements and related entities.

Purpose:
    Provides standardized data structures for representing and manipulating
    software requirements, ensuring consistent data handling across the application.

Key Features:
    - Type annotations for IDE assistance and validation
    - Dataclass-based models for clean serialization
    - Business logic encapsulation within models
    - Integration with database operations

Classes:
    Requirement: Data class for storing requirement information
        - id: Unique identifier
        - content: Textual content of the requirement
        - type: Requirement type (SOURCE, TARGET, etc.)
        - level: Optional hierarchical level

Example:
    # Creating a requirement
    >>> from praxis_requirements_analyzer.models import Requirement
    >>> req = Requirement(id="REQ-001", content="The system shall...", type="SOURCE")
    
    # Using with batch operations
    >>> requirements = [
    ...     Requirement(id="REQ-001", content="First requirement", type="SOURCE"),
    ...     Requirement(id="REQ-002", content="Second requirement", type="SOURCE")
    ... ]
"""

from .requirement import Requirement

__all__ = ['Requirement'] 