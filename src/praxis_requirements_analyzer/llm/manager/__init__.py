"""
LLM Manager Package

This package provides the LLMManager class for centralized management of
LLM model interactions.

Classes:
    LLMManager: Centralized manager for LLM model initialization and interactions

Example:
    >>> from praxis_requirements_analyzer.llm.manager import LLMManager
    >>> manager = LLMManager()
    >>> await manager.initialize_models()
"""

from .llm_manager import LLMManager

__all__ = ['LLMManager'] 