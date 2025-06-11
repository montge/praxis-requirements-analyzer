"""
LLM Manager Package

This package provides the LLMManager class for centralized management of
LLM model interactions.

Classes:
    LLMManager: Centralized manager for LLM model initialization and interactions

Example:
    >>> from src.llm.manager import LLMManager
    >>> manager = LLMManager()
    >>> await manager.initialize_models()
"""

from src.llm.manager.llm_manager import LLMManager

__all__ = ['LLMManager'] 