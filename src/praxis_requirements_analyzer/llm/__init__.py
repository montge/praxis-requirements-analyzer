"""
LLM (Language Learning Model) package.

This package provides functionality for managing and interacting with various LLM providers
including Anthropic Claude, OpenAI, and HuggingFace models.

Key Components:
    - LLMManager: Central manager for all LLM models
    - Client implementations: Direct access to specific model clients

Classes:
    LLMManager: Main class for managing LLM model interactions and providing
        a unified interface across different providers.
    ClaudeClient: Client for Anthropic's Claude models
    OpenAIClient: Client for OpenAI's GPT models
    HuggingFaceClient: Client for HuggingFace models

Example:
    # Using the manager (recommended)
    >>> from praxis_requirements_analyzer.llm import LLMManager
    >>> llm_manager = LLMManager()
    >>> await llm_manager.initialize_models()
    >>> response = await llm_manager.test_model("claude-3", "Hello!")

    # Using a specific client directly
    >>> from praxis_requirements_analyzer.llm import ClaudeClient
    >>> claude = ClaudeClient("CLAUDE_3_5")
    >>> await claude.initialize()
"""

# Main manager
from .manager.llm_manager import LLMManager

# Model clients for direct access
from .models.anthropic import ClaudeClient
from .models.openai import OpenAIClient
from .models.huggingface import HuggingFaceClient, HuggingFaceVLLMClient, HuggingFaceEmbeddingsClient

# Abstract base class
from .base_llm import BaseLLM

__all__ = [
    'LLMManager',
    'ClaudeClient',
    'OpenAIClient',
    'HuggingFaceClient',
    'HuggingFaceVLLMClient',
    'HuggingFaceEmbeddingsClient',
    'BaseLLM'
] 