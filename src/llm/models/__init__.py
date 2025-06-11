"""
LLM Models Package

This package provides client implementations for various LLM providers.

Classes:
    ClaudeClient: Client for Anthropic's Claude models
    HuggingFaceClient: Client for HuggingFace's models
    OpenAIClient: Client for OpenAI's GPT models

Example:
    >>> from src.llm.models import ClaudeClient
    >>> client = ClaudeClient("CLAUDE_3")
    >>> await client.initialize()
"""

from .anthropic import ClaudeClient
from .huggingface import HuggingFaceClient
from .openai import OpenAIClient

__all__ = ['ClaudeClient', 'HuggingFaceClient', 'OpenAIClient'] 