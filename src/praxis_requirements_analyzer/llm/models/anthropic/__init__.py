"""
Anthropic Models Package

This package provides the ClaudeClient implementation for interacting with
Anthropic's Claude models.

Classes:
    ClaudeClient: Client implementation for Claude-2 and Claude-3 models

Example:
    >>> from praxis_requirements_analyzer.llm.models.anthropic import ClaudeClient
    >>> client = ClaudeClient("CLAUDE_3")
    >>> await client.initialize()
"""

from .claude_client import ClaudeClient

__all__ = ['ClaudeClient'] 