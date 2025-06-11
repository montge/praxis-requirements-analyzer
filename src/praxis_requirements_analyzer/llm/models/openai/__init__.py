"""
OpenAI Models Package

This package provides the OpenAIClient implementation for interacting with
OpenAI's GPT models through their API.

Models:
    - GPT-3.5 (faster, more economical)
    - GPT-4/GPT-4o (more capable, better reasoning)

Features:
    - Authentication management
    - Request caching to reduce costs
    - Detailed logging and debugging
    - Comprehensive error handling
    - Support for system instructions

Classes:
    OpenAIClient: Client implementation for GPT models

Example:
    >>> from praxis_requirements_analyzer.llm.models.openai import OpenAIClient
    >>> client = OpenAIClient("GPT4")
    >>> await client.initialize()
    >>> response = await client.generate_response(
    ...     messages=[
    ...         {"role": "system", "content": "You are a helpful assistant."},
    ...         {"role": "user", "content": "Hello!"}
    ...     ]
    ... )
    >>> print(response["content"])
"""

from .openai_client import OpenAIClient

__all__ = ['OpenAIClient'] 