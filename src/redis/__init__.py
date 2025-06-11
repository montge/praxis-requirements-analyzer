"""
Redis Package

This package provides Redis client functionality for the requirements analysis system.

Key Components:
- RedisClient: Manages Redis connections and vector search operations

Features:
- Vector similarity search using Redis Stack
- Automatic index creation and management
- Connection pooling and error handling
- Support for embedding storage and retrieval

Usage:
    from praxis_requirements_analyzer.redis import RedisClient
    
    client = RedisClient(
        host="localhost",
        port=6379,
        password=None
    )
    client.connect()
"""

from .redis_client import RedisClient

__all__ = ['RedisClient'] 