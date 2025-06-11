"""
Praxis Requirements Analyzer

This package provides a comprehensive system for analyzing software requirements
using LLMs, vector similarity, and graph databases.

Packages:
    llm: Large Language Model integrations and management
    models: Data models and schemas for requirements analysis
    neo4j: Graph database client and schema management
    redis: Vector similarity search and caching
    requirements_analyzer: Main requirements analysis workflow
    utils: Shared utilities and logging

Example:
    # LLM Management
    >>> from praxis_requirements_analyzer.llm import LLMManager
    >>> llm_manager = LLMManager()
    
    # Requirements Analysis
    >>> from praxis_requirements_analyzer.requirements_analyzer import RequirementsWorkflow
    >>> workflow = RequirementsWorkflow(llm_manager, ...)
    
    # Database Clients
    >>> from praxis_requirements_analyzer.neo4j import Neo4jClient
    >>> from praxis_requirements_analyzer.redis import RedisClient
"""

from . import llm
from . import models
from . import neo4j
from . import redis
from . import requirements_analyzer
from . import utils

__all__ = [
    'llm',
    'models',
    'neo4j',
    'redis',
    'requirements_analyzer',
    'utils'
]

# Package version
__version__ = "0.2.0" 