"""
Neo4j Graph Database Package

This package provides Neo4j graph database integration for the requirements analysis system.

Key Components:
- Neo4jClient: Base client for Neo4j database connections
- RequirementsClient: Specialized client for requirements data operations
- SchemaExtractor: Utility for extracting and managing database schema

Features:
- Async connection management
- Automated schema validation and creation
- Requirements storage and graph traversal
- Error handling and connection pooling
- Support for relationship analysis and visualization

Usage:
    from praxis_requirements_analyzer.neo4j import Neo4jClient, RequirementsClient
    
    # Basic Connection
    client = Neo4jClient(
        uri="neo4j://localhost:7687",
        user="neo4j",
        password="password"
    )
    await client.connect()
    
    # Requirements Operations
    req_client = RequirementsClient(client)
    await req_client.store_requirement(requirement_data)
"""

from .neo4j_client import Neo4jClient
from .requirements_client import RequirementsClient, Requirement
from .schema_extractor import SchemaExtractor

__all__ = [
    'Neo4jClient',
    'RequirementsClient',
    'Requirement',
    'SchemaExtractor'
] 