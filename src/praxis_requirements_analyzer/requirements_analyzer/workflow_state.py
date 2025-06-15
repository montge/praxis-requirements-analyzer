"""
Workflow State Module

This module defines the state classes used in the requirements analysis workflow.

Classes:
    RequirementMatch: Data model for requirement matches
    WorkflowState: State container for the analysis workflow

Example:
    >>> from praxis_requirements_analyzer.requirements_analyzer.workflow_state import WorkflowState, RequirementMatch
    >>> match = RequirementMatch(
    ...     source_id="REQ-001",
    ...     target_id="REQ-002",
    ...     score=0.85,
    ...     vector_similarity=0.92,
    ...     reasoning="Strong functional overlap"
    ... )
    >>> state = WorkflowState(source_requirement=src_req, target_requirements=tgt_reqs)
    >>> state.matches.append(match)
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from ..neo4j.requirements_client import Requirement

class RequirementMatch(BaseModel):
    """
    Data model for requirement matches.
    
    This class represents a match between source and target requirements,
    including match quality metrics and validation status.
    
    Attributes:
        source_id (str): ID of the source requirement
        target_id (str): ID of the target requirement
        score (float): Overall match score (0-1)
        vector_similarity (float): Cosine similarity between requirement vectors
        reasoning (str): LLM's explanation for the match
        confidence (float): LLM's confidence in the match (0-1)
        validated (bool): Whether the match has been validated
        
    Example:
        >>> match = RequirementMatch(
        ...     source_id="REQ-001",
        ...     target_id="REQ-002",
        ...     score=0.85,
        ...     vector_similarity=0.92,
        ...     reasoning="Strong functional overlap"
        ... )
    """
    source_id: str
    target_id: str
    score: float
    vector_similarity: float
    reasoning: str
    confidence: float = 0.0
    validated: bool = False

class WorkflowState(BaseModel):
    """
    State container for the requirements analysis workflow.
    
    This class maintains the state of the analysis workflow, including
    the current requirements being processed and their matches.
    
    Attributes:
        source_requirement (Requirement): Current source requirement being processed
        target_requirements (List[Requirement]): List of target requirements to match against
        embeddings (Dict[str, Dict[str, Any]]): Dictionary storing requirement embeddings
        candidates (List[Dict[str, Any]]): List of candidate matches based on vector similarity
        matches (List[RequirementMatch]): List of validated requirement matches
        meta_judgments (List[Dict[str, Any]]): Meta-judge evaluations of matches
        
    Example:
        >>> state = WorkflowState(
        ...     source_requirement=source_req,
        ...     target_requirements=target_reqs
        ... )
        >>> state.matches.append(match)
    """
    source_requirement: Requirement
    target_requirements: List[Requirement]
    embeddings: Dict[str, Dict[str, Any]] = {}
    candidates: List[Dict[str, Any]] = []
    matches: List[RequirementMatch] = []
    meta_judgments: List[Dict[str, Any]] = [] 