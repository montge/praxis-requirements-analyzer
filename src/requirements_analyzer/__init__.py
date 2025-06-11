"""
Requirements Analyzer Package

This package provides functionality for analyzing and matching software requirements
using LLMs and vector similarity.

Classes:
    RequirementsWorkflow: Main workflow for requirements analysis and matching
    RequirementsPromptManager: Manager for LLM prompts used in analysis
    WorkflowState: State container for analysis workflow
    RequirementMatch: Data model for requirement matches

Example:
    >>> from praxis_requirements_analyzer.requirements_analyzer import RequirementsWorkflow, RequirementsPromptManager
    >>> prompt_manager = RequirementsPromptManager()
    >>> workflow = RequirementsWorkflow(llm_manager, prompt_manager, ...)
    >>> matches = await workflow.process_requirements_batch(source_reqs, target_reqs)
"""

from .requirements_workflow import RequirementsWorkflow
from .requirements_prompt_manager import RequirementsPromptManager
from .workflow_state import WorkflowState, RequirementMatch

__all__ = [
    'RequirementsWorkflow',
    'RequirementsPromptManager',
    'WorkflowState',
    'RequirementMatch'
] 