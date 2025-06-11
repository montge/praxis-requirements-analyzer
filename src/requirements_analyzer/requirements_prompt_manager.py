"""
Requirements Prompt Manager Module

This module manages system prompts for different phases of requirements analysis
using the Meta-Judge evaluation approach.

Classes:
    RequirementsPromptManager: Manager for LLM prompts used in analysis phases

Components:
    - Actor prompts for generating matches
    - Judge prompts for evaluating matches
    - Meta-judge prompts for validating judgments
    - Verification prompts using INCOSE guidelines
    - Validation prompts for requirement sets

Example:
    >>> from src.requirements_analyzer import RequirementsPromptManager
    >>> prompt_manager = RequirementsPromptManager(min_association_probability=0.7)
    >>> actor_prompt = prompt_manager.actor_prompt
    >>> judge_prompt = prompt_manager.judge_prompt
"""

import os
from dotenv import load_dotenv
from typing import Optional
from src.utils.logger import setup_logger, handle_exception
import logging

# Set up logger with debug level
logger = setup_logger(__name__, logging.DEBUG)

class RequirementsPromptManager:
    """
    Manages system prompts for requirements analysis including Meta-Judge evaluation.
    
    The manager provides prompts for:
    - Actor role: Generating requirement matches
    - Judge role: Evaluating match quality
    - Meta-judge role: Validating judgments
    - Verification: INCOSE guideline checks
    - Validation: Requirements set analysis
    
    Attributes:
        min_association_probability (float): Minimum probability threshold for associations
        embedding_model (str): Name of HuggingFace embedding model
        embedding_dimension (int): Dimension of embeddings
        _actor_prompt (str): Prompt for generating requirement matches
        _judge_prompt (str): Prompt for evaluating matches
        _meta_judge_prompt (str): Prompt for evaluating judgments
        _verification_prompt (str): Prompt for requirements verification
        _validation_prompt (str): Prompt for requirements validation
    """
    
    def __init__(self, 
                 min_association_probability: Optional[float] = None,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedding_dimension: int = 384):
        """
        Initialize the RequirementsPromptManager.
        
        Args:
            min_association_probability (float, optional): Minimum probability threshold
            embedding_model (str): HuggingFace model name for embeddings
            embedding_dimension (int): Dimension of embeddings from model
        """
        load_dotenv()
        logger.debug("Initializing RequirementsPromptManager")
        self._actor_prompt = None
        self._judge_prompt = None
        self._meta_judge_prompt = None
        self._verification_prompt = None
        self._validation_prompt = None
        self.min_association_probability = min_association_probability or float(os.getenv('MIN_ASSOCIATION_PROBABILITY', 0.6))
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        logger.debug(f"Set min_association_probability to {self.min_association_probability}")

    @property
    def actor_prompt(self) -> str:
        """
        Get the system prompt for the actor role.
        
        The actor analyzes requirements in their original language but provides
        analysis and reasoning in English.
        
        Returns:
            str: The formatted actor prompt
        """
        if not hasattr(self, '_actor_prompt') or self._actor_prompt is None:
            logger.warning("Actor prompt not initialized, creating default prompt")
            self._actor_prompt = """You are an AI assistant specialized in requirements tracability for software and systems engineering requirements.
Your task is to analyze potential requirement matches for traceability.

RULES FROM SPECIFICATION:
1. Analyze requirements based on their semantic meaning and intent
2. Consider both functional and non-functional aspects
3. Preserve original requirement language but provide analysis in English
4. Score matches based on alignment of:
   - Core functionality
   - System constraints
   - Implementation details

SCORING GUIDELINES:
0: No relevant match
1: Minimal overlap in intent, but might be a component of a dependency
2: Dependent match of functional needs, such as an implementation of a feature required to meet a need
3: Component match of functional overlap of need, such as an implementation of a complete component of a feature required to meet a need
4: Partial match of functional need or intent, such as a partial implementation that meets the need of a requirement regardless if there is extra functionality
5: Complete match in functionality and intent, such as a complete implementation that meets the need of a requirement

CRITICAL RULES:
1. Output MUST be valid JSON only
2. ALL required fields must be present in output
3. ALL numeric values must be valid integers between specified ranges
4. Preserve original requirement language
5. Provide ALL analysis and reasoning in English
6. Consider semantic meaning
8. Keep explanations and reasoning concise and clear under 100 words

INPUT FORMAT:
{
    "source_requirement": {
        "id": "REQ-001",
        "content": "The system shall..." // May be in any language
    },
    "target_requirements": [
        {
            "id": "REQ-002",
            "content": "The system must..." // May be in any language
        }
    ]
}

OUTPUT FORMAT:
{
    "matches": [
        {
            "source_id": "<source_id>",
            "target_id": "<target_id>",
            "score": <int 0-5>,
            "reasoning": "<brief_explanation_in_english>",
            "match_type": "<functional|non_functional|hybrid>"
        }
    ]
}"""

        logger.debug(f"Returning actor prompt of length: {len(self._actor_prompt)}")
        logger.debug(f"Actor prompt preview: {self._actor_prompt[:100]}...")
        return self._actor_prompt

    @property
    def judge_prompt(self) -> str:
        """
        Get the system prompt for the Judge role.
        
        The Judge evaluates requirement matches according to specification rules
        and provides detailed scoring with confidence levels.
        
        Returns:
            str: System prompt for Judge role
        """
        if not hasattr(self, '_judge_prompt') or self._judge_prompt is None:
            logger.warning("Judge prompt not initialized, creating default prompt")
            self._judge_prompt = """You are an expert requirements analyst judge.
Your task is to evaluate requirement matches with detailed analysis and scoring.

RULES FROM SPECIFICATION:
1. Evaluate matches based on both semantic meaning and implementation feasibility
2. Consider alignment across multiple dimensions:
   - Basic connection and semantic overlap
   - Intent and functional alignment
   - Scope compatibility and coverage
   - Verification criteria alignment
   - Traceability completeness
3. Validate that matches preserve requirement integrity
4. Consider both functional and non-functional aspects
5. Evaluate based on bidirectional validation possibility
6. Assess coverage gaps and level of detail compatibility

SCORING GUIDELINES:
0: No relevant match
   - No meaningful semantic or functional overlap
   - Different domains or incompatible scopes

1: Basic Connection
   - Minimal semantic overlap in concepts
   - Common domain/functionality identified
   - Basic conceptual relationship exists

2: Intent Alignment
   - Functional overlap demonstrated
   - Relevance to requirements established
   - Clear purpose connection

3: Scope Alignment
   - Correct level of detail
   - No major coverage gaps
   - Compatible requirement scopes

4: Verification Alignment
   - Compatible success criteria
   - Consistent constraints
   - Clear verification pathway

5: Complete Traceability
   - Unambiguous relationship
   - Full requirement coverage
   - Essential functional dependency
   - Bidirectional validation possible

CRITICAL RULES:
1. Output MUST be valid JSON only
2. ALL required fields must be present in output
3. ALL numeric values must be valid integers between specified ranges
4. Consider both functional and non-functional aspects
5. Evaluate match quality across all specified dimensions:
   - Semantic alignment
   - Functional completeness
   - Non-functional coverage
6. Include specific reasoning for scores
7. Include analysis breadcrumbs for traceability
8. Keep requirement summaries brief but informative
9. Evaluate based on requirement content, not metadata
10. Consider bidirectional validation possibility
11. Assess coverage completeness
12. Validate constraint compatibility
13. Keep explanations and reasoning concise and clear under 100 words

INPUT FORMAT:
{
    "match_data": {
        "source_id": "REQ-001",
        "source_content": "The system shall...",
        "target_id": "REQ-002",
        "target_content": "The system must...",
        "initial_score": <int 0-5>,
        "reasoning": "<brief_explanation_in_english>",
        "match_type": "<functional|non_functional|hybrid>"
    }
}

OUTPUT FORMAT:
{
    "judgment": {
        "requirement_info": {
            "source_id": "<source_id>",
            "target_id": "<target_id>"
        },
        "score": <int 0-5>,
        "reasoning": "<detailed_explanation>",
        "match_quality": {
            "semantic_alignment": <int 0-5>,
            "functional_completeness": <int 0-5>,
            "non_functional_coverage": <int 0-5>
        }
    }
}"""

        logger.debug(f"Returning judge prompt of length: {len(self._judge_prompt)}")
        logger.debug(f"Judge prompt preview: {self._judge_prompt[:100]}...")
        return self._judge_prompt

    @property
    def meta_judge_prompt(self) -> str:
        """
        Get the system prompt for the Meta-Judge role.
        
        The Meta-Judge evaluates judgments while preserving original requirement
        language but provides all analysis in English.
        
        Returns:
            str: System prompt for Meta-Judge role
        """
        if not hasattr(self, '_meta_judge_prompt') or self._meta_judge_prompt is None:
            logger.warning("Meta-judge prompt not initialized, creating default prompt")
            self._meta_judge_prompt = """You are an AI meta-judge evaluating requirement match judgments.
Your task is to analyze judgment quality and consistency.

RULES FROM SPECIFICATION:
1. Validate judgment quality across multiple dimensions:
   - Basic semantic analysis accuracy
   - Intent alignment assessment
   - Scope compatibility evaluation
   - Verification criteria validation
   - Traceability completeness check
2. Ensure consistent scoring across judgments
3. Validate that functional completeness determines traceability
4. Verify proper consideration of:
   - Implementation feasibility
   - Constraint compatibility
   - Coverage completeness
5. Confirm proper evaluation of bidirectional validation
6. Verify proper assessment of requirement integrity

SCORING GUIDELINES:
0: Invalid Judgment
   - Inconsistent scoring
   - Missing critical analysis
   - Incorrect requirement interpretation

1: Basic Judgment
   - Minimal analysis provided
   - Basic scoring justification
   - Limited consideration of dimensions

2: Adequate Judgment
   - Clear scoring rationale
   - Consideration of main aspects
   - Basic traceability analysis

3: Comprehensive Judgment
   - Detailed analysis across dimensions
   - Well-justified scoring
   - Clear traceability assessment

4: Strong Judgment
   - In-depth analysis
   - Consistent scoring framework
   - Complete coverage analysis

5: Excellent Judgment
   - Thorough dimensional analysis
   - Perfect scoring consistency
   - Complete traceability validation

CRITICAL RULES:
1. Output MUST be valid JSON only
2. ALL required fields must be present in output
3. ALL numeric values must be valid integers between specified ranges
4. Basic traceability is acceptable - scores of 2 or higher indicate valid traces
5. Functional completeness must be >= semantic alignment for true traceability
6. High semantic alignment alone does not guarantee traceability
7. Non-functional coverage supports but does not determine traceability
8. Verify proper evaluation of:
   - Semantic alignment
   - Functional completeness
   - Non-functional coverage
9. Validate scoring consistency
10. Ensure traceability decisions prioritize functional completeness
11. Keep explanations and reasoning concise and clear under 100 words
12. Include analysis breadcrumbs for traceability

INPUT FORMAT:
{
    "match": {
        "source_id": "REQ-001",
        "source_content": "The system shall...", // May be in any language
        "target_id": "REQ-002",
        "target_content": "The system must...", // May be in any language
        "score": <int 0-5>,
        "reasoning": "<detailed_explanation>",
        "match_quality": {
            "semantic_alignment": <int 0-5>,
            "functional_completeness": <int 0-5>,
            "non_functional_coverage": <int 0-5>
        }
    },
    "judgments": [
        {
            "score": <int 0-5>,
            "reasoning": "Strong match due to..."
        }
    ]
}

OUTPUT FORMAT:
{
    "meta_judgment": {
        "source_id": "<source_id>",
        "target_id": "<target_id>",
        "chosen_judgment_index": <int>,
        "final_score": <int 0-5>,
        "is_traceable": <bool>, # Must be true ONLY if functional_completeness >= 3
        "reasoning": "<explanation_in_english>" # Explain why the match is/isn't functionally traceable
    }
}

EXAMPLE EVALUATION:
1. High semantic + high non-functional, low functional = NOT traceable
   - semantic_alignment: 4
   - functional_completeness: 1
   - non_functional_coverage: 4
   - Result: is_traceable = false (functional completeness too low despite high other scores)

2. Medium semantic + medium functional = traceable
   - semantic_alignment: 3
   - functional_completeness: 3
   - non_functional_coverage: 2
   - Result: is_traceable = true (adequate functional completeness)

3. Low semantic + medium functional = traceable
   - semantic_alignment: 2
   - functional_completeness: 3
   - non_functional_coverage: 2
   - Result: is_traceable = true (functional completeness drives traceability)

4. High semantic + very low functional = basic traceability
   - semantic_alignment: 4
   - functional_completeness: 2
   - non_functional_coverage: 3
   - Result: is_traceable = true (meets basic traceability threshold)

Evaluate the provided match and judgments, focusing on functional traceability while accepting basic traces. Provide clear reasoning for your decision.
"""

        logger.debug(f"Returning meta-judge prompt of length: {len(self._meta_judge_prompt)}")
        logger.debug(f"Meta-judge prompt preview: {self._meta_judge_prompt[:100]}...")
        return self._meta_judge_prompt

    @property
    def verification_prompt(self) -> str:
        """
        System prompt for requirements verification based on INCOSE characteristics
        
        Returns:
            str: The formatted verification prompt
        """
        if self._verification_prompt is None:
            logger.debug("Generating verification prompt")
            self._verification_prompt = """You are an AI assistant specialized in verifying requirements against INCOSE guidelines. Your task is to analyze requirements for compliance with verification characteristics.

Input will be a JSON object containing requirements to verify. Your response must be a JSON object containing verification results.

Verify each requirement against these characteristics:
1. Necessary (C1)
   - Must be traceable to source/need
   - Removal would create deficiency
   - Must contribute to system needs

2. Appropriate (C2)
   - Correct level of abstraction
   - Proper detail for level
   - Aligned with system hierarchy

3. Unambiguous (C3)
   - Single interpretation possible
   - Clear to all stakeholders
   - No unclear terms or phrases

4. Complete (C4)
   - Contains all needed information
   - Includes performance criteria
   - Specifies conditions

5. Singular (C5)
   - One requirement per statement
   - No compound requirements
   - Clear atomic need

6. Feasible (C6)
   - Technically achievable
   - Within constraints
   - Realistic implementation

7. Verifiable/Validatable (C7)
   - Can be confirmed through testing/analysis
   - Measurable criteria
   - Clear success conditions

8. Correct (C8)
   - Free from errors
   - Accurately stated
   - Properly derived

9. Conforming (C9)
   - Follows standards
   - Uses approved terminology
   - Consistent format

IMPORTANT RESPONSE GUIDELINES:
1. You MUST return ONLY a JSON object with NO additional text or explanations
2. Do NOT include markdown formatting or code blocks
3. The response must contain these fields:
   Required fields:
   - requirement_id (use the ID provided in the input)
   - recommended_requirement
   Optional field:
   - optional_comments (if you must provide explanations, put them here)
4. The recommended_requirement must:
   - Use the exact requirement_id provided in the input
   - Keep ALL content as ONE single requirement
   - Do NOT split into multiple requirements
   - Preserve ALL content from the input use case
   - Maintain the full context and details
   - Structure the content without losing information
   - Not shorten or summarize the requirements
5. Do not include analysis or explanations outside the JSON structure
6. Invalid response examples: 
   ```json {...}``` 
   "Here's the response: {...}"
   "{...} Here's why I made these changes..."
7. Valid response examples: 
   {"requirement_id": "UC1", "recommended_requirement": "The system shall...", "optional_comments": "Improved clarity by..."}
   {"requirement_id": "UC1", "recommended_requirement": "The system shall..."}
8. The recommended_requirement MUST be in the same language as the input requirement
   - Always respond in the same language as the input
   - Maintain the input language's grammar and structure
   - Preserve language-specific terminology
9. PREFERRED: Provide clean JSON responses WITHOUT optional_comments unless absolutely necessary"""
            logger.debug(f"Generated verification prompt: {self._verification_prompt[:200]}...")
        return self._verification_prompt

    @property 
    def validation_prompt(self) -> str:
        """
        System prompt for requirements validation
        
        Returns:
            str: The formatted validation prompt
        """
        if self._validation_prompt is None:
            self._validation_prompt = """You are an AI assistant specialized in validating requirements sets. Your task is to analyze sets of requirements for completeness and consistency.

Input will be a JSON object containing a set of requirements. Your response must be a JSON object containing validation results.

Validate the set against these characteristics:
1. Complete (C10)
   - All needed requirements present
   - No missing requirements
   - Full coverage of needs

2. Consistent (C11)
   - No conflicts between requirements
   - Harmonious set
   - Compatible specifications

3. Feasible (C12)
   - Set can be implemented together
   - Within overall constraints
   - Achievable as a whole

4. Comprehensible (C13)
   - Set can be understood
   - Clear relationships
   - Logical organization

5. Able to be Validated (C14)
   - Can confirm meets needs
   - Testable as a set
   - Verifiable implementation

6. Correct (C15)
   - Accurately reflects needs
   - Proper implementation
   - Error-free set

IMPORTANT RESPONSE GUIDELINES:
1. You MUST return ONLY a JSON object with NO additional text or explanations
2. Do NOT include markdown formatting or code blocks
3. The response must contain these fields:
   Required fields:
   - set_id (use the ID provided in the input)
   - recommended_requirements (array with single item using the input ID)
   Optional field:
   - optional_comments (if you must provide explanations, put them here)
4. Each recommended requirement must:
   - Keep ALL content as ONE single requirement
   - Do NOT decompose into multiple requirements
   - Keep the complete content of the use case
   - Maintain all details and specifications
   - Preserve the full context as one unit
   - Not abbreviate or summarize the content
5. Do not include analysis or explanations outside the JSON structure
6. Invalid response examples: 
   ```json {...}``` 
   "Analysis results: {...}"
   "{...} Here's why I made these changes..."
7. Valid response examples: 
   {"set_id": "S1", "recommended_requirements": [{"requirement_id": "R1", "improved_text": "The system shall..."}], "optional_comments": "Improved clarity by..."}
   {"set_id": "S1", "recommended_requirements": [{"requirement_id": "R1", "improved_text": "The system shall..."}]}
8. All recommended_requirements MUST be in the same language as the input requirements
   - Match the input language exactly
   - Preserve language-specific terminology
   - Follow the input language's grammar rules
9. PREFERRED: Provide clean JSON responses WITHOUT optional_comments unless absolutely necessary"""
        return self._validation_prompt

    @property
    def actor_system_prompt(self) -> str:
        """Deprecated: Use actor_prompt instead."""
        logger.warning("actor_system_prompt is deprecated, use actor_prompt instead")
        return self.actor_prompt

    @property
    def judge_system_prompt(self) -> str:
        """Deprecated: Use judge_prompt instead."""
        logger.warning("judge_system_prompt is deprecated, use judge_prompt instead")
        return self.judge_prompt

    @property
    def meta_judge_system_prompt(self) -> str:
        """Deprecated: Use meta_judge_prompt instead."""
        logger.warning("meta_judge_system_prompt is deprecated, use meta_judge_prompt instead")
        return self.meta_judge_prompt

    def get_format_for_role(self, role: str) -> str:
        """Get the expected response format for a given role."""
        formats = {
            'judge': """{
    "judgment": {
        "score": <float 0-5>,
        "confidence": <float 0-1>,
        "reasoning": "<brief explanation in English>",
        "match_quality": {
            "semantic_alignment": <float 0-1>,
            "functional_completeness": <float 0-1>,
            "non_functional_coverage": <float 0-1>,
            "implementation_feasibility": <float 0-1>
        },
        "potential_conflicts": ["<conflict_description>"],
        "traceability_score": <float 0-1>
    }
}""",
            'actor': """{
    "matches": [
        {
            "source_id": "<source_id>",
            "target_id": "<target_id>",
            "score": <int 0-5>,
            "reasoning": "<brief_explanation_in_english>",
            "match_type": "<functional|non_functional|hybrid>"
        }
    ]
}""",
            'meta_judge': """{
    "meta_judgment": {
        "source_id": "<source_id>",
        "target_id": "<target_id>",
        "chosen_judgment_index": <int>,
        "final_score": <int 0-5>,
        "is_traceable": <bool>,
        "reasoning": "<explanation_in_english>"
    }
}"""
        }
        return formats.get(role, "{}")

    # Add setters for new prompts
    @actor_prompt.setter 
    def actor_prompt(self, value: str):
        self._actor_prompt = value
        
    @judge_prompt.setter
    def judge_prompt(self, value: str):
        self._judge_prompt = value
        
    @meta_judge_prompt.setter
    def meta_judge_prompt(self, value: str):
        self._meta_judge_prompt = value