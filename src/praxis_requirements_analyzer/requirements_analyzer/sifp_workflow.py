"""
# src/requirements_analyzer/sifp_workflow.py

SIFP Workflow Module

This module implements the workflow for SIFP estimation using LLMs and vector similarity.
It provides a complete pipeline for analyzing software requirements and estimating
SIFP points using the UGEP and UGDG approach.

Example:
    >>> workflow = SIFPWorkflow(llm_manager, prompt_manager, "claude-2", redis_client, embedding_client)
    >>> estimations = await workflow.process_requirements_batch(requirements)
    >>> for estimation in estimations:
    ...     print(f"Total SIFP points: {estimation['sifp_points']['total']}")

Classes:
    SIFPWorkflow: Manages the complete workflow for SIFP estimation

Attributes:
    logger: Logger instance for this module
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import json
from ..utils.logger import setup_logger, handle_exception
from ..models.requirement import Requirement
from ..llm.manager import LLMManager
from .sifp_prompt_manager import SIFPPromptManager
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
import asyncio
import re

logger = setup_logger(__name__, logging.DEBUG)

class SIFPWorkflow:
    """
    Implements LangGraph workflow for SIFP estimation using vector similarity and Meta-Judge approach.
    
    This class manages the complete pipeline for analyzing requirements and estimating SIFP points:
    - Vector similarity search using Redis
    - Actor role for UGEP/UGDG identification
    - Judge role for estimation validation
    - Meta-judge role for consistency checking
    
    Attributes:
        llm_manager (LLMManager): Manager for LLM interactions
        prompt_manager (SIFPPromptManager): Manager for prompt templates
        model_name (str): Name of the LLM model to use
        redis_client (Any): Redis client for vector storage
        embedding_client (Any): Client for generating embeddings
        logger (Logger): Logger instance for this class
        req_schema (Dict): Schema for requirements index
        req_index (AsyncSearchIndex): RedisVL index for requirements
        
    Example:
        >>> workflow = SIFPWorkflow(
        ...     llm_manager=llm_manager,
        ...     prompt_manager=prompt_manager,
        ...     model_name="claude-2",
        ...     redis_client=redis_client,
        ...     embedding_client=embedding_client
        ... )
        >>> estimations = await workflow.process_requirements_batch(requirements)
    """
    
    def __init__(self,
                 llm_manager: LLMManager,
                 prompt_manager: SIFPPromptManager,
                 model_name: str,
                 redis_client: Any,
                 embedding_client: Any) -> None:
        """
        Initialize the SIFP workflow with necessary components.
        
        Args:
            llm_manager: Manager for LLM interactions
            prompt_manager: Manager for prompt templates
            model_name: Name of the LLM model to use
            redis_client: Redis client for vector storage
            embedding_client: Client for generating embeddings
        
        Raises:
            ValueError: If Redis client is not properly initialized
        """
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        self.model_name = model_name
        self.redis_client = redis_client
        self.embedding_client = embedding_client
        self.logger = setup_logger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Verify Redis client
        if not redis_client or not redis_client.client:
            raise ValueError("Redis client must be initialized before creating workflow")
        
        try:
            # Test Redis client connection
            asyncio.create_task(redis_client.client.ping())
        except Exception as e:
            raise ValueError(f"Redis client failed connection test: {str(e)}")

        # Define schema for requirements index
        self.req_schema = {
            "index": {
                "name": "sifp_reqs",
                "prefix": "req:",
                "storage_type": "hash"
            },
            "fields": [
                {"name": "text", "type": "text"},
                {"name": "metadata", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 384,
                        "distance_metric": "cosine",
                        "algorithm": "hnsw",
                        "datatype": "float32"
                    }
                }
            ]
        }

        # Initialize RedisVL index
        self.req_index = AsyncSearchIndex.from_dict(self.req_schema)

        self.logger.debug("Initialized SIFPWorkflow with:")
        self.logger.debug(f"Model name: {model_name}")
        self.logger.debug(f"Redis client type: {type(redis_client)}")
        self.logger.debug(f"Embedding client type: {type(embedding_client)}")

    def _serialize_vector(self, vector: List[float]) -> bytes:
        """
        Serialize vector to binary format for Redis storage.
        
        Args:
            vector: List of float values representing the embedding
            
        Returns:
            bytes: Binary representation of the vector
        """
        try:
            # Convert to numpy array if needed
            if not isinstance(vector, np.ndarray):
                vector = np.array(vector, dtype=np.float32)
            
            # Ensure correct shape and type
            vector = vector.astype(np.float32)
            
            # Pack into bytes
            return vector.tobytes()
            
        except Exception as e:
            self.logger.error(f"Error serializing vector: {str(e)}")
            raise

    def _prepare_redis_record(self, key: str, content: str, embedding: List[float], req_id: str = None) -> Dict:
        """
        Prepare a record for Redis storage with proper format.
        
        Args:
            key: The Redis key for the record
            content: The text content
            embedding: The vector embedding
            req_id: Optional requirement ID for logging
            
        Returns:
            Dict containing the fields in Redis-compatible format
        """
        try:
            # Convert embedding to string if needed
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # Return just the fields dictionary for Redis hset
            fields = {
                "content": content,
                "embedding": self._serialize_vector(embedding)
            }
            
            log_prefix = f"[Req: {req_id}]" if req_id else ""
            self.logger.debug(f"{log_prefix} Prepared Redis fields for key {key}: {fields}")
            return fields
            
        except Exception as e:
            log_prefix = f"[Req: {req_id}]" if req_id else ""
            self.logger.error(f"{log_prefix} Error preparing Redis record: {str(e)}")
            self.logger.error(f"{log_prefix} Input types: key={type(key)}, content={type(content)}, embedding={type(embedding)}")
            raise

    async def init_indices(self):
        """Initialize Redis indices"""
        try:
            # Verify Redis client is available
            if not self.redis_client or not self.redis_client.client:
                raise ValueError("Redis client not properly initialized")
            
            # Verify Redis connection
            try:
                await self.redis_client.client.ping()
                self.logger.info("Redis connection verified")
            except Exception as e:
                self.logger.error(f"Redis connection failed: {str(e)}")
                raise

            # Clear any existing indices first
            await self.clear_indices()

            # Create index with Redis client passed directly to constructor
            try:
                # Recreate index with client passed in constructor
                self.req_schema['client'] = self.redis_client.client
                self.req_index = AsyncSearchIndex.from_dict(self.req_schema)
                self.logger.info("Set Redis client for index")
            except Exception as e:
                self.logger.error(f"Failed to set Redis client for index: {str(e)}")
                raise
            
            # Create index
            try:
                await self.req_index.create(overwrite=True)
                self.logger.info("Created index successfully")
                index_info = await self.req_index.info()
                self.logger.info(f"Index info: {index_info}")
            except Exception as e:
                self.logger.error(f"Failed to create index: {str(e)}")
                raise

        except Exception as e:
            self.logger.error(f"Failed to create indices: {str(e)}")
            raise

    async def clear_indices(self):
        """Clear existing indices before processing new requirements."""
        try:
            self.logger.info("Clearing existing indices")
            
            # Delete all keys with requirements prefix
            req_keys = await self.redis_client.client.keys(f"{self.req_schema['index']['prefix']}*")
            if req_keys:
                self.logger.debug(f"Found {len(req_keys)} existing keys: {[k.decode() for k in req_keys]}")
                await self.redis_client.client.delete(*req_keys)
                self.logger.info(f"Deleted {len(req_keys)} keys")

        except Exception as e:
            self.logger.error(f"Error clearing indices: {str(e)}")
            raise

    @handle_exception("Error pre-processing requirements")
    async def preprocess_requirements(self, requirements: List[Requirement]) -> List[Dict]:
        """
        Preprocess requirements by calculating embeddings and formatting for Redis storage.
        
        Args:
            requirements: List of requirements to process
            
        Returns:
            List of processed records ready for Redis storage
        """
        try:
            self.logger.info("Starting requirements preprocessing")
            self.logger.debug(f"Input requirements: {[{'id': r.id, 'type': r.type} for r in requirements]}")
            
            if not requirements:
                self.logger.warning("No requirements to preprocess")
                return []
            
            records = []
            
            # Process requirements
            self.logger.info(f"Processing {len(requirements)} requirements")
            for req in requirements:
                try:
                    self.logger.debug(f"[Req: {req.id}] Processing requirement of type {req.type}")
                    self.logger.debug(f"[Req: {req.id}] Content: {req.content[:100]}...")
                    
                    embedding = self.embedding_client.embed(req.content)
                    self.logger.debug(f"[Req: {req.id}] Created embedding")
                    
                    # Convert embedding to binary format
                    embedding_bytes = self._serialize_vector(embedding)
                    
                    record = {
                        "id": f"req:{req.id}",
                        "text": req.content,
                        "metadata": json.dumps({
                            "id": req.id,
                            "type": req.type,
                            "level": req.level
                        }),
                        "embedding": embedding_bytes
                    }
                    records.append(record)
                    self.logger.debug(f"[Req: {req.id}] Added record")
                    
                except Exception as e:
                    self.logger.error(f"[Req: {req.id}] Failed to process requirement: {str(e)}", exc_info=True)
                    raise

            self.logger.info(f"Completed preprocessing {len(records)} records")
            return records

        except Exception as e:
            self.logger.error(f"Error in preprocessing requirements: {str(e)}", exc_info=True)
            raise

    @handle_exception("Error processing requirements batch")
    async def process_requirements_batch(self, requirements: List[Requirement]) -> List[Dict[str, Any]]:
        """
        Process a batch of requirements through the complete SIFP estimation pipeline.
        
        Args:
            requirements: List of requirements to process
            
        Returns:
            List[Dict[str, Any]]: List of final estimations with scores and evaluations
        """
        try:
            self.logger.debug(f"Starting batch processing of {len(requirements)} requirements")
            self.logger.debug(f"Requirements data: {[{'id': r.id, 'type': r.type, 'content': r.content[:100]+'...'} for r in requirements]}")
            
            # Clear and prepare indices
            await self.clear_indices()
            
            if not requirements:
                self.logger.warning("No requirements provided for processing")
                return []
            
            records = await self.preprocess_requirements(requirements)
            self.logger.debug(f"Preprocessed {len(records)} records")
            
            # Load records into index
            if records:
                self.logger.debug(f"Loading {len(records)} records into index")
                await self.req_index.load(records, keys=[r["id"] for r in records])
                self.logger.debug("Records loaded into index")
            else:
                self.logger.warning("No records to load into index")
                return []

            final_estimations = []
            
            # Process each requirement
            for record in records:
                try:
                    metadata = json.loads(record["metadata"])
                    self.logger.debug(f"Processing requirement {metadata['id']} of type {metadata['type']}")
                    
                    # Get similar requirements for context
                    embedding = np.frombuffer(record["embedding"], dtype=np.float32)
                    self.logger.debug(f"Generated embedding for requirement {metadata['id']}")
                    
                    query = VectorQuery(
                        vector=embedding.tolist(),
                        vector_field_name="embedding",
                        return_fields=["text", "metadata", "vector_distance"],
                        num_results=5
                    )
                    
                    vector_results = await self.req_index.query(query)
                    self.logger.debug(f"Found {len(vector_results)} similar requirements")
                    
                    # Prepare data for actor evaluation
                    requirement_data = {
                        "id": metadata["id"],
                        "content": record["text"],
                        "type": metadata["type"]
                    }
                    
                    # Get actor evaluation (UGEP/UGDG identification)
                    self.logger.debug(f"Getting actor evaluation for requirement {requirement_data['id']}")
                    actor_analysis = await self.evaluate_with_actor(requirement_data)
                    self.logger.debug(f"Actor analysis complete for {requirement_data['id']}")
                    
                    # Get judge evaluation
                    judgment = await self.evaluate_with_judge({
                        "requirement_id": requirement_data["id"],
                        "requirement_content": requirement_data["content"],
                        "type": requirement_data["type"],
                        "analysis": actor_analysis
                    })
                    self.logger.debug(f"Judge evaluation complete for {requirement_data['id']}")
                    
                    # Get meta-judge evaluation
                    meta_judgment = await self.evaluate_with_meta_judge(
                        requirement_data,
                        [judgment]
                    )
                    self.logger.debug(f"Meta-judge evaluation complete for {requirement_data['id']}")
                    
                    # Store final estimation data
                    final_estimation = {
                        "requirement_id": requirement_data["id"],
                        "actor_analysis": actor_analysis,
                        "judge_score": judgment.get("score", 0),
                        "judge_confidence": judgment.get("confidence", 0),
                        "judge_evaluation": judgment.get("evaluation", {}),
                        "final_estimation": meta_judgment.get("final_estimation", {}),
                        "is_valid": meta_judgment.get("is_valid", False),
                        "confidence": meta_judgment.get("confidence", 0),
                        "reasoning": meta_judgment.get("reasoning", "")
                    }
                    
                    final_estimations.append(final_estimation)
                    self.logger.debug(f"Added final estimation for {requirement_data['id']}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing requirement: {str(e)}", exc_info=True)
                    continue
                    
            self.logger.debug(f"Completed batch processing with {len(final_estimations)} estimations")
            return final_estimations
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}", exc_info=True)
            raise

    def _clean_mathematical_expressions(self, json_str: str) -> str:
        """
        Clean up mathematical expressions in JSON values to make them valid JSON.
        
        Args:
            json_str: JSON string that may contain mathematical expressions
            
        Returns:
            str: Cleaned JSON string with evaluated mathematical expressions
        """
        import re
        
        try:
            # First pass: Remove comments and evaluate expressions in them
            def evaluate_comment_expr(match):
                try:
                    expr = match.group(1).strip()
                    if '+' in expr or '*' in expr:
                        # Basic arithmetic evaluation
                        return ''  # Remove the comment entirely
                except:
                    pass
                return ''  # Remove comment if we can't evaluate
            
            # Remove comments with expressions
            json_str = re.sub(r'//\s*([^,\n}]*)', evaluate_comment_expr, json_str)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)  # Remove multi-line comments
            
            # Second pass: Evaluate mathematical expressions in values
            def evaluate_math_expr(match):
                try:
                    expr = match.group(1).strip()
                    if '+' in expr or '*' in expr:
                        # Remove any whitespace around operators
                        expr = re.sub(r'\s*([\+\*])\s*', r'\1', expr)
                        # Split by operators while preserving them
                        parts = re.findall(r'[\d.]+|[\+\*]', expr)
                        result = float(parts[0])
                        for i in range(1, len(parts), 2):
                            op, num = parts[i], float(parts[i+1])
                            if op == '*':
                                result *= num
                            elif op == '+':
                                result += num
                        return str(result)
                    return expr
                except:
                    return expr
            
            # Find and evaluate expressions like "4.6 + 7.0" or "4.6 * 6"
            json_str = re.sub(r':\s*([\d.]+(?:\s*[\+\*]\s*[\d.]+)+)(?=\s*[,}\]])', 
                            lambda m: f': {evaluate_math_expr(m)}', 
                            json_str)
            
            # Third pass: Clean up any trailing commas
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            # Fourth pass: Remove any remaining whitespace between values
            json_str = re.sub(r'\s+', ' ', json_str)
            
            return json_str
            
        except Exception as e:
            self.logger.warning(f"Error cleaning mathematical expressions: {str(e)}")
            return json_str

    def extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON from LLM response, handling various formats.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            str: The extracted JSON string
            
        Raises:
            ValueError: If valid JSON cannot be extracted
        """
        self.logger.debug(f"Extracting JSON from response of length {len(response)}")
        
        # Try different extraction methods
        extraction_attempts = []
        
        # First attempt: Check if the response is already valid JSON
        try:
            json.loads(response)
            self.logger.debug("Response is already valid JSON")
            return response
        except json.JSONDecodeError:
            extraction_attempts.append("Direct JSON parsing failed")
        
        # Second attempt: Try to extract JSON from Markdown code blocks
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        code_matches = re.findall(code_block_pattern, response)
        
        for match in code_matches:
            # Try to clean the match of comments before parsing
            clean_match = self._clean_json_comments(match)
            try:
                json.loads(clean_match)
                self.logger.debug("Extracted JSON from code block")
                return clean_match
            except json.JSONDecodeError:
                extraction_attempts.append(f"Code block extraction failed for: {match[:50]}...")
        
        # Third attempt: Look for JSON-like structures with curly braces
        try:
            # Find the first { and the last } in the text
            start = response.find('{')
            end = response.rfind('}')
            
            if start != -1 and end != -1 and start < end:
                potential_json = response[start:end+1]
                # Clean comments before parsing
                clean_json = self._clean_json_comments(potential_json)
                json.loads(clean_json)
                self.logger.debug("Extracted JSON using brace matching")
                return clean_json
        except json.JSONDecodeError:
            extraction_attempts.append("Brace matching extraction failed")
        
        # All extraction attempts failed
        self.logger.error(f"Extraction attempts failed:\n{extraction_attempts}")
        self.logger.error(f"Response content preview: {response[:100]}...")
        raise ValueError("Could not extract valid JSON from response after multiple attempts")

    def _clean_json_comments(self, json_str: str) -> str:
        """
        Clean JSON string by removing comments and other invalid JSON elements.
        
        Args:
            json_str: JSON string possibly containing comments
            
        Returns:
            str: Cleaned JSON string
        """
        # Remove // comments
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        
        # Remove /* */ comments
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Remove trailing commas (which are invalid in standard JSON)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Normalize whitespace
        json_str = json_str.strip()
        
        return json_str

    async def validate_and_fix_json_response(self, response: str, role: str) -> Union[Dict, List]:
        """
        Validate and fix JSON responses from LLMs with enhanced error handling.
        
        Args:
            response: Raw response string from LLM
            role: Role that generated the response (actor/judge/meta-judge)
        
        Returns:
            Dict or List: Validated and potentially fixed JSON structure
            
        Raises:
            ValueError: If JSON cannot be validated or fixed
        """
        try:
            # Clean up the response to extract just the JSON part
            json_str = self.extract_json_from_response(response)
            
            # Parse the JSON
            result = json.loads(json_str)
            
            # Handle role-specific fixes
            if role == 'actor':
                # If response is missing analysis wrapper, add it
                if 'analysis' not in result and 'requirement_id' in result:
                    self.logger.warning("Adding missing analysis wrapper to actor response")
                    result = {"analysis": result}
                    
            elif role == 'judge':
                # If response is missing judgment wrapper, add it
                if 'judgment' not in result and 'requirement_id' in result:
                    self.logger.warning("Adding missing judgment wrapper to judge response")
                    result = {"judgment": result}
                
            elif role == 'meta_judge':
                # If response is missing meta_judgment wrapper, add it
                if 'meta_judgment' not in result and 'requirement_id' in result:
                    self.logger.warning("Adding missing meta_judgment wrapper to meta-judge response")
                    result = {"meta_judgment": result}
                    
            # Validate required fields based on role
            self.validate_role_specific_fields(result, role)
                    
            # Format JSON with indentation for better readability in logs
            formatted_json = json.dumps(result, indent=2)
            self.logger.debug(f"Validated {role} response structure:\n{formatted_json}")
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON validation failed for {role}: {str(e)}")
            self.logger.error(f"Response content: {response[:200]}...")
            raise ValueError(f"Invalid JSON from {role}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error in JSON validation: {str(e)}")
            raise

    def validate_role_specific_fields(self, result: Dict, role: str) -> None:
        """
        Validate that the JSON response contains required fields for each role.
        
        Args:
            result: Parsed JSON response
            role: Role that generated the response
            
        Raises:
            ValueError: If required fields are missing
        """
        try:
            if role == 'actor':
                analysis = result.get('analysis', {})
                required_fields = {'requirement_id', 'ugeps', 'ugdgs', 'sifp_points', 'confidence'}
                missing = required_fields - set(analysis.keys())
                if missing:
                    raise ValueError(f"Actor response missing required fields: {missing}")
                    
            elif role == 'judge':
                judgment = result.get('judgment', {})
                required_fields = {'requirement_id', 'score', 'confidence', 'evaluation', 'reasoning'}
                missing = required_fields - set(judgment.keys())
                if missing:
                    raise ValueError(f"Judge response missing required fields: {missing}")
                    
            elif role == 'meta_judge':
                meta_judgment = result.get('meta_judgment', {})
                required_fields = {'requirement_id', 'final_score', 'confidence', 'is_valid', 'reasoning'}
                missing = required_fields - set(meta_judgment.keys())
                if missing:
                    raise ValueError(f"Meta-judge response missing required fields: {missing}")
                    
        except Exception as e:
            self.logger.error(f"Field validation failed for {role}: {str(e)}")
            raise ValueError(f"Invalid response structure for {role}: {str(e)}")

    async def evaluate_with_actor(self, requirement: Dict) -> Dict:
        """
        Evaluate a requirement using the actor role to identify UGEPs and UGDGs.
        
        Expected output format from prompt:
        {
            "analysis": {
                "requirement_id": str,
                "ugeps": [{
                    "description": str,
                    "is_auxiliary": bool,
                    "weight": float,
                    "justification": str
                }],
                "ugdgs": [{
                    "description": str,
                    "is_fundamental": bool,
                    "weight": float,
                    "justification": str
                }],
                "sifp_points": {
                    "add": float,
                    "aux": float,
                    "total": float
                },
                "confidence": float
            }
        }
        """
        current_role = "ACTOR"
        try:
            # Format input data for debugging
            input_data = {
                "requirement": requirement
            }
            self.logger.debug(f"Actor input data:\n{json.dumps(input_data, indent=2)}")
            
            # Get actor response
            response = await self.llm_manager.agenerate(
                model_name=self.model_name,
                messages=[{
                    "role": "user",
                    "content": json.dumps(input_data)
                }],
                prompt=self.prompt_manager.actor_prompt,
                current_role=current_role
            )
            
            # Validate and fix JSON if needed
            result = await self.validate_and_fix_json_response(response, 'actor')
            
            # Extract analysis from result
            analysis = result.get("analysis")
            if not analysis:
                raise ValueError("Actor response missing 'analysis' field")
            
            # Return the analysis directly
            return analysis

        except Exception as e:
            self.logger.error(f"Actor evaluation failed: {str(e)}")
            raise

    async def evaluate_with_judge(self, estimation_data: Dict) -> Dict:
        """
        Evaluate a SIFP estimation using the judge role.
        
        Expected output format from prompt:
        {
            "judgment": {
                "requirement_id": str,
                "score": int,  # 0-5
                "confidence": float,  # 0-1
                "evaluation": {
                    "ugep_accuracy": float,  # 0-1
                    "ugdg_accuracy": float,  # 0-1
                    "calculation_accuracy": float  # 0-1
                },
                "reasoning": str,
                "suggested_corrections": {
                    "ugeps": [...],
                    "ugdgs": [...],
                    "sifp_points": {...}
                }
            }
        }
        """
        current_role = "JUDGE"
        try:
            # Format input data for debugging
            input_data = {
                "estimation": estimation_data
            }
            self.logger.debug(f"Judge input data:\n{json.dumps(input_data, indent=2)}")
            
            # Get judge response
            response = await self.llm_manager.agenerate(
                model_name=self.model_name,
                messages=[{
                    "role": "user",
                    "content": json.dumps(input_data)
                }],
                prompt=self.prompt_manager.judge_prompt,
                current_role=current_role
            )
            
            # Validate and fix JSON if needed
            result = await self.validate_and_fix_json_response(response, 'judge')
            
            # Extract judgment from result
            judgment = result.get("judgment")
            if not judgment:
                raise ValueError("Judge response missing 'judgment' field")
            
            # Return the judgment directly
            return judgment

        except Exception as e:
            self.logger.error(f"Judge evaluation failed: {str(e)}")
            raise

    async def evaluate_with_meta_judge(self, requirement: Dict, judgments: List[Dict]) -> Dict:
        """
        Use Meta-Judge to evaluate multiple judgments and ensure estimation consistency.
        
        Expected output format from prompt:
        {
            "meta_judgment": {
                "requirement_id": str,
                "chosen_judgment_index": int,
                "final_score": int,  # 0-5
                "confidence": float,  # 0-1
                "is_valid": bool,
                "reasoning": str,
                "final_estimation": {
                    "ugeps": [...],
                    "ugdgs": [...],
                    "sifp_points": {
                        "add": float,
                        "aux": float,
                        "total": float
                    }
                }
            }
        }
        """
        current_role = "META-JUDGE"
        try:
            # Format input data for debugging
            input_data = {
                "requirement": requirement,
                "judgments": judgments
            }
            self.logger.debug(f"Meta-judge input data:\n{json.dumps(input_data, indent=2)}")
            
            # Get meta-judge response
            response = await self.llm_manager.agenerate(
                model_name=self.model_name,
                messages=[{
                    "role": "user",
                    "content": json.dumps(input_data)
                }],
                prompt=self.prompt_manager.meta_judge_prompt,
                current_role=current_role
            )
            
            # Validate and fix JSON if needed
            result = await self.validate_and_fix_json_response(response, 'meta_judge')
            
            # Extract meta_judgment from result
            meta_judgment = result.get("meta_judgment")
            if not meta_judgment:
                raise ValueError("Meta-judge response missing 'meta_judgment' field")
            
            # Return the meta_judgment directly
            return meta_judgment

        except Exception as e:
            self.logger.error(f"Meta-judge evaluation failed: {str(e)}")
            raise

    def validate_actor_input(self, input_data: Dict) -> bool:
        """Validate actor input format matches prompt specification."""
        try:
            required_fields = {"id", "content", "type"}
            requirement = input_data.get("requirement", {})
            
            if not all(field in requirement for field in required_fields):
                self.logger.error(f"Missing required fields. Expected: {required_fields}")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Actor input validation failed: {str(e)}")
            return False

    def validate_judge_input(self, input_data: Dict) -> bool:
        """Validate judge input format matches prompt specification."""
        try:
            required_fields = {
                "requirement_id", "requirement_content", "type",
                "analysis"
            }
            
            estimation = input_data.get("estimation", {})
            
            if not all(field in estimation for field in required_fields):
                self.logger.error(f"Missing required fields. Expected: {required_fields}")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Judge input validation failed: {str(e)}")
            return False

    def validate_meta_judge_input(self, input_data: Dict) -> bool:
        """Validate meta-judge input format matches prompt specification."""
        try:
            required_requirement_fields = {"id", "content", "type"}
            required_judgment_fields = {"score", "confidence", "evaluation"}
            
            requirement = input_data.get("requirement", {})
            judgments = input_data.get("judgments", [])
            
            if not all(field in requirement for field in required_requirement_fields):
                self.logger.error(f"Missing required requirement fields. Expected: {required_requirement_fields}")
                return False
            
            for judgment in judgments:
                if not all(field in judgment for field in required_judgment_fields):
                    self.logger.error(f"Missing required judgment fields. Expected: {required_judgment_fields}")
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"Meta-judge input validation failed: {str(e)}")
            return False 