"""
Requirements Workflow Module

This module implements the workflow for analyzing and matching requirements using LLMs 
and vector similarity. It provides a complete pipeline for requirement matching including 
vector search and multi-stage LLM evaluation.

Example:
    >>> workflow = RequirementsWorkflow(llm_manager, prompt_manager, "claude-2", redis_client, embedding_client)
    >>> matches = await workflow.process_requirements_batch(source_reqs, target_reqs)
    >>> for match in matches:
    ...     print(f"Match score: {match['final_score']}")

Classes:
    RequirementsWorkflow: Manages the complete workflow for requirements analysis and matching

Attributes:
    logger: Logger instance for this module
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import json
from src.utils.logger import setup_logger, handle_exception
from src.models.requirement import Requirement
from src.llm.manager import LLMManager
from .requirements_prompt_manager import RequirementsPromptManager
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
import asyncio
from asyncio import Semaphore
from itertools import islice
import os

logger = setup_logger(__name__)

class RequirementsWorkflow:
    """
    Implements LangGraph workflow for requirements analysis using vector similarity and Meta-Judge approach.
    
    This class manages the complete pipeline for matching and evaluating requirements including:
    - Vector similarity search using Redis
    - Actor role for initial match evaluation
    - Judge role for detailed scoring
    - Meta-judge role for validation
    
    Attributes:
        llm_manager (LLMManager): Manager for LLM interactions
        prompt_manager (RequirementsPromptManager): Manager for prompt templates
        model_name (str): Name of the LLM model to use
        redis_client (Any): Redis client for vector storage
        embedding_client (Any): Client for generating embeddings
        logger (Logger): Logger instance for this class
        src_schema (Dict): Schema for source requirements index
        tgt_schema (Dict): Schema for target requirements index
        src_index (AsyncSearchIndex): RedisVL index for source requirements
        tgt_index (AsyncSearchIndex): RedisVL index for target requirements
        
    Example:
        >>> workflow = RequirementsWorkflow(
        ...     llm_manager=llm_manager,
        ...     prompt_manager=prompt_manager,
        ...     model_name="claude-2",
        ...     redis_client=redis_client,
        ...     embedding_client=embedding_client
        ... )
        >>> matches = await workflow.process_requirements_batch(source_reqs, target_reqs)
    """
    
    def __init__(self,
                 llm_manager: LLMManager,
                 prompt_manager: RequirementsPromptManager,
                 model_name: str,
                 redis_client: Any,
                 embedding_client: Any) -> None:
        """
        Initialize the requirements workflow with necessary components.
        
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

        # Define schema for source and target indices with updated vector configuration
        self.src_schema = {
            "index": {
                "name": "src_reqs",
                "prefix": "src_req:",
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
                        "algorithm": "flat",  # Changed to flat for testing
                        "datatype": "float32"
                    }
                }
            ]
        }
        
        self.tgt_schema = {
            "index": {
                "name": "tgt_reqs", 
                "prefix": "tgt_req:",
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

        # Initialize RedisVL indices without Redis client yet
        self.src_index = AsyncSearchIndex.from_dict(self.src_schema)
        self.tgt_index = AsyncSearchIndex.from_dict(self.tgt_schema)

        self.logger.debug("Initialized RequirementsWorkflow with:")
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

            # Set Redis client for both indices
            try:
                await self.src_index.set_client(self.redis_client.client)
                await self.tgt_index.set_client(self.redis_client.client)
                self.logger.info("Set Redis clients for indices")
            except Exception as e:
                self.logger.error(f"Failed to set Redis clients for indices: {str(e)}")
                raise
            
            # Create source index
            try:
                await self.src_index.create(overwrite=True)
                self.logger.info("Created source index successfully")
                src_info = await self.src_index.info()
                self.logger.info(f"Source index info: {src_info}")
            except Exception as e:
                self.logger.error(f"Failed to create source index: {str(e)}")
                raise

            # Create target index
            try:
                await self.tgt_index.create(overwrite=True)
                self.logger.info("Created target index successfully")
                tgt_info = await self.tgt_index.info()
                self.logger.info(f"Target index info: {tgt_info}")
            except Exception as e:
                self.logger.error(f"Failed to create target index: {str(e)}")
                raise

        except Exception as e:
            self.logger.error(f"Failed to create indices: {str(e)}")
            raise

    async def clear_indices(self):
        """Clear existing indices before processing new requirements."""
        try:
            self.logger.info("Clearing existing indices")
            
            # Delete all keys with source prefix
            src_keys = await self.redis_client.client.keys(f"{self.src_schema['index']['prefix']}*")
            if src_keys:
                self.logger.debug(f"Found {len(src_keys)} existing source keys: {[k.decode() for k in src_keys]}")
                await self.redis_client.client.delete(*src_keys)
                self.logger.info(f"Deleted {len(src_keys)} source keys")

            # Delete all keys with target prefix
            tgt_keys = await self.redis_client.client.keys(f"{self.tgt_schema['index']['prefix']}*")
            if tgt_keys:
                self.logger.debug(f"Found {len(tgt_keys)} existing target keys: {[k.decode() for k in tgt_keys]}")
                await self.redis_client.client.delete(*tgt_keys)
                self.logger.info(f"Deleted {len(tgt_keys)} target keys")

        except Exception as e:
            self.logger.error(f"Error clearing indices: {str(e)}")
            raise

    @handle_exception("Error pre-processing requirements")
    async def preprocess_requirements(self,
                                   source_requirements: List[Requirement],
                                   target_requirements: List[Requirement]) -> Tuple[List[Dict], List[Dict]]:
        """
        Preprocess requirements by calculating embeddings and formatting for Redis storage.
        
        Args:
            source_requirements: List of source requirements
            target_requirements: List of target requirements
            
        Returns:
            Tuple containing processed source and target records
        """
        self.logger.info("Starting requirements preprocessing")
        
        source_records = []
        target_records = []
        
        # Process source requirements
        self.logger.info(f"Processing {len(source_requirements)} source requirements")
        for source in source_requirements:
            try:
                self.logger.info(f"Processing source requirement {source.id}: Creating embedding...")
                source_embedding = self.embedding_client.embed(source.content)
                
                # Convert embedding to binary format
                embedding_bytes = self._serialize_vector(source_embedding)
                
                source_record = {
                    "id": f"src_req:{source.id}",
                    "text": source.content,
                    "metadata": json.dumps({
                        "id": source.id,
                        "type": source.type,
                        "level": source.level
                    }),
                    "embedding": embedding_bytes  # Store as binary
                }
                source_records.append(source_record)
                self.logger.debug(f"Processing source requirement {source.id}: Processed source requirement")
                
            except Exception as e:
                self.logger.error(f"Processing source requirement {source.id}: Failed to process source: {str(e)}")
                raise

        # Process target requirements
        self.logger.info(f"Processing {len(target_requirements)} target requirements")
        for target in target_requirements:
            try:
                self.logger.info(f"Processing target requirement {target.id}: Creating embedding...")
                target_embedding = self.embedding_client.embed(target.content)
                
                # Convert embedding to binary format
                embedding_bytes = self._serialize_vector(target_embedding)
                
                target_key = f"tgt_req:{target.id}"
                target_record = {
                    "id": target_key,
                    "text": target.content,
                    "embedding": embedding_bytes,  # Store as binary
                    "metadata": json.dumps({
                        "type": target.type,
                        "id": target.id,
                        "level": target.level
                    })
                }
                target_records.append(target_record)
                self.logger.debug(f"Processing target requirement {target.id}: Processed target requirement")
                
            except Exception as e:
                self.logger.error(f"Processing target requirement {target.id}: Failed to process target: {str(e)}")
                raise

        self.logger.info("Completed requirements preprocessing")
        return source_records, target_records

    @handle_exception("Error processing requirements batch")
    async def process_requirements_batch(self,
                                      source_requirements: List[Requirement],
                                      target_requirements: List[Requirement],
                                      batch_size: int = 5,
                                      max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """
        Process a batch of source requirements against target requirements.
        
        Args:
            source_requirements: List of source requirements to process
            target_requirements: List of target requirements to match against
            batch_size: Size of target requirement batches for processing
            max_concurrent: Maximum number of concurrent source requirements to process
            
        Returns:
            List of match results with scores and metadata
        """
        try:
            # Get environment variables for rate limiting controls
            batch_size = int(os.getenv("BATCH_SIZE", str(batch_size)))
            max_concurrent = int(os.getenv("MAX_CONCURRENT_JOBS", str(max_concurrent)))
            batch_pause_seconds = float(os.getenv("BATCH_PAUSE_SECONDS", "2.0"))
            dynamic_throttling = os.getenv("ENABLE_DYNAMIC_THROTTLING", "True").lower() == "true"
            error_threshold = int(os.getenv("ERROR_THRESHOLD", "3"))
            
            # Tracking variables for dynamic throttling
            recent_errors = 0
            current_concurrency = max_concurrent
            
            # Track processing statistics
            total_sources = len(source_requirements)
            total_targets = len(target_requirements)
            total_pairs = total_sources * total_targets
            
            self.logger.info(f"Starting batch processing of {total_sources} source requirements against {total_targets} target requirements")
            self.logger.info(f"Total possible requirement pairs to evaluate: {total_pairs}")
            self.logger.info(f"Processing configuration: batch_size={batch_size}, max_concurrent={max_concurrent}, "
                          f"batch_pause={batch_pause_seconds}s, dynamic_throttling={dynamic_throttling}")
            
            # Clear and prepare indices
            await self.clear_indices()
            source_records, target_records = await self.preprocess_requirements(
                source_requirements, target_requirements
            )
            
            # Load records into indices
            if source_records:
                await self.src_index.load(source_records, keys=[r["id"] for r in source_records])
            if target_records:
                await self.tgt_index.load(target_records, keys=[r["id"] for r in target_records])
            
            all_matches = []
            semaphore = asyncio.Semaphore(current_concurrency)  # Start with configured concurrency
            processed_count = 0
            
            async def process_source_requirement(source_req: Requirement, src_idx: int):
                nonlocal processed_count, recent_errors, current_concurrency
                try:
                    async with semaphore:
                        self.logger.info(f"Starting source requirement {src_idx}/{total_sources}: {source_req.id}")
                        
                        # Process target requirements in batches
                        matches = []
                        target_batches = [
                            target_requirements[i:i + batch_size]
                            for i in range(0, len(target_requirements), batch_size)
                        ]
                        
                        source_record = {
                            "id": source_req.id,
                            "content": source_req.content,
                            "type": source_req.type,
                            "level": source_req.level
                        }
                        
                        total_batches = len(target_batches)
                        self.logger.debug(f"Processing {total_batches} batches for source {src_idx}/{total_sources}: {source_req.id}")
                        
                        for batch_idx, target_batch in enumerate(target_batches):
                            batch_num = batch_idx + 1
                            try:
                                # Add a pause between batches to avoid overwhelming API rate limits
                                if batch_idx > 0 and batch_pause_seconds > 0:
                                    self.logger.debug(f"Pausing {batch_pause_seconds}s between batches")
                                    await asyncio.sleep(batch_pause_seconds)
                                
                                self.logger.debug(f"Processing target batch {batch_num}/{total_batches} "
                                              f"for source {src_idx}/{total_sources} ({source_req.id})")
                                
                                # Prepare target records for this batch
                                target_records = [{
                                    "id": target.id,
                                    "content": target.content,
                                    "type": target.type,
                                    "level": target.level
                                } for target in target_batch]
                                
                                # Get initial matches from actor
                                batch_matches = await self.evaluate_match_with_actor(
                                    source_record,
                                    target_records
                                )
                                
                                if batch_matches:
                                    # Process each match with judge and meta-judge
                                    processed_matches = await asyncio.gather(*[
                                        self.process_matches([match]) for match in batch_matches
                                    ])
                                    current_matches = [m for sublist in processed_matches for m in sublist]
                                    matches.extend(current_matches)
                                    
                                    # Update processed count
                                    processed_count += len(current_matches)
                                    self.logger.debug(f"Processed {processed_count}/{total_pairs} possible pairs so far "
                                                  f"({(processed_count/total_pairs)*100:.1f}% complete)")
                                    # Keep a less frequent INFO level log for overall progress tracking
                                    if processed_count % (max(1, total_pairs // 10)) == 0:  # Log roughly every 10% of progress
                                        self.logger.info(f"Progress: {processed_count}/{total_pairs} pairs processed "
                                                      f"({(processed_count/total_pairs)*100:.1f}%)")
                                
                                # Reset error counter on successful batch processing
                                if dynamic_throttling and recent_errors > 0:
                                    recent_errors = max(0, recent_errors - 1)  # Gradually reduce error count
                                    
                                    # If error situation is improving, gradually increase concurrency back
                                    if recent_errors == 0 and current_concurrency < max_concurrent:
                                        current_concurrency = min(current_concurrency + 1, max_concurrent)
                                        self.logger.info(f"Increasing concurrency to {current_concurrency} as errors have subsided")
                                        # Update semaphore for next tasks
                                        # Note: This won't affect currently executing tasks
                                
                            except Exception as batch_error:
                                # Count rate limit errors for dynamic throttling
                                if dynamic_throttling and ("rate limit" in str(batch_error).lower() or 
                                                          "429" in str(batch_error)):
                                    recent_errors += 1
                                    self.logger.warning(f"Rate limit error detected. Error count: {recent_errors}/{error_threshold}")
                                    
                                    # If we exceed threshold, reduce concurrency
                                    if recent_errors >= error_threshold and current_concurrency > 1:
                                        # Reduce concurrency by 50% (minimum 1)
                                        new_concurrency = max(1, current_concurrency // 2)
                                        self.logger.warning(f"Too many rate limit errors. Reducing concurrency from {current_concurrency} to {new_concurrency}")
                                        current_concurrency = new_concurrency
                                        # We can't directly modify the semaphore, but future tasks will use the new concurrency
                                        
                                        # Add a longer pause to allow quota to refresh
                                        pause_time = 30.0  # 30 second pause on severe rate limiting
                                        self.logger.warning(f"Adding a {pause_time}s pause to allow API quota to refresh")
                                        await asyncio.sleep(pause_time)
                                
                                self.logger.error(f"Error processing batch {batch_num}/{total_batches} for source {source_req.id}: {str(batch_error)}")
                                # Continue with next batch rather than failing the entire source requirement
                        
                        self.logger.info(f"Processed {len(matches)} matches for source {source_req.id}")
                        return matches
                        
                except Exception as e:
                    self.logger.error(f"Error processing source requirement {source_req.id}: {str(e)}")
                    return []

            # Process source requirements in smaller batches
            source_batch_size = max(1, current_concurrency)  # Process in batches matching concurrency
            for i in range(0, len(source_requirements), source_batch_size):
                # Create a batch of source requirements
                source_batch = source_requirements[i:i + source_batch_size]
                batch_start = i + 1
                batch_end = min(i + len(source_batch), total_sources)
                
                self.logger.info(f"Processing source requirements batch {batch_start}-{batch_end}/{total_sources}")
                
                # Update semaphore based on current concurrency (which might have been adjusted)
                semaphore = asyncio.Semaphore(current_concurrency)
                
                # Create tasks for this batch
                tasks = [
                    process_source_requirement(src, idx) 
                    for idx, src in enumerate(source_batch, start=batch_start)
                ]
                
                # Wait for all tasks in this batch to complete
                batch_results = await asyncio.gather(*tasks)
                
                # Add results to overall matches
                for result in batch_results:
                    if result:
                        all_matches.extend(result)
                
                # Add a pause between batches if we're not at the end
                if batch_end < total_sources and batch_pause_seconds > 0:
                    self.logger.info(f"Completed source batch {batch_start}-{batch_end}. Pausing {batch_pause_seconds*2}s before next batch")
                    await asyncio.sleep(batch_pause_seconds * 2)  # Double pause between source batches
            
            if not all_matches:
                self.logger.warning("No matches found in any batch")
                return []
                
            self.logger.info(f"Completed batch processing with {len(all_matches)} total matches")
            self.logger.info(f"Total matches ({len(all_matches)}) represents {(len(all_matches)/total_pairs)*100:.1f}% of all possible requirement pairs")
            return all_matches
            
        except Exception as e:
            self.logger.error(f"Error in process_requirements_batch: {str(e)}")
            raise

    async def validate_and_fix_json_response(self, response: str, role: str) -> Union[Dict, List]:
        """
        Validate and fix JSON responses from LLMs.
        
        Args:
            response: Raw response string from LLM
            role: Role that generated the response (actor/judge/meta-judge)
        
        Returns:
            Dict or List: Validated and potentially fixed JSON structure
        """
        try:
            # Clean up the response to extract just the JSON part
            json_str = self.extract_json_from_response(response)
            
            # Parse the JSON
            result = json.loads(json_str)
            
            # Handle role-specific fixes
            if role == 'judge':
                # If response is missing judgment wrapper, add it
                if 'judgment' not in result and 'requirement_info' in result:
                    self.logger.warning("Adding missing judgment wrapper to judge response")
                    result = {"judgment": result}
                
            elif role == 'meta-judge':
                # If response is missing meta_judgment wrapper, add it
                if 'meta_judgment' not in result and 'chosen_judgment_index' in result:
                    self.logger.warning("Adding missing meta_judgment wrapper to meta-judge response")
                    result = {"meta_judgment": result}
                    
            elif role == 'actor':
                # If response is a list of matches without wrapper, add it
                if isinstance(result, list):
                    self.logger.warning("Adding missing matches wrapper to actor response")
                    result = {"matches": result}
                
            # Format JSON with indentation for better readability
            formatted_json = json.dumps(result, indent=2)
            self.logger.debug(f"Validated {role} response structure:\n{formatted_json}")
            return result
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Invalid JSON from {role}, attempting correction")
            # Additional JSON fixing logic if needed...
            raise
            
        except Exception as e:
            self.logger.error(f"JSON validation failed: {str(e)}")
            raise

    async def evaluate_match_with_actor(self, source_req: Dict, target_reqs: List[Dict]) -> List[Dict]:
        """
        Evaluate requirement matches using the actor role.
        
        Args:
            source_req: Source requirement data
            target_reqs: List of target requirements to evaluate
            
        Returns:
            List[Dict]: List of matches for all pairs, with actor's assessment including:
                - source_id: Source requirement ID
                - target_id: Target requirement ID
                - score: Actor's score (0-5)
                - reasoning: Actor's detailed reasoning
                - match_type: Type of match (functional/non-functional/none)
        """
        current_role = "ACTOR"
        try:
            # Get actor prompt from prompt manager
            actor_prompt = self.prompt_manager.actor_prompt
            if not actor_prompt:
                error_msg = "Actor prompt is None or empty"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            self.logger.debug(f"Using actor prompt of length: {len(actor_prompt)}")
            self.logger.debug(f"Actor prompt preview: {actor_prompt[:200]}...")
            
            # Prepare input data for actor
            input_data = {
                "source_requirement": {
                    "id": source_req["id"],
                    "content": source_req["content"],
                    "type": source_req["type"],
                    "level": source_req["level"]
                },
                "target_requirements": [{
                    "id": target["id"],
                    "content": target["content"],
                    "type": target["type"],
                    "level": target["level"]
                } for target in target_reqs]
            }
            
            # Log information about current evaluation at INFO level
            self.logger.debug(f"[{current_role}] Evaluating source requirement {source_req['id']} against {len(target_reqs)} target requirements")
            target_ids = [target["id"] for target in target_reqs]
            self.logger.debug(f"[{current_role}] Target requirements: {', '.join(target_ids)}")
            
            # Format input data for debugging
            self.logger.debug(f"Actor input data:\n{json.dumps(input_data, indent=2)}")
            
            # Generate response with system prompt
            response = await self.llm_manager.agenerate(
                model_name=self.model_name,
                messages=[{
                    "role": "user",
                    "content": json.dumps(input_data)
                }],
                prompt=actor_prompt,
                current_role=current_role
            )
            
            self.logger.debug(f"[{current_role}] Received response for source {source_req['id']}")
            
            # Validate and fix JSON if needed
            try:
                result = await self.validate_and_fix_json_response(response, 'actor')
                
                # Handle the case where result is a list
                if isinstance(result, list):
                    batch_matches = result
                else:
                    # Try to get matches from dictionary response
                    batch_matches = result.get("matches", [])
                
                # Create a map of existing matches
                existing_matches = {
                    (match.get("source_id", source_req["id"]), match.get("target_id")): match 
                    for match in batch_matches if isinstance(match, dict)
                }
                
                # Ensure we have a match entry for every target requirement
                standardized_matches = []
                for target in target_reqs:
                    match_key = (source_req["id"], target["id"])
                    if match_key in existing_matches:
                        # Use existing match data
                        match = existing_matches[match_key]
                        standardized_match = {
                            "source_id": match.get("source_id", source_req["id"]),
                            "target_id": match.get("target_id"),
                            "score": int(match.get("score", 0)),  # Ensure score is stored consistently
                            "reasoning": match.get("reasoning", "No reasoning provided by actor"),  # Store actor's reasoning
                            "match_type": match.get("match_type", "functional")
                        }
                    else:
                        # Create default match entry for unmatched pair
                        self.logger.debug(f"Creating default match entry for {source_req['id']} -> {target['id']}")
                        standardized_match = {
                            "source_id": source_req["id"],
                            "target_id": target["id"],
                            "score": 0,
                            "reasoning": "No match identified by actor",
                            "match_type": "none"
                        }
                    standardized_matches.append(standardized_match)
                
                self.logger.info(f"Processed {len(standardized_matches)} matches for source {source_req['id']}")
                return standardized_matches
                
            except Exception as e:
                self.logger.error(f"Error processing actor response: {str(e)}")
                # Create default matches for all targets on error
                return [{
                    "source_id": source_req["id"],
                    "target_id": target["id"],
                    "score": 0,
                    "reasoning": f"Error processing actor response: {str(e)}",
                    "match_type": "error"
                } for target in target_reqs]
            
        except Exception as e:
            self.logger.error(f"Actor evaluation failed: {str(e)}")
            # Create default matches for all targets on error
            return [{
                "source_id": source_req["id"],
                "target_id": target["id"],
                "score": 0,
                "reasoning": f"Actor evaluation failed: {str(e)}",
                "match_type": "error"
            } for target in target_reqs]

    async def evaluate_match_with_judge(self, match_data: Dict) -> Dict:
        """
        Evaluate a requirement match using the judge role.
        
        Expected output format from prompt:
        {
            "judgment": {
                "requirement_info": {
                    "source_id": str,
                    "target_id": str
                },
                "score": int,  # 0-5
                "reasoning": str,
                "match_quality": {
                    "semantic_alignment": int,  # 0-5
                    "functional_completeness": int,  # 0-5
                    "non_functional_coverage": int  # 0-5
                }
            }
        }
        """
        current_role = "JUDGE"
        try:
            # Get requirement content from Redis
            source_key = f"src_req:{match_data['source_id']}"
            target_key = f"tgt_req:{match_data['target_id']}"
            
            # Log information about current evaluation at INFO level
            self.logger.debug(f"[{current_role}] Evaluating match between source {match_data['source_id']} and target {match_data['target_id']}")
            
            self.logger.debug(f"Retrieving source requirement {source_key} from Redis")
            source_content = await self.redis_client.client.hget(source_key, "text")
            if not source_content:
                raise ValueError(f"Source requirement {match_data['source_id']} not found in Redis")
            
            self.logger.debug(f"Retrieving target requirement {target_key} from Redis")
            target_content = await self.redis_client.client.hget(target_key, "text")
            if not target_content:
                raise ValueError(f"Target requirement {match_data['target_id']} not found in Redis")
            
            # Prepare complete match data with content
            complete_match_data = {
                "match_data": {
                    "source_id": match_data['source_id'],
                    "source_content": source_content.decode('utf-8'),
                    "target_id": match_data['target_id'],
                    "target_content": target_content.decode('utf-8'),
                    "initial_score": match_data.get('score', 0),
                    "reasoning": match_data.get('reasoning', ''),
                    "match_type": match_data.get('match_type', 'functional')
                }
            }
            
            # Format complete match data for debugging
            self.logger.debug(f"Judge input data:\n{json.dumps(complete_match_data, indent=2)}")
            
            # Get judge response
            response = await self.llm_manager.agenerate(
                model_name=self.model_name,
                messages=[{
                    "role": "user",
                    "content": json.dumps(complete_match_data)
                }],
                prompt=self.prompt_manager.judge_prompt,
                current_role=current_role
            )
            
            self.logger.debug(f"[{current_role}] Received response for match {match_data['source_id']} -> {match_data['target_id']}")
            
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

    async def evaluate_with_meta_judge(self, match: Dict, judgments: List[Dict]) -> Dict:
        """
        Use Meta-Judge to evaluate multiple judgments.
        
        Expected output format from prompt:
        {
            "meta_judgment": {
                "source_id": str,
                "target_id": str,
                "chosen_judgment_index": int,
                "final_score": int,  # 0-5
                "is_traceable": bool,
                "reasoning": str
            }
        }
        """
        current_role = "META-JUDGE"
        try:
            # Log information about current evaluation at INFO level
            self.logger.debug(f"[{current_role}] Making final judgment on match between source {match['source_id']} and target {match['target_id']}")
            
            # Format input data for debugging
            input_data = {
                "match": match,
                "judgments": judgments
            }
            self.logger.debug(f"Meta-judge input data:\n{json.dumps(input_data, indent=2)}")
            
            # Format input exactly as specified in prompt manager
            meta_input = {
                "match": {
                    "source_id": match["source_id"],
                    "target_id": match["target_id"],
                    "score": match.get("score", 0),
                    "reasoning": match.get("reasoning", ""),
                    "match_quality": judgments[0].get("match_quality", {
                        "semantic_alignment": 0,
                        "functional_completeness": 0,
                        "non_functional_coverage": 0
                    }) if judgments else {}
                },
                "judgments": judgments
            }
            
            # Get meta-judge response
            response = await self.llm_manager.agenerate(
                model_name=self.model_name,
                messages=[{
                    "role": "user",
                    "content": json.dumps(meta_input)
                }],
                prompt=self.prompt_manager.meta_judge_prompt,
                current_role=current_role
            )
            
            self.logger.debug(f"[{current_role}] Received response for match {match['source_id']} -> {match['target_id']}")
            
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

    async def process_matches(self, matches: List[Dict]) -> List[Dict]:
        """
        Process matches through the complete evaluation pipeline.
        
        Returns:
            List[Dict]: List of evaluated matches, each containing:
                - source_id: Source requirement ID
                - target_id: Target requirement ID
                - actor_score: Actor's score
                - actor_reasoning: Actor's reasoning
                - judge_score: Judge's evaluation score
                - judge_reasoning: Judge's reasoning
                - final_score: Meta-judge's final score
                - is_traceable: Whether the match is traceable
                - meta_judge_reasoning: Meta-judge's reasoning
                - match_quality: Detailed quality metrics
        """
        final_matches = []
        total_matches = len(matches)
        
        for idx, match in enumerate(matches, 1):
            try:
                self.logger.debug(f"Processing match {idx}/{total_matches}: {match['source_id']} -> {match['target_id']}")
                
                # Get judge evaluation first
                judgment = await self.evaluate_match_with_judge(match)
                
                # Get meta-judge validation
                meta_judgment = await self.evaluate_with_meta_judge(match, [judgment])
                
                # Include all matches in results, with their traceable status
                final_match = {
                    "source_id": match["source_id"],
                    "target_id": match["target_id"],
                    "actor_score": match.get("score", 0),  # Store actor's score
                    "actor_reasoning": match.get("reasoning", "No actor reasoning provided"),  # Store actor's reasoning
                    "judge_score": judgment.get("score", 0),
                    "judge_reasoning": judgment.get("reasoning", "No judge reasoning provided"),  # Store judge's reasoning
                    "final_score": meta_judgment.get("final_score", 0),
                    "is_traceable": meta_judgment.get("is_traceable", False),
                    "meta_judge_reasoning": meta_judgment.get("reasoning", "No meta-judge reasoning provided"),  # Store meta-judge's reasoning
                    "match_quality": judgment.get("match_quality", {
                        "semantic_alignment": 0,
                        "functional_completeness": 0,
                        "non_functional_coverage": 0
                    })
                }
                
                self.logger.debug(f"Completed match {idx}/{total_matches}: {match['source_id']} -> {match['target_id']}"
                              f" with final score {final_match.get('final_score', 0)}")
                
                final_matches.append(final_match)
                
            except Exception as e:
                self.logger.error(f"Failed to process match {idx}/{total_matches}: {str(e)}")
                self.logger.debug("Error details:", exc_info=True)
                continue
                
        return final_matches

    def validate_actor_input(self, input_data: Dict) -> bool:
        """Validate actor input format matches prompt specification."""
        try:
            required_source_fields = {"id", "content"}
            required_target_fields = {"id", "content"}
            
            source = input_data.get("source_requirement", {})
            targets = input_data.get("target_requirements", [])
            
            if not all(field in source for field in required_source_fields):
                self.logger.error(f"Missing required source fields. Expected: {required_source_fields}")
                return False
            
            for target in targets:
                if not all(field in target for field in required_target_fields):
                    self.logger.error(f"Missing required target fields. Expected: {required_target_fields}")
                    return False
                
            return True
        except Exception as e:
            self.logger.error(f"Actor input validation failed: {str(e)}")
            return False

    def validate_judge_input(self, input_data: Dict) -> bool:
        """Validate judge input format matches prompt specification."""
        try:
            required_fields = {
                "source_id", "source_content", "target_id", 
                "target_content", "initial_score"
            }
            
            match_data = input_data.get("match_data", {})
            
            if not all(field in match_data for field in required_fields):
                self.logger.error(f"Missing required judge fields. Expected: {required_fields}")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Judge input validation failed: {str(e)}")
            return False

    def validate_meta_judge_input(self, input_data: Dict) -> bool:
        """Validate meta-judge input format matches prompt specification."""
        try:
            required_match_fields = {"source_id", "source_content", "target_id", "target_content"}
            required_judgment_fields = {"score", "reasoning"}
            required_meta_judgment_fields = {"chosen_judgment_index", "final_score", "is_traceable", "reasoning"}
            
            match = input_data.get("match", {})
            judgments = input_data.get("judgments", [])
            meta_judgment = input_data.get("meta_judgment", {})
            
            if not all(field in match for field in required_match_fields):
                self.logger.error(f"Missing required match fields. Expected: {required_match_fields}")
                return False
            
            for judgment in judgments:
                if not all(field in judgment for field in required_judgment_fields):
                    self.logger.error(f"Missing required judgment fields. Expected: {required_judgment_fields}")
                    return False
                
            if not all(field in meta_judgment for field in required_meta_judgment_fields):
                self.logger.error(f"Missing required meta-judgment fields. Expected: {required_meta_judgment_fields}")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Meta-judge input validation failed: {str(e)}")
            return False

    def extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON content from LLM response.
        
        Args:
            response: Raw response string from LLM
            
        Returns:
            str: Cleaned JSON string ready for parsing
        """
        try:
            # If response is already valid JSON, return it
            json.loads(response)
            return response
        except json.JSONDecodeError:
            self.logger.debug("Response is not valid JSON, attempting to extract...")
            
            # Try to extract JSON between curly braces
            try:
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    # Verify it's valid JSON
                    json.loads(json_str)
                    return json_str
            except:
                pass
            
            # Try to extract JSON between square brackets
            try:
                start = response.find('[')
                end = response.rfind(']') + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    # Verify it's valid JSON
                    json.loads(json_str)
                    return json_str
            except:
                pass
            
            # Try to extract between response tags if present
            if "<response>" in response and "</response>" in response:
                try:
                    json_str = response.split("<response>")[1].split("</response>")[0].strip()
                    # Verify it's valid JSON
                    json.loads(json_str)
                    return json_str
                except:
                    pass
                
            # If all extraction attempts fail, return original response
            self.logger.warning("Could not extract valid JSON, returning original response")
            return response