"""Redis Client Module for vector operations and requirement storage."""

from typing import Dict, List, Any
import json
import os
from redis.asyncio import Redis
from redisvl.index import AsyncSearchIndex
from .redis_schema import (
    get_source_requirements_schema,
    get_target_requirements_schema,
    get_matches_schema
)
from ..utils.logger import setup_logger, handle_exception
import time

class RedisClient:
    """Redis client for managing requirements and vector operations."""
    
    def __init__(self, redis_client: Redis):
        """Initialize Redis client with vector search capabilities."""
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self.client = redis_client
        self.src_index = None
        self.tgt_index = None
        self.matches_index = None
        
        # Ensure REDIS_URL env var is set for redisvl library
        if "REDIS_URL" not in os.environ and self.client:
            connection_info = self.client.connection_pool.connection_kwargs
            host = connection_info.get('host', os.getenv('REDIS_HOST', 'localhost'))
            port = connection_info.get('port', os.getenv('REDIS_PORT', 6379))
            password = connection_info.get('password', os.getenv('REDIS_PASSWORD', ''))
            
            if password:
                redis_url = f"redis://:{password}@{host}:{port}"
            else:
                redis_url = f"redis://{host}:{port}"
                
            os.environ["REDIS_URL"] = redis_url
            self.logger.info(f"Set REDIS_URL environment variable to {redis_url}")

    async def initialize_indices(self) -> None:
        """Initialize all required Redis indices."""
        try:
            # Ensure REDIS_URL is set before creating indices
            if "REDIS_URL" not in os.environ:
                # Get connection info from client
                connection_info = self.client.connection_pool.connection_kwargs
                host = connection_info.get('host', os.getenv('REDIS_HOST', 'localhost'))
                port = connection_info.get('port', os.getenv('REDIS_PORT', 6379))
                password = connection_info.get('password', os.getenv('REDIS_PASSWORD', ''))
                
                if password:
                    redis_url = f"redis://:{password}@{host}:{port}"
                else:
                    redis_url = f"redis://{host}:{port}"
                    
                os.environ["REDIS_URL"] = redis_url
                self.logger.info(f"Set REDIS_URL environment variable to {redis_url}")
                
            # Create source requirements index
            src_schema = get_source_requirements_schema()
            src_schema['client'] = self.client
            self.src_index = AsyncSearchIndex.from_dict(src_schema)
            await self.src_index.create(overwrite=True)
            self.logger.debug("Created source requirements index")

            # Create target requirements index
            tgt_schema = get_target_requirements_schema()
            tgt_schema['client'] = self.client
            self.tgt_index = AsyncSearchIndex.from_dict(tgt_schema)
            await self.tgt_index.create(overwrite=True)
            self.logger.debug("Created target requirements index")

            # Create matches index
            matches_schema = get_matches_schema()
            matches_schema['client'] = self.client
            self.matches_index = AsyncSearchIndex.from_dict(matches_schema)
            await self.matches_index.create(overwrite=True)
            self.logger.debug("Created matches index")

        except Exception as e:
            self.logger.error(f"Error initializing indices: {str(e)}")
            raise

    async def store_source_requirement(self, req_id: str, content: str, embedding: List[float], metadata: Dict = None) -> None:
        """Store a source requirement."""
        try:
            record = {
                "id": f"src_req:{req_id}",
                "text": content,
                "embedding": embedding,
                "metadata": metadata or {},
                "timestamp": time.time()
            }
            await self.src_index.load([record])
            self.logger.debug(f"Stored source requirement {req_id}")
        except Exception as e:
            self.logger.error(f"Error storing source requirement {req_id}: {str(e)}")
            raise

    async def store_target_requirement(self, req_id: str, content: str, embedding: List[float], 
                                     req_type: str, metadata: Dict = None) -> None:
        """Store a target requirement."""
        try:
            record = {
                "id": f"tgt_req:{req_id}",
                "text": content,
                "embedding": embedding,
                "type": req_type,
                "metadata": metadata or {},
                "timestamp": time.time()
            }
            await self.tgt_index.load([record])
            self.logger.debug(f"Stored target requirement {req_id}")
        except Exception as e:
            self.logger.error(f"Error storing target requirement {req_id}: {str(e)}")
            raise

    async def store_match(self, source_id: str, target_id: str, score: float, 
                         vector_similarity: float, reasoning: str) -> None:
        """Store a requirement match."""
        try:
            record = {
                "id": f"match:{source_id}:{target_id}",
                "source_id": source_id,
                "target_id": target_id,
                "score": score,
                "vector_similarity": vector_similarity,
                "reasoning": reasoning,
                "timestamp": time.time()
            }
            await self.matches_index.load([record])
            self.logger.debug(f"Stored match between {source_id} and {target_id}")
        except Exception as e:
            self.logger.error(f"Error storing match: {str(e)}")
            raise

    async def load(self, records: List[Dict[str, Any]]) -> None:
        """Load records into the appropriate index based on their ID prefix."""
        try:
            source_records = []
            target_records = []
            
            for record in records:
                # Convert record to Redis-compatible format
                processed_record = {
                    "id": record["id"],
                    "text": record["text"],
                    # Convert embedding to comma-separated string
                    "embedding": ",".join(map(str, record["embedding"])),
                    "metadata": record["metadata"]
                }
                
                self.logger.error(f"Processed record format: {processed_record.keys()}")
                
                if record['id'].startswith('src_req:'):
                    source_records.append(processed_record)
                elif record['id'].startswith('tgt_req:'):
                    target_records.append(processed_record)
                else:
                    self.logger.error(f"Unknown record type with ID: {record['id']}")
            
            # Load records into indices
            if source_records:
                self.logger.error(f"Loading {len(source_records)} source records")
                try:
                    await self.src_index.load(source_records)
                except Exception as e:
                    self.logger.error(f"Error loading source records: {str(e)}")
                    self.logger.error(f"First source record format: {source_records[0].keys()}")
                    raise
                    
            if target_records:
                self.logger.error(f"Loading {len(target_records)} target records")
                try:
                    await self.tgt_index.load(target_records)
                except Exception as e:
                    self.logger.error(f"Error loading target records: {str(e)}")
                    self.logger.error(f"First target record format: {target_records[0].keys()}")
                    raise
                    
        except Exception as e:
            self.logger.error(f"Error in load method: {str(e)}")
            self.logger.error(f"Record count: {len(records)}")
            if records:
                self.logger.error(f"Original record format: {records[0].keys()}")
            raise
 