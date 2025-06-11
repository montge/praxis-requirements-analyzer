"""Redis schema configuration for requirements vector search."""

from typing import Dict, Any
from redisvl.schema import IndexSchema

def get_source_requirements_schema() -> Dict[str, Any]:
    """Get the schema for source requirements vector search index."""
    return {
        "index": {
            "name": "src_reqs",
            "prefix": "src_req:",
            "storage_type": "hash"
        },
        "fields": [
            {
                "name": "text",
                "type": "text"
            },
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": 384,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                    "datatype": "float32"
                }
            },
            {
                "name": "metadata",
                "type": "text"
            }
        ]
    }

def get_target_requirements_schema() -> Dict[str, Any]:
    """Get the schema for target requirements vector search index."""
    return {
        "index": {
            "name": "tgt_reqs",
            "prefix": "tgt_req:",
            "storage_type": "hash"
        },
        "fields": [
            {
                "name": "text",
                "type": "text"
            },
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": 384,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                    "datatype": "float32"
                }
            },
            {
                "name": "metadata",
                "type": "text"
            },
            {
                "name": "cluster_id",
                "type": "tag"
            },
            {
                "name": "type",
                "type": "tag"
            }
        ]
    }

def get_matches_schema() -> Dict[str, Any]:
    """Get the schema for requirement matches."""
    return {
        "index": {
            "name": "matches",
            "prefix": "match:",
            "storage_type": "hash"
        },
        "fields": [
            {
                "name": "source_id",
                "type": "tag"
            },
            {
                "name": "target_id", 
                "type": "tag"
            },
            {
                "name": "score",
                "type": "numeric"
            },
            {
                "name": "vector_similarity",
                "type": "numeric"
            },
            {
                "name": "reasoning",
                "type": "text"
            },
            {
                "name": "timestamp",
                "type": "numeric"
            }
        ]
    } 