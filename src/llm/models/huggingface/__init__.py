"""
HuggingFace Models Package

This package provides clients for interacting with HuggingFace's models and APIs.

Key Components:
- HuggingFaceClient: Client for text generation using HF Inference API
- HuggingFaceVLLMClient: Client for local text generation using vLLM
- HuggingFaceEmbeddingsClient: Client for generating embeddings using sentence-transformers

Features:
- Text generation with multiple model support
- Local inference with vLLM for high performance
- Embedding generation optimized for requirements analysis
- Automatic retries and error handling
- Configurable model parameters
- Detailed logging and debugging

Usage:
    from src.llm.models.huggingface import HuggingFaceClient, HuggingFaceVLLMClient, HuggingFaceEmbeddingsClient
    
    # For text generation using HF API
    hf_client = HuggingFaceClient(model_id="HuggingFaceH4/zephyr-7b-beta")
    
    # For local text generation using vLLM
    vllm_client = HuggingFaceVLLMClient(model_id="HuggingFaceH4/zephyr-7b-beta")
    
    # For embeddings
    embeddings_client = HuggingFaceEmbeddingsClient(
        model_id="sentence-transformers/all-MiniLM-L6-v2"
    )
"""

from .hf_client import HuggingFaceClient
from .hf_vllm_client import HuggingFaceVLLMClient
from .hf_embeddings_client import HuggingFaceEmbeddingsClient

__all__ = [
    'HuggingFaceClient',
    'HuggingFaceVLLMClient',
    'HuggingFaceEmbeddingsClient'
] 