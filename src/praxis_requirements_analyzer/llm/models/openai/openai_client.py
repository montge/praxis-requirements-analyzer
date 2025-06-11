"""
Client implementation for OpenAI models.
Provides integration with OpenAI's GPT models through their API.

This module handles authentication, message formatting, and response processing
for OpenAI's large language models including GPT-3.5 and GPT-4 series.
"""

import os
from typing import Dict, Any, Optional, List
from pydantic import SecretStr  # Import SecretStr for proper type handling
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from ...base_llm import BaseLLM
from ....utils.logger import setup_logger, log_error

logger = setup_logger(__name__)

class OpenAIClient(BaseLLM):
    """
    Client implementation for OpenAI models.
    
    Handles authentication and interaction with OpenAI's GPT models through
    the OpenAI API. Supports caching to reduce token usage and costs.
    
    Args:
        model_name (str): Name of the OpenAI model to use (e.g., 'GPT4')
        
    Attributes:
        model_id (str): The actual model identifier used by OpenAI (e.g., 'gpt-4-turbo')
        client (ChatOpenAI): The underlying OpenAI chat client
        cache_enabled (bool): Whether to enable LangChain caching
        cache_hits (int): Counter for cache hits
        total_requests (int): Counter for total requests
        
    Supported Models:
        - GPT-3.5 series (GPT35, GPT3): Lower cost, faster response times
        - GPT-4 series (GPT4, GPT4O): Higher capabilities, better reasoning
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the OpenAI client.
        
        Args:
            model_name (str): Name of the OpenAI model to use (e.g., 'GPT4')
        """
        super().__init__(model_name)
        self.cache_hits = 0
        self.total_requests = 0
        self.cache_enabled = False
    
    async def initialize(self) -> None:
        """
        Initialize the OpenAI client with API credentials.
        
        Sets up the ChatOpenAI client with the appropriate model ID and API key.
        Model ID is fetched from environment variables based on model_name.
        For example, if model_name is 'GPT4', it will look for 'GPT4_MODEL_ID'.
        
        Environment Variables:
            - {model_name}_MODEL_ID: The OpenAI model identifier
            - OPENAI_API_KEY: API key for authentication
            - ENABLE_OPENAI_CACHE: Whether to enable response caching (default: "True")
        
        Raises:
            ValueError: If required environment variables are missing
            Exception: If initialization fails due to API errors
        """
        try:
            self.model_id = os.getenv(f"{self.model_name}_MODEL_ID")
            if not self.model_id:
                raise ValueError(f"Model ID not found for {self.model_name} in environment variables")
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            
            # Enable caching by default
            self.cache_enabled = os.getenv("ENABLE_OPENAI_CACHE", "True").lower() == "true"
            
            # Initialize cache statistics
            self.cache_hits = 0
            self.total_requests = 0
            
            # Set up LangChain caching if enabled
            if self.cache_enabled:
                set_llm_cache(InMemoryCache())
                logger.info(f"LangChain caching enabled for {self.model_name}")
            
            # Create the ChatOpenAI client with proper SecretStr type for API key
            # Modern versions of langchain_openai expect api_key as SecretStr
            api_key_secret = SecretStr(api_key)
            self.client = ChatOpenAI(
                model=self.model_id,
                api_key=api_key_secret,
                cache=self.cache_enabled
            )
            logger.info(f"Initialized {self.model_name} client with model {self.model_id}")
        except Exception as e:
            log_error(logger, f"Failed to initialize OpenAI client for {self.model_name}", e)
            raise
            
    async def generate_response(self,
                              messages: List[Dict[str, str]],
                              max_tokens: int = 2048,
                              temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a response using the OpenAI model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
                     Roles can be 'system', 'user', or 'assistant'
            max_tokens: Maximum tokens in response (default: 2048)
                       Higher values allow longer responses but cost more
            temperature: Sampling temperature (default: 0.7)
                        0.0 = deterministic, 1.0 = maximum creativity
            
        Returns:
            Dict containing:
                - model: The model ID used
                - content: The generated response text
                - role: Always "assistant"
            
        Raises:
            ValueError: For invalid input parameters
            RateLimitError: When API rate limits are exceeded
            AuthenticationError: For API key issues
            BadRequestError: For invalid requests
            ServerError: For OpenAI service issues
            Exception: For other unexpected errors
        """
        try:
            logger.info(f"OpenAI generating response with {len(messages)} messages")
            
            # Increment total requests counter
            self.total_requests += 1
            
            # Convert dict messages to LangChain message objects
            langchain_messages = []
            
            # Process messages in order: system -> user -> assistant
            system_messages = [msg for msg in messages if msg["role"] == "system"]
            user_messages = [msg for msg in messages if msg["role"] == "user"]
            assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
            
            logger.info(f"Message breakdown - System: {len(system_messages)}, "
                       f"User: {len(user_messages)}, Assistant: {len(assistant_messages)}")
            
            for msg in system_messages:
                logger.info("Adding system message")
                logger.debug(f"System message content: {msg['content'][:100]}...")
                langchain_messages.append(SystemMessage(content=msg["content"]))
                
            for msg in user_messages:
                logger.info("Adding user message")
                logger.debug(f"User message content: {msg['content'][:100]}...")
                langchain_messages.append(HumanMessage(content=msg["content"]))
                
            for msg in assistant_messages:
                logger.info("Adding assistant message")
                logger.debug(f"Assistant message content: {msg['content'][:100]}...")
                langchain_messages.append(AIMessage(content=msg["content"]))
            
            logger.info(f"Sending {len(langchain_messages)} messages to OpenAI")
            
            # Update client settings
            self.client.max_tokens = max_tokens
            self.client.temperature = temperature
            
            # Check if we're using caching
            if self.cache_enabled:
                logger.debug("LangChain caching enabled for this request")
            
            # Generate response
            response = await self.client.ainvoke(langchain_messages)
            
            # Determine if this was a cache hit (this is an approximation)
            # In a real implementation, you might want to use LangChain's callback handlers
            # to get precise information about cache hits
            is_cache_hit = False
            if hasattr(response, 'generation_info'):
                is_cache_hit = getattr(response, 'generation_info', {}).get('cached', False)
            
            if is_cache_hit:
                self.cache_hits += 1
                logger.debug(f"Cache hit detected (hit #{self.cache_hits})")
            
            # Log cache statistics
            if self.cache_enabled and self.total_requests > 0:
                cache_hit_rate = (self.cache_hits / self.total_requests) * 100
                logger.debug(f"Cache statistics: {self.cache_hits}/{self.total_requests} hits ({cache_hit_rate:.1f}%)")
            
            logger.info("Received response from OpenAI")
            logger.debug(f"Raw response from OpenAI:\n{response.content}")
            
            return {
                "model": self.model_id,
                "content": response.content,
                "role": "assistant"
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # Check for specific OpenAI error types
            if "Rate limit" in error_msg or "429" in error_msg:
                logger.error(f"OpenAI Rate Limit exceeded: {error_msg}")
                logger.warning("Consider implementing exponential backoff or reducing request frequency")
            elif "Authentication" in error_msg or "401" in error_msg:
                logger.error(f"OpenAI Authentication error: {error_msg}")
                logger.warning("Check your OPENAI_API_KEY environment variable or account status")
            elif "Invalid request" in error_msg or "400" in error_msg:
                logger.error(f"OpenAI Invalid Request error: {error_msg}")
                logger.warning("Review your request parameters for missing or invalid values")
            elif "Server error" in error_msg or "500" in error_msg:
                logger.error(f"OpenAI Server error: {error_msg}")
                logger.warning("This is likely a temporary issue with OpenAI's services")
            else:
                # Log other errors
                log_error(logger, f"Failed to generate response from OpenAI model {self.model_id}", e)
            
            # Re-raise the exception after logging
            raise 