"""
Client implementation for HuggingFace models.
Provides integration with HuggingFace's model hub and inference API.
"""

import os
from typing import Dict, Any, Optional, List
from huggingface_hub import InferenceClient
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from ...base_llm import BaseLLM
from src.utils.logger import setup_logger, log_error
import asyncio
import json
import concurrent.futures

logger = setup_logger(__name__)

class HuggingFaceClient(BaseLLM):
    """
    Client implementation for HuggingFace models.
    
    Handles authentication and interaction with models hosted on HuggingFace's
    model hub through their inference API.
    
    Args:
        model_name (str): Name of the HuggingFace model to use
        max_tokens (int): Maximum number of tokens for generation
        
    Attributes:
        model_id (str): The actual model identifier on HuggingFace
        client (InferenceClient): The underlying HuggingFace client
        cache_enabled (bool): Whether to enable LangChain caching
        cache_hits (int): Counter for cache hits
        total_requests (int): Counter for total requests
    """
    
    def __init__(self, model_name: str, max_tokens: int = 2048):
        """
        Initialize the HuggingFace client.
        
        Args:
            model_name (str): Name of the HuggingFace model to use
            max_tokens (int): Maximum number of tokens for generation
        """
        super().__init__(model_name)
        self._max_tokens = max_tokens
        self.cache_hits = 0
        self.total_requests = 0
        self._executor = concurrent.futures.ThreadPoolExecutor()

    async def initialize(self) -> None:
        """
        Initialize the HuggingFace client with API credentials.
        
        Sets up the InferenceClient with model configuration.
        
        Raises:
            Exception: If initialization fails due to missing credentials or API errors
        """
        try:
            # Get model ID from environment variables
            self.model_id = os.getenv(f"{self.model_name}_MODEL_ID")
            if not self.model_id:
                raise ValueError(f"Model ID not found for {self.model_name} in environment variables")
            
            logger.info(f"Loading model ID for {self.model_name}: {self.model_id}")

            # Use the configured max_tokens value
            self.max_tokens = getattr(self, 'max_tokens', 2048)
            logger.info(f"Model {self.model_name} using {self.max_tokens} tokens")
            
            # Enable caching by default
            self.cache_enabled = os.getenv("ENABLE_HF_CACHE", "True").lower() == "true"
            
            # Set up LangChain caching if enabled
            if self.cache_enabled:
                set_llm_cache(InMemoryCache())
                logger.info(f"LangChain caching enabled for {self.model_name}")
            
            await self._setup_client()
            
        except Exception as e:
            log_error(logger, f"Failed to initialize HuggingFace client for {self.model_name}", e)
            raise

    async def _setup_client(self) -> None:
        """Setup the HuggingFace InferenceClient with current configuration"""
        # Initialize the HuggingFace InferenceClient
        self.client = InferenceClient(
            model=self.model_id,
            token=os.getenv("HF_TOKEN"),
            timeout=120
        )
        logger.info(f"Initialized {self.model_name} client with model {self.model_id}")

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        self._max_tokens = value
        if hasattr(self, 'client'):
            asyncio.create_task(self._setup_client())

    def _extract_response_text(self, response: Any) -> str:
        """
        Extract text from a chat completion response object.
        
        This method handles different response formats from the HuggingFace
        chat_completion API. It supports multiple response structures:
        
        1. OpenAI-compatible format: response.choices[0].message.content
        2. HuggingFace-specific format: response.generated_text
        3. Dictionary format with choices: response['choices'][0]['message']['content']
        4. Dictionary format with generated_text: response['generated_text']
        
        Args:
            response: The response object from chat_completion
            
        Returns:
            The extracted text as a string
            
        Note:
            Falls back to string representation if no known structure is detected.
            This handles API response structure changes gracefully.
        """
        # Try various ways to get the response text based on possible structures
        try:
            # First check if it's a dictionary with choices
            if hasattr(response, 'choices') and response.choices:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    return response.choices[0].message.content
                    
            # Check for OpenAI-compatible format
            if hasattr(response, 'choices') and response.choices:
                if 'message' in response.choices[0] and 'content' in response.choices[0]['message']:
                    return response.choices[0]['message']['content']
                    
            # Check for HuggingFace-specific format
            if hasattr(response, 'generated_text'):
                return response.generated_text
                
            # Try accessing as dictionary
            if isinstance(response, dict):
                if 'choices' in response and response['choices']:
                    if 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                        return response['choices'][0]['message']['content']
                if 'generated_text' in response:
                    return response['generated_text']
            
            # Fallback: convert the entire response to a string
            logger.warning(f"Could not extract text from response structure, using string representation: {type(response)}")
            return str(response)
            
        except Exception as e:
            logger.warning(f"Error extracting response text: {str(e)}, falling back to string representation")
            return str(response)

    async def generate_response(self,
                              messages: List[Dict[str, str]],
                              max_tokens: int = 2048,
                              temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a response using the HuggingFace model through chat_completion API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for response generation
            
        Returns:
            Dictionary containing the generated response
            
        Raises:
            Exception: If generation fails
        """
        try:
            logger.info(f"HuggingFace generating response with {len(messages)} messages")
            
            # Increment total requests counter
            self.total_requests += 1
            
            # Format messages for the chat_completion API
            formatted_messages = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                # Convert role to match HuggingFace's expectations
                if role == "system":
                    role = "system"
                    logger.info("Adding system message")
                    logger.debug(f"System message content: {content[:100]}...")
                elif role == "user":
                    role = "user"
                    logger.info("Adding user message")
                    logger.debug(f"User message content: {content[:100]}...")
                elif role == "assistant":
                    role = "assistant"
                    logger.info("Adding assistant message")
                    logger.debug(f"Assistant message content: {content[:100]}...")
                else:
                    logger.warning(f"Unknown role: {role}, treating as user")
                    role = "user"
                
                formatted_messages.append({"role": role, "content": content})
            
            logger.info(f"Combined prompt length: {sum(len(msg['content']) for msg in formatted_messages)} chars")
            
            # Log the use of caching
            if self.cache_enabled:
                logger.debug("LangChain caching enabled for this request")
            
            logger.info("Sending prompt to HuggingFace")
            
            # Use the chat_completion API instead of the deprecated post method
            # Run in a thread to avoid blocking the event loop with synchronous HF API call
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._executor,
                lambda: self.client.chat_completion(
                    messages=formatted_messages,
                    temperature=temperature,
                    top_p=0.95,
                    max_tokens=max_tokens,
                    model=self.model_id
                )
            )
            
            # Log cache statistics if we have some mechanism to detect cache hits
            # This may need to be implemented differently with the new API
            if self.cache_enabled and self.total_requests > 0:
                cache_hit_rate = (self.cache_hits / self.total_requests) * 100
                logger.debug(f"Cache statistics: {self.cache_hits}/{self.total_requests} hits ({cache_hit_rate:.1f}%)")
            
            logger.info("Received response from HuggingFace")
            
            # Extract the text content from the response using our helper method
            response_text = self._extract_response_text(response)
            logger.debug(f"Raw response from HuggingFace:\n{response_text}")
            
            return {
                "model": self.model_id,
                "content": response_text,
                "role": "assistant"
            }
            
        except Exception as e:
            log_error(logger, f"Failed to generate response from HuggingFace model {self.model_id}", e)
            raise 