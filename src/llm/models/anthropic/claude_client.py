"""
Client implementation for Anthropic's Claude models.
Provides integration with Claude-2 and Claude-3 through the Anthropic API.
"""

import os
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from ...base_llm import BaseLLM
from src.utils.logger import setup_logger, log_error
import anthropic
from anthropic.types import MessageParam

logger = setup_logger(__name__)

class TokenQuotaManager:
    """
    Manages token quota and rate limiting for Anthropic API requests.
    
    Implements a token bucket algorithm to stay within Anthropic's rate limits.
    Default rate limit is 200,000 tokens per minute for Anthropic API.
    
    Attributes:
        max_tokens_per_minute: Maximum tokens allowed per minute
        token_bucket: Current available tokens in the bucket
        last_refill_time: Last time the token bucket was refilled
        lock: Asyncio lock for thread safety
    """
    
    def __init__(self, max_tokens_per_minute: int = 200000):
        """Initialize token quota manager with rate limits."""
        # Use the environment variable if provided, otherwise default to 200,000
        self.max_tokens_per_minute = int(os.getenv("ANTHROPIC_MAX_TOKENS_PER_MINUTE", str(max_tokens_per_minute)))
        # Start with a full bucket
        self.token_bucket = self.max_tokens_per_minute
        self.last_refill_time = time.time()
        self.lock = asyncio.Lock()
        logger.info(f"Initialized TokenQuotaManager with limit of {self.max_tokens_per_minute} tokens per minute")
        
    async def refill_bucket(self):
        """Refill the token bucket based on elapsed time."""
        now = time.time()
        elapsed_seconds = now - self.last_refill_time
        # Calculate tokens to add (tokens per second * elapsed seconds)
        tokens_per_second = self.max_tokens_per_minute / 60.0
        tokens_to_add = int(tokens_per_second * elapsed_seconds)
        
        # Add tokens, but don't exceed the maximum
        self.token_bucket = min(self.token_bucket + tokens_to_add, self.max_tokens_per_minute)
        self.last_refill_time = now
        logger.debug(f"Refilled token bucket. Current tokens: {self.token_bucket}/{self.max_tokens_per_minute}")
    
    async def request_tokens(self, token_estimate: int) -> float:
        """
        Request tokens from the bucket for an API call.
        
        Args:
            token_estimate: Estimated number of tokens needed for the request
            
        Returns:
            float: Wait time in seconds if tokens aren't immediately available
            
        Notes:
            If there aren't enough tokens, this will calculate how long to wait
            before retrying based on the token refill rate.
        """
        async with self.lock:
            # First refill the bucket based on elapsed time
            await self.refill_bucket()
            
            # If we have enough tokens, consume them and return no wait time
            if self.token_bucket >= token_estimate:
                self.token_bucket -= token_estimate
                logger.debug(f"Consumed {token_estimate} tokens. Remaining: {self.token_bucket}")
                return 0.0
            
            # Calculate wait time to accumulate enough tokens
            tokens_needed = token_estimate - self.token_bucket
            tokens_per_second = self.max_tokens_per_minute / 60.0
            wait_time = tokens_needed / tokens_per_second
            
            # Add a small buffer (10%) to ensure we'll have enough tokens after waiting
            wait_time *= 1.1
            
            logger.warning(f"Token bucket has insufficient tokens: {self.token_bucket}/{token_estimate} needed")
            logger.warning(f"Need to wait {wait_time:.2f} seconds to accumulate enough tokens")
            
            return wait_time

class ClaudeClient(BaseLLM):
    """
    Client implementation for Anthropic's Claude models.
    
    Handles authentication and interaction with Claude models through the Anthropic API.
    Supports Claude-2, Claude-3.5, and Claude-3.7 model variants.
    Some implementations support system prompt caching to reduce token usage.
    
    Args:
        model_name (str): Name of the Claude model to use (CLAUDE_2, CLAUDE_3_5, or CLAUDE_3_7)
        
    Attributes:
        model_id (str): The actual model identifier used by Anthropic
        client (anthropic.Anthropic): The underlying Anthropic client
        system_prompt_cache_enabled (bool): Whether to enable caching for system prompts
        system_prompt_cache_hits (int): Counter for system prompt cache hits
        token_quota_manager (TokenQuotaManager): Manager for token quota and rate limiting
    """
    
    async def initialize(self) -> None:
        """Initialize the Claude client with API credentials."""
        try:
            self.model_id = os.getenv(f"{self.model_name}_MODEL_ID")
            
            # Enable system prompt caching by default
            self.system_prompt_cache_enabled = os.getenv("ENABLE_SYSTEM_PROMPT_CACHE", "True").lower() == "true"
            
            # Initialize cache statistics
            self.system_prompt_cache_hits = 0
            self.total_requests_with_system = 0
            
            # Initialize the direct Anthropic client
            self.client = anthropic.AsyncAnthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            
            # Initialize token quota manager
            max_tokens_per_minute = int(os.getenv("ANTHROPIC_MAX_TOKENS_PER_MINUTE", "200000"))
            self.token_quota_manager = TokenQuotaManager(max_tokens_per_minute)
            
            # Default parameters
            self.default_temperature = 0.7
            self.default_max_tokens = 4000
            
            logger.info(f"Initialized {self.model_name} client with model {self.model_id}")
            if self.system_prompt_cache_enabled:
                logger.info(f"System prompt caching enabled for {self.model_name}")
            else:
                logger.info(f"System prompt caching disabled for {self.model_name}")
                
        except Exception as e:
            log_error(logger, f"Failed to initialize Claude client for {self.model_id}", e)
            raise
            
    async def _retry_with_exponential_backoff(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Retry a function with exponential backoff, with special handling for rate limits.
        
        Uses environment variables:
            MAX_API_RETRIES: Maximum number of retry attempts (default: 10)
            API_INITIAL_DELAY: Initial delay in seconds (default: 1.0)
            API_EXP_BASE: Base for exponential backoff (default: 2.0)
            API_MAX_DELAY: Maximum delay between retries in seconds (default: 60.0)
            
        Args:
            func: The async function to retry
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function if successful
            
        Raises:
            Exception: If all retry attempts fail
        """
        # Get configuration from environment variables
        max_retries = int(os.getenv("MAX_API_RETRIES", "10"))
        initial_delay = float(os.getenv("API_INITIAL_DELAY", "1.0"))
        exponential_base = float(os.getenv("API_EXP_BASE", "2.0"))
        max_delay = float(os.getenv("API_MAX_DELAY", "60.0"))
        
        retry_count = 0
        current_delay = initial_delay
        last_exception = None

        logger.debug(f"Using retry config: max_retries={max_retries}, initial_delay={initial_delay}, " +
                   f"exponential_base={exponential_base}, max_delay={max_delay}")

        while retry_count < max_retries:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                retry_count += 1
                
                # Log detailed error information
                logger.error(f"Request failed (attempt {retry_count}/{max_retries})")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {str(e)}")
                
                if retry_count == max_retries:
                    logger.critical(
                        f"Max retries ({max_retries}) reached. "
                        f"Final error: {type(e).__name__}: {str(e)}"
                    )
                    raise last_exception
                
                # Calculate next delay with exponential backoff
                wait_time = min(current_delay * (exponential_base ** (retry_count - 1)), max_delay)
                
                # Adjust delay based on error type
                if isinstance(e, anthropic.RateLimitError) or "rate_limit_error" in str(e) or "429" in str(e):
                    # For rate limit errors, use a longer delay
                    wait_time = min(wait_time * 4, max_delay)
                    logger.warning(
                        f"Rate limit error encountered. "
                        f"Implementing longer delay of {wait_time:.2f} seconds..."
                    )
                    
                    # Also pause our token bucket by reserving max tokens
                    # This ensures we'll wait until our quota refreshes
                    token_refresh_wait = 60.0  # Wait for a full minute
                    logger.warning(f"Pausing token usage for {token_refresh_wait} seconds to refresh quota")
                    await asyncio.sleep(token_refresh_wait)
                elif "529" in str(e) or "502" in str(e):
                    # For gateway errors, use a moderate delay
                    wait_time = min(wait_time * 2, max_delay)
                    logger.warning(
                        f"Gateway error encountered. "
                        f"Implementing moderate delay of {wait_time:.2f} seconds..."
                    )
                else:
                    logger.warning(
                        f"Attempt {retry_count} failed. "
                        f"Retrying in {wait_time:.2f} seconds..."
                    )
                
                # Log retry attempt details
                logger.info(
                    f"Retry {retry_count}/{max_retries}: "
                    f"Waiting {wait_time:.2f}s before next attempt"
                )
                
                # Use asyncio.sleep instead of time.sleep for non-blocking wait
                await asyncio.sleep(wait_time)
                current_delay = wait_time

        # Ensure we don't try to raise None
        if last_exception is not None:
            raise last_exception
        else:
            # This should never happen as we'd have returned from the function if successful
            raise RuntimeError("Max retries reached but no exception was captured")

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        This is a rough approximation - about 4 characters per token for English text.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        return len(text) // 4

    async def generate_response(self,
                              messages: List[Dict[str, str]],
                              max_tokens: int = 4000,
                              temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a response using the Claude model with retry logic.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens in response. Defaults to 4000.
            temperature: Sampling temperature. Defaults to 0.7.
            
        Returns:
            Dict containing model response and metadata
            
        Raises:
            Exception: If response generation fails after all retries
        """
        try:
            logger.debug(f"Claude generating response with {len(messages)} messages")
            
            # Separate messages by role
            system_messages = [msg for msg in messages if msg["role"] == "system"]
            user_messages = [msg for msg in messages if msg["role"] == "user"]
            assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
            
            logger.debug(f"Message breakdown - System: {len(system_messages)}, "
                       f"User: {len(user_messages)}, Assistant: {len(assistant_messages)}")
            
            # Check if we have system messages to determine if we should use caching
            has_system_messages = len(system_messages) > 0
            
            # Track requests with system messages for cache statistics
            if has_system_messages:
                self.total_requests_with_system += 1
            
            # Determine if we should use cache_control for this request
            # Only use cache_control if system prompt caching is enabled AND we have system messages
            use_cache_control = self.system_prompt_cache_enabled and has_system_messages
            
            # Estimate system prompt tokens for logging
            system_token_estimate = 0
            if has_system_messages:
                for msg in system_messages:
                    system_token_estimate += self._estimate_tokens(msg["content"])
            
            if use_cache_control:
                # Increment cache hit counter (assuming cache will be used)
                self.system_prompt_cache_hits += 1
                
                # Calculate cache hit rate
                cache_hit_rate = (self.system_prompt_cache_hits / self.total_requests_with_system) * 100
                
                logger.debug(f"Using system prompt caching for this request (hit #{self.system_prompt_cache_hits})")
                logger.debug(f"System prompt cache hit rate: {cache_hit_rate:.1f}%")
                logger.debug(f"Estimated system prompt tokens being cached: ~{system_token_estimate}")
            elif has_system_messages:
                logger.debug(f"System prompt caching disabled, sending ~{system_token_estimate} tokens uncached")
            
            # Convert to Anthropic API format
            anthropic_messages = []
            
            # Add user and assistant messages in the correct order
            for msg in user_messages + assistant_messages:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Prepare system content with caching if enabled
            system_content = None
            if has_system_messages:
                # For multiple system messages, join them with newlines
                if len(system_messages) == 1:
                    # Single system message
                    system_content = [{"type": "text", "text": system_messages[0]["content"]}]
                    
                    # Add cache_control if enabled
                    if use_cache_control:
                        # Type annotation to help the linter understand this is valid
                        cache_control_value: Dict[str, str] = {"type": "ephemeral"}
                        system_content[0]["cache_control"] = cache_control_value  # type: ignore
                else:
                    # Multiple system messages - create a list of system content blocks
                    system_content = []
                    for msg in system_messages:
                        content_block = {"type": "text", "text": msg["content"]}
                        # Add cache_control if enabled
                        if use_cache_control:
                            # Type annotation to help the linter understand this is valid
                            cache_control_value: Dict[str, str] = {"type": "ephemeral"}
                            content_block["cache_control"] = cache_control_value  # type: ignore
                        system_content.append(content_block)
            
            # Estimate total input tokens for rate limiting
            user_token_estimate = 0
            for msg in user_messages:
                user_token_estimate += self._estimate_tokens(msg["content"])
            
            assistant_token_estimate = 0
            for msg in assistant_messages:
                assistant_token_estimate += self._estimate_tokens(msg["content"])
            
            # Calculate total token estimate including max output tokens
            total_token_estimate = system_token_estimate + user_token_estimate + assistant_token_estimate + max_tokens
            logger.debug(f"Estimated total tokens for request: ~{total_token_estimate}")
            
            # Check token quota and wait if necessary
            wait_time = await self.token_quota_manager.request_tokens(total_token_estimate)
            if wait_time > 0:
                logger.warning(f"Token quota exceeded. Waiting {wait_time:.2f} seconds before proceeding")
                await asyncio.sleep(wait_time)
            
            logger.debug(f"Sending request to Claude with {len(anthropic_messages)} messages")
            
            # Generate response with retry logic
            response = await self._retry_with_exponential_backoff(
                self.client.messages.create,
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_content,
                messages=anthropic_messages
            )
            
            # Log the full response at INFO level
            logger.debug(f"Received response from Claude:")
            logger.debug(f"Model: {self.model_id}")
            logger.debug(f"Response content: {response.content[0].text}")
            
            # Log usage statistics if available
            if hasattr(response, 'usage'):
                logger.debug(f"Token usage: {response.usage}")
                if hasattr(response.usage, 'cache_creation_input_tokens'):
                    logger.debug(f"Cache creation tokens: {response.usage.cache_creation_input_tokens}")
                if hasattr(response.usage, 'cache_read_input_tokens'):
                    logger.debug(f"Cache read tokens: {response.usage.cache_read_input_tokens}")
            
            # Log cache statistics after each response
            if self.system_prompt_cache_enabled and self.total_requests_with_system > 0:
                cache_hit_rate = (self.system_prompt_cache_hits / self.total_requests_with_system) * 100
                logger.debug(f"System prompt cache statistics: {self.system_prompt_cache_hits}/{self.total_requests_with_system} hits ({cache_hit_rate:.1f}%)")
                
                # Estimate token savings if we have system messages
                if system_token_estimate > 0:
                    estimated_savings = system_token_estimate * (self.system_prompt_cache_hits - 1)  # -1 because first hit isn't a saving
                    if estimated_savings > 0:
                        logger.debug(f"Estimated token savings from system prompt caching: ~{estimated_savings} tokens")
            
            return {
                "model": self.model_id,
                "content": response.content[0].text,
                "role": "assistant"
            }
            
        except Exception as e:
            log_error(logger, f"Failed to generate response from Claude model {self.model_id}", e)
            raise 