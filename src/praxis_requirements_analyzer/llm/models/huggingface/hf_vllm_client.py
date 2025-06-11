"""
Client implementation for HuggingFace models using vLLM for local inference.
Provides high-performance LLM inference and serving using vLLM's optimizations.
"""

import os
from typing import Dict, Any, Optional, List, cast
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from ...base_llm import BaseLLM
from ....utils.logger import setup_logger, log_error
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = setup_logger(__name__)
_executor = ThreadPoolExecutor()

class HuggingFaceVLLMClient(BaseLLM):
    """
    Client implementation for HuggingFace models running locally with vLLM.
    
    Handles initialization and interaction with models running locally using vLLM's
    optimized inference engine with PagedAttention.
    
    Args:
        model_name (str): Name of the HuggingFace model to use
        max_tokens (int): Maximum number of tokens for generation
        
    Attributes:
        model_id (str): The actual model identifier on HuggingFace
        client (vllm.LLM): The underlying vLLM engine instance
    """
    
    def __init__(self, model_name: str, max_tokens: int = 2048):
        """
        Initialize the HuggingFace vLLM client.
        
        Args:
            model_name (str): Name of the HuggingFace model to use
            max_tokens (int): Maximum number of tokens for generation
        """
        super().__init__(model_name)
        self._max_tokens = max_tokens
        self._llm: Optional[LLM] = None
        self._sampling_params: Optional[SamplingParams] = None

    async def initialize(self) -> None:
        """
        Initialize the vLLM engine with the specified model.
        
        Sets up the vLLM engine with model configuration.
        The initialization happens in a separate thread to avoid blocking the event loop
        since vLLM initialization can be compute-intensive.
        
        Raises:
            Exception: If initialization fails due to missing model or vLLM errors
        """
        try:
            # Get model ID from environment variables
            self.model_id = os.getenv(f"{self.model_name}_MODEL_ID")
            if not self.model_id:
                raise ValueError(f"Model ID not found for {self.model_name} in environment variables")
            
            logger.info(f"Loading model ID for {self.model_name}: {self.model_id}")

            # Set max tokens
            self.max_tokens = getattr(self, 'max_tokens', 2048)
            logger.info(f"Model {self.model_name} using {self.max_tokens} tokens")
            
            # Initialize default sampling parameters
            self._sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=self.max_tokens
            )
            
            # Initialize vLLM in a separate thread to avoid blocking
            await self._setup_client()
            
        except Exception as e:
            log_error(logger, f"Failed to initialize vLLM client for {self.model_name}", e)
            raise

    async def _setup_client(self) -> None:
        """
        Setup the vLLM engine with current configuration.
        
        This runs in a separate thread since model loading is compute-intensive.
        """
        def _initialize_vllm():
            """
            Initialize the vLLM engine with the current model.
            
            Hardware Requirements:
                - CUDA-compatible GPU with at least 8GB VRAM (16GB+ recommended)
                - 16GB+ system RAM
                
            Configuration Options:
                - dtype: Controls precision ("auto", "float16", "bfloat16", "float32")
                - gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
                - swap_space: Amount of CPU RAM to use as swap in GB
                - tensor_parallel_size: Number of GPUs for tensor parallelism
                - trust_remote_code: Whether to trust remote code in model repos
                
            Returns:
                vllm.LLM: Initialized LLM instance
                
            Raises:
                ValueError: If model_id is missing
                Exception: For vLLM initialization errors (CUDA, memory issues)
            """
            try:
                # Ensure model_id is not None
                if not self.model_id:
                    raise ValueError(f"Missing model_id for {self.model_name}")
                
                # Set up vLLM engine with the specified model
                return LLM(
                    model=self.model_id,
                    dtype="auto",  # Automatically determine the appropriate dtype
                    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
                    swap_space=4,  # Use 4GB of CPU RAM as swap space
                    tensor_parallel_size=1  # Use only one GPU by default
                )
            except Exception as e:
                log_error(logger, f"Error initializing vLLM for {self.model_id}", e)
                raise
                
        # Run initialization in a thread pool
        loop = asyncio.get_event_loop()
        self._llm = await loop.run_in_executor(_executor, _initialize_vllm)
        logger.info(f"Initialized {self.model_name} vLLM client with model {self.model_id}")

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        self._max_tokens = value
        if hasattr(self, '_sampling_params') and self._sampling_params is not None:
            self._sampling_params.max_tokens = value

    async def generate_response(self,
                              messages: List[Dict[str, str]],
                              max_tokens: Optional[int] = None,
                              temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a response using the local vLLM engine.
        
        Args:
            messages: List of message dictionaries with role and content
            max_tokens: Maximum number of tokens to generate (overrides instance setting)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            
        Returns:
            Dict containing the generated response and metadata
            
        Raises:
            Exception: If generation fails
        """
        try:
            logger.info(f"vLLM generating response with {len(messages)} messages")
            
            # Combine messages into a single prompt with clear role separation
            prompt_parts = []
            
            # Process messages in order: system -> user -> assistant
            system_messages = [msg for msg in messages if msg["role"] == "system"]
            user_messages = [msg for msg in messages if msg["role"] == "user"]
            assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
            
            logger.info(f"Message breakdown - System: {len(system_messages)}, "
                       f"User: {len(user_messages)}, Assistant: {len(assistant_messages)}")
            
            # Add system instructions first
            for msg in system_messages:
                logger.info("Adding system message")
                logger.debug(f"System message content: {msg['content'][:100]}...")
                prompt_parts.append(f"System: {msg['content']}\n")
            
            # Add conversation history
            for msg in user_messages:
                logger.info("Adding user message")
                logger.debug(f"User message content: {msg['content'][:100]}...")
                prompt_parts.append(f"User: {msg['content']}\n")
                
            for msg in assistant_messages:
                logger.info("Adding assistant message")
                logger.debug(f"Assistant message content: {msg['content'][:100]}...")
                prompt_parts.append(f"Assistant: {msg['content']}\n")
            
            prompt = "\n".join(prompt_parts)
            logger.info(f"Combined prompt length: {len(prompt)} chars")
            
            # Update sampling parameters if needed
            current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            if max_tokens is not None or temperature != 0.7:
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=0.95,
                    max_tokens=current_max_tokens
                )
            else:
                sampling_params = self._sampling_params
                if sampling_params is None:
                    # Fallback sampling params in case they weren't initialized
                    sampling_params = SamplingParams(
                        temperature=0.7,
                        top_p=0.95,
                        max_tokens=self.max_tokens
                    )
            
            logger.info("Sending prompt to vLLM engine")
            
            # Run generation in a separate thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            
            # Ensure the LLM engine is initialized
            if self._llm is None:
                raise ValueError(f"vLLM engine for {self.model_name} is not initialized")
                
            # Safely execute the vLLM generate call in a thread
            llm_instance = cast(LLM, self._llm)  # This helps with type checking
            
            # Use generate_for_async helper function to call vLLM API safely
            def generate_for_async(prompt_str: str, params: SamplingParams) -> RequestOutput:
                """
                Call vLLM generate with proper error handling in async context.
                
                This helper function safely bridges between asyncio and vLLM's
                synchronous API by providing structured error handling and response
                validation.
                
                Args:
                    prompt_str: The text prompt to send to the model
                    params: SamplingParams object with generation parameters
                    
                Returns:
                    RequestOutput: vLLM's response object
                    
                Raises:
                    ValueError: If generation fails or returns empty results
                """
                # vLLM generate returns a list of RequestOutput objects
                outputs = llm_instance.generate(prompt_str, params)
                if not outputs:
                    raise ValueError("No result returned from vLLM engine")
                    
                # Get the first result
                if not outputs[0].outputs:
                    raise ValueError("No output text in vLLM response")
                    
                return outputs[0]
                
            # Run the generation in a separate thread
            response = await loop.run_in_executor(
                _executor,
                lambda: generate_for_async(prompt, sampling_params)
            )
            
            # Extract the generated text safely
            if not hasattr(response, 'outputs') or not response.outputs:
                raise ValueError("Invalid response format from vLLM")
                
            first_output = response.outputs[0]
            generated_text = first_output.text if hasattr(first_output, 'text') else ""
            
            if not generated_text:
                raise ValueError("Empty text generated by vLLM")
            
            logger.info("Received response from vLLM")
            logger.debug(f"Raw response from vLLM:\n{generated_text}")
            
            return {
                "model": self.model_id,
                "content": generated_text,
                "role": "assistant"
            }
            
        except Exception as e:
            log_error(logger, f"Failed to generate response from vLLM with model {self.model_id}", e)
            raise 