"""
LLM Manager module for handling multiple LLM providers and models.

This module provides centralized management of model initialization and interactions
with various LLM providers including Anthropic, OpenAI, and HuggingFace.

Example:
    >>> manager = LLMManager()
    >>> await manager.initialize_models()
    >>> response = await manager.agenerate(
    ...     model_name="claude-2",
    ...     prompt="System instruction",
    ...     messages=[{"role": "user", "content": "Hello"}]
    ... )

Attributes:
    logger: Logger instance for this module

Classes:
    LLMManager: Main class for managing LLM model interactions
"""

from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv
from ...utils.logger import setup_logger, log_error
from ..models.anthropic.claude_client import ClaudeClient
from ..models.huggingface.hf_client import HuggingFaceClient
# Import vLLM client only when needed
# from ..models.huggingface.hf_vllm_client import HuggingFaceVLLMClient
from ..models.openai.openai_client import OpenAIClient
from ..base_llm import BaseLLM

logger = setup_logger(__name__)

class LLMManager:
    """
    Manages different LLM models and their initialization.
    Handles model selection and generation requests.
    """
    
    def __init__(self):
        self.logger = logger
        self.models = {}
        self.model_configs = {
            'CLAUDE_2': {
                'env_key': 'CLAUDE_2_MODEL_ID',
                'default': 'claude-2.1'
            },
            'CLAUDE_3_5': {
                'env_key': 'CLAUDE_3_5_MODEL_ID',
                'default': 'claude-3-5-sonnet-20241022'
            },
            'CLAUDE_3_7': {
                'env_key': 'CLAUDE_3_7_MODEL_ID',
                'default': 'claude-3-7-sonnet-20250219'
            },
            'OPENAI': {
                'env_key': 'OPENAI_MODEL_ID',
                'default': 'gpt-4'
            },
            'ZEPHYR': {
                'env_key': 'ZEPHYR_MODEL_ID',
                'default': 'HuggingFaceH4/zephyr-7b-beta'
            },
            'PHI_3': {
                'env_key': 'PHI_3_MODEL_ID',
                'default': 'microsoft/Phi-3.5-mini-instruct'
            },
            'LLAMA_3': {
                'env_key': 'LLAMA_3_MODEL_ID',
                'default': 'meta-llama/Llama-3.2-3B-Instruct'
            },
            'LLAMA_4': {
                'env_key': 'LLAMA_4_MODEL_ID',
                'default': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct'
            },
            'DEEPSEEK_R1': {
                'env_key': 'DEEPSEEK_R1_DISTILL_QWEN_32B_MODEL_ID',
                'default': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
            }
        }
        # Check if vLLM is enabled
        self.enable_vllm = os.getenv('ENABLE_VLLM', 'False').lower() == 'true'
        if self.enable_vllm:
            self.logger.info("vLLM is enabled for local HuggingFace model inference")
        else:
            self.logger.info("Using HuggingFace Inference API (vLLM is disabled)")

    async def initialize_models(self):
        """Initialize all supported models."""
        self.logger.info("Starting model initialization")
        try:
            # Initialize Claude models
            claude_2_model = os.getenv('CLAUDE_2_MODEL_ID', 'claude-2.1')
            claude_3_5_model = os.getenv('CLAUDE_3_5_MODEL_ID', 'claude-3-5-sonnet-20241022')
            claude_3_7_model = os.getenv('CLAUDE_3_7_MODEL_ID', 'claude-3-7-sonnet-20250219')
            
            # Create separate instances for each Claude model
            claude_2_client = ClaudeClient(model_name='CLAUDE_2')
            claude_3_5_client = ClaudeClient(model_name='CLAUDE_3_5')
            claude_3_7_client = ClaudeClient(model_name='CLAUDE_3_7')
            
            await claude_2_client.initialize()
            await claude_3_5_client.initialize()
            await claude_3_7_client.initialize()
            
            self.models[claude_2_model] = claude_2_client
            self.models[claude_3_5_model] = claude_3_5_client
            self.models[claude_3_7_model] = claude_3_7_client
            
            self.logger.info(f"Initialized CLAUDE_2 client with model {claude_2_model}")
            self.logger.info(f"Initialized CLAUDE_3_5 client with model {claude_3_5_model}")
            self.logger.info(f"Initialized CLAUDE_3_7 client with model {claude_3_7_model}")

            # Initialize OpenAI
            openai_model = os.getenv('OPENAI_MODEL_ID', 'gpt-4o')
            openai_client = OpenAIClient(model_name='OPENAI')
            await openai_client.initialize()
            self.models[openai_model] = openai_client
            self.logger.info(f"Initialized OPENAI client with model {openai_model}")

            # Initialize HuggingFace models
            hf_models = {
                'ZEPHYR': os.getenv('ZEPHYR_MODEL_ID', 'HuggingFaceH4/zephyr-7b-beta'),
                'PHI_3': os.getenv('PHI_3_MODEL_ID', 'microsoft/Phi-3.5-mini-instruct'),
                'LLAMA_3': os.getenv('LLAMA_3_MODEL_ID', 'meta-llama/Llama-3.2-3B-Instruct'),
                'LLAMA_4': os.getenv('LLAMA_4_MODEL_ID', 'meta-llama/Llama-4-Maverick-17B-128E-Instruct'),
                'DEEPSEEK_R1': os.getenv('DEEPSEEK_R1_MODEL_ID', 'deepseek-ai/DeepSeek-R1'),
                'DEEPSEEK_R1_DISTILL_QWEN_32B': os.getenv('DEEPSEEK_R1_DISTILL_QWEN_32B_MODEL_ID', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B')
            }
            
            # Initialize each HuggingFace model with the appropriate client based on ENABLE_VLLM setting
            for model_key, model_id in hf_models.items():
                # Choose between vLLM client and regular client based on configuration
                if self.enable_vllm:
                    # Lazy import the HuggingFaceVLLMClient only when needed
                    from ..models.huggingface.hf_vllm_client import HuggingFaceVLLMClient
                    self.logger.info(f"Using vLLM for {model_key} with model {model_id}")
                    hf_client = HuggingFaceVLLMClient(model_name=model_key)
                else:
                    self.logger.info(f"Using HuggingFace API for {model_key} with model {model_id}")
                    hf_client = HuggingFaceClient(model_name=model_key)
                
                await hf_client.initialize()
                self.models[model_id] = hf_client
                self.logger.info(f"Initialized {model_key} client with model {model_id}")

            self.logger.info("Model initialization completed successfully")

        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}", exc_info=True)
            raise

    async def agenerate(self, model_name: str, messages: List[Dict], prompt: Optional[str] = None, current_role: Optional[str] = None) -> str:
        """
        Generate a response using the specified model.
        
        Args:
            model_name: Name of the model to use
            messages: List of message dictionaries
            prompt: Optional system prompt
            current_role: Current role (ACTOR, JUDGE, META_JUDGE)
        
        Returns:
            Generated response text
        """
        try:
            role_info = "UNKNOWN" if current_role is None else current_role
            self.logger.debug(f"[{role_info}] Starting generation with model: {model_name}")
            
            if model_name not in self.models:
                error_msg = f"Model {model_name} not found"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            client = self.models[model_name]
            # Handle None prompt value
            actual_prompt = "" if prompt is None else prompt
            response = await client.agenerate(
                model_name=model_name,
                messages=messages,
                prompt=actual_prompt
            )
            
            if not response:
                error_msg = f"Empty response from model {model_name}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            return response

        except Exception as e:
            role_info = "UNKNOWN" if current_role is None else current_role
            error_msg = f"[{role_info}] Error during generation: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error("Detailed error trace:", exc_info=True)
            raise ValueError(error_msg)

    async def test_model(self, 
                        model_name: str, 
                        prompt: str, 
                        system_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Test a specific model with a prompt.
        
        This method provides a simple way to test model functionality by sending
        a single prompt and optional system message.
        
        Args:
            model_name (str): Name of the model to test
            prompt (str): The prompt to send to the model
            system_message (Optional[str], optional): System message for chat models.
                Defaults to None.
            
        Returns:
            Dict[str, Any]: Results dictionary containing:
                - model: Name of the model used
                - response: Generated response from the model
                - timing: Performance metrics
                - error: Error message if an error occurred
                
        Raises:
            ValueError: If model_name is not found in initialized models
            Exception: If model testing fails
            
        Example:
            >>> result = await manager.test_model(
            ...     model_name="claude-2",
            ...     prompt="Hello!",
            ...     system_message="You are a helpful assistant"
            ... )
            >>> print(result["response"])
        """
        try:
            logger.info(f"Testing model: {model_name}")
            logger.debug(f"Prompt: {prompt}")
            
            model = self.models.get(model_name)
            if not model:
                error_msg = f"Model {model_name} not found"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            messages = []
            if system_message:
                messages.append({
                    "role": "system",
                    "content": system_message
                })
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = await model.generate_response(messages=messages)
            logger.debug(f"Response received from {model_name}")
            return response

        except Exception as e:
            log_error(logger, f"Error testing model {model_name}", e)
            return {
                "model": model_name,
                "error": str(e),
                "prompt": prompt
            } 