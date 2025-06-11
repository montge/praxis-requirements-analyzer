"""
Base module for LLM (Large Language Model) implementations.
Provides abstract base class for different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from langchain.schema import HumanMessage, SystemMessage
from ...utils.logger import setup_logger

class BaseLLM(ABC):
    """
    Abstract base class for LLM implementations.
    
    This class defines the interface that all LLM implementations must follow.
    It provides basic functionality for model initialization and response generation.
    Some implementations (like ClaudeClient) may support additional features such as
    system prompt caching to reduce token usage.
    
    Args:
        model_name (str): Name of the model to use
        
    Attributes:
        model_name (str): Name of the model
        model_id (Optional[str]): Actual model identifier used by the provider
        client: The underlying client instance for the LLM provider
    """
    
    def __init__(self, model_name: str):
        """
        Initialize base LLM client.
        
        Args:
            model_name: Name/type of the model (e.g., 'CLAUDE_2', 'OPENAI', etc.)
        """
        self.model_name = model_name
        self.model_id = None
        self.client = None
        self.logger = setup_logger(f"src.llm.{self.__class__.__name__}")
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the LLM connection.
        
        This method should be implemented by each LLM provider to set up
        their specific client configuration and authentication.
        
        Raises:
            Exception: If initialization fails
        """
        raise NotImplementedError
    
    @abstractmethod
    async def generate_response(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional keyword arguments
            
        Returns:
            Dict[str, Any]: Response containing at minimum:
                - model: The model ID used
                - response: The generated response text
                - prompt: The original prompt
                
        Raises:
            Exception: If response generation fails
        """
        raise NotImplementedError

    async def agenerate(self, model_name: str, messages: List[Dict], prompt: Optional[str] = None) -> str:
        """
        Generate a response using the model.
        
        Args:
            model_name: Name of the model to use
            messages: List of message dictionaries
            prompt: Optional system prompt. Some implementations may cache system prompts
                to reduce token usage.
            
        Returns:
            Generated response text
        """
        try:
            # Add system prompt if provided and not empty
            if prompt and prompt.strip():
                messages.insert(0, {"role": "system", "content": prompt})
            
            response = await self.generate_response(messages)
            return response.get("content", "")
            
        except Exception as e:
            self.logger.error(f"Error in agenerate: {str(e)}", exc_info=True)
            raise 