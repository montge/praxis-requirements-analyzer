#!/usr/bin/env python3
"""
Main entry point for the Praxis Requirements Analyzer package.

This script demonstrates the basic usage of the requirements analysis system.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path to allow importing the package
sys.path.insert(0, str(Path(__file__).parent))

from praxis_requirements_analyzer.utils.logger import setup_logger
from praxis_requirements_analyzer.llm.manager import LLMManager

logger = setup_logger(__name__)

async def main():
    """Main function demonstrating package usage."""
    try:
        logger.info("Starting Praxis Requirements Analyzer")
        
        # Initialize LLM Manager
        logger.info("Initializing LLM Manager...")
        llm_manager = LLMManager()
        await llm_manager.initialize_models()
        
        logger.info("System initialized successfully!")
        logger.info("Available models: %s", list(llm_manager.models.keys()))
        
        # Example usage
        if llm_manager.models:
            model_name = list(llm_manager.models.keys())[0]
            logger.info(f"Testing model: {model_name}")
            
            test_messages = [
                {"role": "user", "content": "Hello! Please confirm the system is working."}
            ]
            
            response = await llm_manager.agenerate(
                model_name=model_name,
                messages=test_messages,
                prompt="You are a helpful assistant."
            )
            
            logger.info(f"Test response: {response}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 
 