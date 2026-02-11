"""
LLM Configuration Module
Centralized configuration for all LLM model usage in the project
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Configuration for LLM models"""
    
    model_name: str = " "
    base_url: str = " "
    api_key: str = field(default_factory=lambda: os.environ.get(
        "OPENAI_API_KEY", 
        " "
    ))
    temperature: float = 0.7
    max_tokens: int = 2000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format"""
        return {
            "model_name": self.model_name,
            "client_args": {"base_url": self.base_url},
            "api_key": self.api_key,
            "generate_kwargs": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        }
    
    def get_client_args(self) -> Dict[str, Any]:
        """Get client arguments"""
        return {"base_url": self.base_url}
    
    def get_generate_kwargs(self) -> Dict[str, Any]:
        """Get generation arguments"""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


def create_llm_model(config: Optional[LLMConfig] = None, **kwargs):
    """
    Create an OpenAIChatModel instance with the given configuration
    
    Args:
        config: LLM configuration object. If None, uses default configuration
        **kwargs: Additional keyword arguments to override config values
        
    Returns:
        OpenAIChatModel instance
        
    Example:
        # Use default configuration
        model = create_llm_model()
        
        # Use custom configuration
        config = LLMConfig(temperature=0.5, max_tokens=1000)
        model = create_llm_model(config)
        
        # Override specific parameters
        model = create_llm_model(temperature=0.7)
    """
    from agentscope.model import OpenAIChatModel
    
    if config is None:
        config = LLMConfig()
    
    # Merge config with kwargs overrides
    model_params = config.to_dict()
    
    if "model_name" in kwargs:
        model_params["model_name"] = kwargs["model_name"]
    if "base_url" in kwargs:
        model_params["client_args"]["base_url"] = kwargs["base_url"]
    if "api_key" in kwargs:
        model_params["api_key"] = kwargs["api_key"]
    if "temperature" in kwargs:
        model_params["generate_kwargs"]["temperature"] = kwargs["temperature"]
    if "max_tokens" in kwargs:
        model_params["generate_kwargs"]["max_tokens"] = kwargs["max_tokens"]
    
    return OpenAIChatModel(**model_params)


# Default configuration instance
DEFAULT_LLM_CONFIG = LLMConfig()
