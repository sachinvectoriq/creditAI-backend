"""
Shared Azure OpenAI client singleton.
This ensures all services share the same LLM instance for efficiency.
"""
import asyncio
from typing import Optional, List, Dict, Any
from functools import lru_cache
import tiktoken
from openai import AsyncAzureOpenAI
from llama_index.llms.azure_openai import AzureOpenAI as LlamaAzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings as LlamaSettings
from llama_index.core.indices.prompt_helper import PromptHelper
import logging

from shared.config import get_settings

logger = logging.getLogger(__name__)


class AzureOpenAIClient:
    """Singleton Azure OpenAI client for all AI operations"""
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize Azure OpenAI clients if not already initialized"""
        if self._initialized:
            return
            
        settings = get_settings()
        
        try:
            # Async client for direct API calls
            self.async_client = AsyncAzureOpenAI(
                api_key=settings.azure_openai_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint,
            )
            
            # LlamaIndex LLM for RAG operations
            self.llama_llm = LlamaAzureOpenAI(
                model="gpt-4o",
                deployment_name=settings.azure_openai_deployment,
                api_key=settings.azure_openai_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                temperature=0.1,
                max_tokens=3000,
            )
            
            # Embedding model for vector operations
            self.embed_model = AzureOpenAIEmbedding(
                deployment_name=settings.azure_openai_embedding_deployment,
                api_key=settings.azure_openai_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
            )
            
            # Prompt helper for context management
            self.prompt_helper = PromptHelper(
                context_window=120000,
                num_output=6000,
                chunk_overlap_ratio=0.1,
            )
            
            # Configure global LlamaIndex settings
            LlamaSettings.llm = self.llama_llm
            LlamaSettings.embed_model = self.embed_model
            
            # Token encoder
            try:
                self.encoding = tiktoken.encoding_for_model("gpt-4")
            except Exception:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            
            self.deployment_name = settings.azure_openai_deployment
            self._initialized = True
            
            logger.info("Azure OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
    
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        **kwargs
    ) -> str:
        """
        Async completion using Azure OpenAI.
        
        Args:
            prompt: The prompt to complete
            max_tokens: Maximum tokens in response
            temperature: Temperature for response generation
            **kwargs: Additional parameters for the API
            
        Returns:
            Generated text response
        """
        try:
            response = await self.async_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in Azure OpenAI completion: {e}")
            raise
    
    async def complete_with_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 0.1,
        **kwargs
    ) -> str:
        """
        Async completion with custom messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens in response
            temperature: Temperature for response generation
            **kwargs: Additional parameters for the API
            
        Returns:
            Generated text response
        """
        try:
            response = await self.async_client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in Azure OpenAI completion: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, max_tokens: int = 8000) -> List[str]:
        """
        Split text into chunks that fit within token limits.
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = await self.async_client.embeddings.create(
                model=get_settings().azure_openai_embedding_deployment,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


@lru_cache(maxsize=1)
def get_openai_client() -> AzureOpenAIClient:
    """
    Get the singleton Azure OpenAI client instance.
    Using lru_cache ensures we only create one instance.
    
    Returns:
        AzureOpenAIClient singleton instance
    """
    return AzureOpenAIClient()


# Utility functions for common AI operations
async def analyze_text_with_prompt(text: str, analysis_prompt: str) -> str:
    """
    Analyze text with a custom prompt using the shared client.
    
    Args:
        text: Text to analyze
        analysis_prompt: Analysis instructions
        
    Returns:
        Analysis result
    """
    client = get_openai_client()
    
    # Check token limit and chunk if necessary
    full_prompt = f"{analysis_prompt}\n\nDocument:\n{text}"
    
    if client.count_tokens(full_prompt) > 100000:
        # Process in chunks if text is too long
        chunks = client.chunk_text(text, max_tokens=50000)
        results = []
        
        for i, chunk in enumerate(chunks):
            chunk_prompt = f"{analysis_prompt}\n\n(Part {i+1} of {len(chunks)})\n\nDocument:\n{chunk}"
            result = await client.complete(chunk_prompt)
            results.append(result)
        
        # Combine results
        combined = "\n\n".join(results)
        summary_prompt = f"Combine and summarize these analyses into a coherent response:\n\n{combined}"
        return await client.complete(summary_prompt)
    else:
        return await client.complete(full_prompt)