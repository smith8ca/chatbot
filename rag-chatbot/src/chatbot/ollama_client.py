"""
Ollama Client for RAG Chatbot

This module provides a client interface for interacting with Ollama language models.
It uses the official ollama Python library for simplified and reliable communication
with the Ollama server.

Features:
    - Model management (list, pull, check availability)
    - Text generation with context support
    - Chat interface for conversational interactions
    - Error handling and logging
    - Configurable model parameters

Author: Charles A. Smith
Version: 1.0.0
"""

import logging
from typing import Any, Dict, List, Optional

import ollama

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama client.

        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL of the Ollama server
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

        # Configure the ollama client
        ollama.Client(host=self.base_url)

        logger.info(
            f"OllamaClient initialized with model: {model_name} at {self.base_url}"
        )

    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate a response using the Ollama language model.

        Args:
            prompt: The user's question or prompt
            context: Optional context information to include

        Returns:
            Generated response from the model

        Raises:
            RuntimeError: If response generation fails
        """
        try:
            # Prepare the full prompt with context if provided
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"

            # Generate response using ollama library
            response = ollama.generate(
                model=self.model_name,
                prompt=full_prompt,
                options={"temperature": 0.7, "top_p": 0.9, "num_predict": 1000},
            )

            if response and "response" in response:
                return response["response"].strip()
            else:
                raise RuntimeError("Invalid response format from Ollama")

        except ollama.ResponseError as e:
            logger.error(f"Ollama response error: {e}")
            raise RuntimeError(f"Ollama response error: {e}")
        except ollama.RequestError as e:
            logger.error(f"Ollama request error: {e}")
            raise RuntimeError(
                f"Failed to connect to Ollama server. Please ensure Ollama is running at {self.base_url}"
            )
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise RuntimeError(f"Failed to generate response: {e}")

    def check_model_availability(self) -> bool:
        """
        Check if the specified model is available.

        Returns:
            True if model is available, False otherwise
        """
        try:
            ollama.show(model=self.model_name)
            return True
        except ollama.ResponseError:
            return False
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False

    def list_available_models(self) -> List[str]:
        """
        List all available models.

        Returns:
            List of available model names
        """
        try:
            models = ollama.list()
            return [model["name"] for model in models.get("models", [])]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from the registry.

        Args:
            model_name: Name of the model to pull

        Returns:
            True if successful, False otherwise
        """
        try:
            ollama.pull(model=model_name)
            logger.info(f"Successfully pulled model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

    def chat(self, messages: List[Dict[str, str]], stream: bool = False) -> str:
        """
        Chat with the model using a conversation format.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            stream: Whether to stream the response

        Returns:
            The model's response
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=stream,
                options={"temperature": 0.7, "top_p": 0.9, "num_predict": 1000},
            )

            if stream:
                # Handle streaming response
                full_response = ""
                for chunk in response:
                    if "message" in chunk and "content" in chunk["message"]:
                        full_response += chunk["message"]["content"]
                return full_response
            else:
                return response["message"]["content"]

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise RuntimeError(f"Failed to chat with model: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary containing model information
        """
        try:
            return ollama.show(model=self.model_name)
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}
