"""
rag.py
-------

This module implements the RAGChatbot class, which provides a Retrieval-Augmented Generation (RAG) chatbot using an Ollama language model and a ChromaDB vector database.

Features:
- Processes user queries by retrieving relevant documents from a vector database and generating responses using a language model.
- Stores new information in the knowledge base for future retrieval.
- Prepares context from retrieved documents to enhance response quality.
- Provides methods to get knowledge base info, clear the knowledge base, and search documents directly.

Dependencies:
- ollama_client: Interface to the Ollama language model for response generation.
- chromadb_client: Interface to ChromaDB for vector-based document retrieval and storage.

Typical usage:
    chatbot = RAGChatbot(ollama_client, chromadb_client)
    response = chatbot.process_query("What is RAG?")
    chatbot.store_information("RAG stands for Retrieval-Augmented Generation.")
"""


import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RAGChatbot:
    def __init__(self, ollama_client, chromadb_client):
        self.ollama_client = ollama_client
        self.chromadb_client = chromadb_client

        # Validate that clients are properly initialized
        if not self.ollama_client:
            raise ValueError("OllamaClient is required")
        if not self.chromadb_client:
            raise ValueError("ChromaDBClient is required")

        logger.info("RAG Chatbot initialized successfully")

    def process_query(self, user_query: str, top_k: int = 3) -> str:
        """Process a user query using RAG (Retrieval-Augmented Generation)."""
        try:
            if not user_query or not user_query.strip():
                return "Please provide a valid question or query."

            # Clean and validate input
            user_query = user_query.strip()

            # Retrieve relevant information from ChromaDB
            logger.info(
                f"Retrieving relevant documents for query: {user_query[:50]}..."
            )
            relevant_docs = self.chromadb_client.retrieve_vector(
                user_query, top_k=top_k
            )

            # Prepare context from retrieved documents
            context = self._prepare_context(relevant_docs)

            # Generate response using Ollama with context
            logger.info("Generating response with Ollama...")
            response = self.ollama_client.generate_response(user_query, context)

            # Post-process response
            response = self._post_process_response(response)

            logger.info("Query processed successfully")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}. Please try again or check if the services are running properly."

    def store_information(
        self, information: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store information in ChromaDB for future retrieval."""
        try:
            if not information or not information.strip():
                raise ValueError("Information cannot be empty")

            # Clean and preprocess the information
            cleaned_info = self._preprocess_text(information)

            if not cleaned_info:
                raise ValueError("Information is empty after preprocessing")

            # Store in ChromaDB
            doc_id = self.chromadb_client.store_vector(cleaned_info, metadata)

            logger.info(f"Information stored successfully with ID: {doc_id}")
            return doc_id

        except Exception as e:
            logger.error(f"Error storing information: {e}")
            raise RuntimeError(f"Failed to store information: {e}")

    def _prepare_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved documents."""
        if not relevant_docs:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            similarity = doc.get("similarity", 0)
            document = doc.get("document", "")

            # Only include documents with reasonable similarity
            if similarity > 0.3:  # Threshold for relevance
                context_parts.append(
                    f"Source {i} (similarity: {similarity:.2f}):\n{document}"
                )

        if not context_parts:
            return "No highly relevant information found in the knowledge base."

        return "\n\n".join(context_parts)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before storing."""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters that might cause issues
        text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)]", "", text)

        # Limit text length to prevent memory issues
        max_length = 10000
        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text.strip()

    def _post_process_response(self, response: str) -> str:
        """Post-process the generated response."""
        # Clean up the response
        response = response.strip()

        # Remove any context markers that might have leaked through
        response = re.sub(r"^Context:.*?\n\n", "", response, flags=re.DOTALL)
        response = re.sub(r"^Answer:\s*", "", response)

        # Ensure response is not empty
        if not response:
            return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."

        return response

    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """Get information about the knowledge base."""
        try:
            return self.chromadb_client.get_collection_info()
        except Exception as e:
            logger.error(f"Error getting knowledge base info: {e}")
            return {"error": str(e)}

    def clear_knowledge_base(self) -> bool:
        """Clear all documents from the knowledge base."""
        try:
            return self.chromadb_client.clear_collection()
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return False

    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search documents without generating a response."""
        try:
            return self.chromadb_client.retrieve_vector(query, top_k)
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
