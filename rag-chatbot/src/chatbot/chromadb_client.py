import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ChromaDBClient:
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )

            # Initialize embedding model
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            logger.info(
                f"ChromaDB client initialized with collection: {collection_name}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        try:
            embedding = self.embedding_model.encode(text).tolist()
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}")

    def _generate_document_id(self, text: str) -> str:
        """Generate a unique ID for the document based on its content."""
        return hashlib.md5(text.encode()).hexdigest()

    def store_vector(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store text and its embedding in ChromaDB."""
        try:
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")

            # Generate embedding
            embedding = self._generate_embedding(text)

            # Generate unique ID
            doc_id = self._generate_document_id(text)

            # Prepare metadata
            if metadata is None:
                metadata = {}

            metadata.update({"text_length": len(text), "stored_at": str(uuid.uuid4())})

            # Store in ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id],
            )

            logger.info(f"Successfully stored document with ID: {doc_id}")
            return doc_id

        except Exception as e:
            logger.error(f"Error storing vector: {e}")
            raise RuntimeError(f"Failed to store document: {e}")

    def retrieve_vector(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar documents based on query text."""
        try:
            if not query_text or not query_text.strip():
                raise ValueError("Query text cannot be empty")

            # Generate embedding for query
            query_embedding = self._generate_embedding(query_text)

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    formatted_results.append(
                        {
                            "document": doc,
                            "metadata": (
                                results["metadatas"][0][i]
                                if results["metadatas"]
                                else {}
                            ),
                            "distance": (
                                results["distances"][0][i]
                                if results["distances"]
                                else 0.0
                            ),
                            "similarity": (
                                1 - results["distances"][0][i]
                                if results["distances"]
                                else 1.0
                            ),
                        }
                    )

            logger.info(f"Retrieved {len(formatted_results)} similar documents")
            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving vectors: {e}")
            raise RuntimeError(f"Failed to retrieve documents: {e}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by its ID."""
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Successfully deleted document with ID: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Successfully cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

    def search_by_metadata(
        self, metadata_filter: Dict[str, Any], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search documents by metadata filters."""
        try:
            results = self.collection.query(
                query_texts=[""],  # Empty query since we're filtering by metadata
                n_results=top_k,
                where=metadata_filter,
                include=["documents", "metadatas", "distances"],
            )

            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    formatted_results.append(
                        {
                            "document": doc,
                            "metadata": (
                                results["metadatas"][0][i]
                                if results["metadatas"]
                                else {}
                            ),
                            "distance": (
                                results["distances"][0][i]
                                if results["distances"]
                                else 0.0
                            ),
                        }
                    )

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            raise RuntimeError(f"Failed to search by metadata: {e}")
