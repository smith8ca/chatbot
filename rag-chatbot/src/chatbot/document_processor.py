"""
Document Processor for RAG Chatbot

This module handles processing of various document types including PDF, TXT, and MD files.
It extracts text content from different file formats and prepares them for storage in the vector database.

Supported formats:
    - PDF: Extracts text using PyPDF2
    - TXT: Plain text files
    - MD: Markdown files

Author: Charles A. Smith
Version: 1.0.0
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import PyPDF2

    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("PyPDF2 not available. PDF processing will be disabled.")

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles processing of various document types."""

    def __init__(self):
        """Initialize the document processor."""
        self.supported_extensions = {".txt", ".md"}
        if PDF_SUPPORT:
            self.supported_extensions.add(".pdf")

        logger.info(
            f"DocumentProcessor initialized. Supported formats: {self.supported_extensions}"
        )

    def process_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process a file and extract its text content.

        Args:
            file_content: Raw file content as bytes
            filename: Name of the file (used to determine file type)

        Returns:
            Dictionary containing extracted text and metadata

        Raises:
            ValueError: If file type is not supported
            RuntimeError: If file processing fails
        """
        try:
            # Determine file type from extension
            file_extension = Path(filename).suffix.lower()

            if file_extension not in self.supported_extensions:
                raise ValueError(
                    f"Unsupported file type: {file_extension}. Supported types: {self.supported_extensions}"
                )

            # Process based on file type
            if file_extension == ".pdf":
                text_content = self._process_pdf(file_content)
            elif file_extension in [".txt", ".md"]:
                text_content = self._process_text(file_content)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # Prepare metadata
            metadata = {
                "filename": filename,
                "file_type": file_extension,
                "text_length": len(text_content),
                "processed_at": str(Path(filename).stem),
            }

            logger.info(
                f"Successfully processed {filename}: {len(text_content)} characters extracted"
            )

            return {"text": text_content, "metadata": metadata}

        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            raise RuntimeError(f"Failed to process file {filename}: {e}")

    def _process_pdf(self, file_content: bytes) -> str:
        """
        Extract text from PDF file.

        Args:
            file_content: PDF file content as bytes

        Returns:
            Extracted text content

        Raises:
            RuntimeError: If PDF processing fails
        """
        if not PDF_SUPPORT:
            raise RuntimeError("PDF processing not available. Please install PyPDF2.")

        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text_content = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    logger.warning(
                        f"Failed to extract text from page {page_num + 1}: {e}"
                    )
                    continue

            if not text_content.strip():
                raise RuntimeError("No text content could be extracted from PDF")

            return text_content.strip()

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise RuntimeError(f"Failed to extract text from PDF: {e}")

    def _process_text(self, file_content: bytes) -> str:
        """
        Extract text from plain text or markdown file.

        Args:
            file_content: File content as bytes

        Returns:
            Extracted text content

        Raises:
            RuntimeError: If text processing fails
        """
        try:
            # Try different encodings
            encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]

            for encoding in encodings:
                try:
                    text_content = file_content.decode(encoding)
                    return text_content.strip()
                except UnicodeDecodeError:
                    continue

            # If all encodings fail, use error handling
            text_content = file_content.decode("utf-8", errors="replace")
            logger.warning("Used UTF-8 with error replacement for text file")
            return text_content.strip()

        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            raise RuntimeError(f"Failed to extract text from file: {e}")

    def get_supported_extensions(self) -> set:
        """
        Get list of supported file extensions.

        Returns:
            Set of supported file extensions
        """
        return self.supported_extensions.copy()

    def is_supported(self, filename: str) -> bool:
        """
        Check if a file type is supported.

        Args:
            filename: Name of the file to check

        Returns:
            True if file type is supported, False otherwise
        """
        file_extension = Path(filename).suffix.lower()
        return file_extension in self.supported_extensions
