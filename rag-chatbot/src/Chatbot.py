"""
RAG Chatbot Streamlit Application

This module provides the main Streamlit web interface for a Retrieval-Augmented Generation (RAG) chatbot.
The application combines Ollama language models with ChromaDB vector storage to create an intelligent
chatbot that can answer questions based on uploaded documents and stored knowledge.

Features:
    - Interactive chat interface with conversation history
    - Document upload and processing via sidebar
    - Real-time RAG-based question answering
    - Error handling and user feedback
    - Environment-based configuration

Dependencies:
    - Streamlit: Web interface framework
    - Ollama: Local LLM inference
    - ChromaDB: Vector database for document storage
    - Sentence Transformers: Text embeddings

Environment Variables:
    OLLAMA_MODEL: Name of the Ollama model to use (default: "llama2")
    OLLAMA_BASE_URL: Ollama server URL (default: "http://localhost:11434")
    CHROMA_COLLECTION: ChromaDB collection name (default: "rag_documents")
    CHROMA_PERSIST_DIR: ChromaDB storage directory (default: "./chroma_db")

Usage:
    Run the application with: streamlit run src/app.py

    Prerequisites:
    1. Install dependencies: pip install -r requirements.txt
    2. Install and start Ollama: ollama serve
    3. Pull a model: ollama pull llama2
    4. Configure environment variables (optional)

Author: Charles A. Smith
Version: 1.0.0
"""

import os
from datetime import datetime

import streamlit as st
from chatbot.chromadb_client import ChromaDBClient
from chatbot.document_processor import DocumentProcessor
from chatbot.feedback_manager import FeedbackManager
from chatbot.ollama_client import OllamaClient
from chatbot.rag import RAGChatbot
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():

    st.title("RAG Chatbot")
    st.write("Welcome to the Chuckworks Chatbot!")
    st.set_page_config(page_title="Chatbot", layout="wide")

    # Disclaimer
    with st.expander("Disclaimer"):
        st.info(
            "_This chatbot is for informational purposes only. Responses may not always be accurate or complete._ "
            "_Do not rely on this tool for critical decisions. Uploaded documents are processed locally and not shared externally._"
        )

    # Initialize clients with configuration
    try:
        ollama_client = OllamaClient(
            model_name=os.getenv("OLLAMA_MODEL", "llama2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

        chromadb_client = ChromaDBClient(
            collection_name=os.getenv("CHROMA_COLLECTION", "rag_documents"),
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        )

        chatbot = RAGChatbot(ollama_client, chromadb_client)

        # Initialize feedback manager
        feedback_manager = FeedbackManager()

        # Initialize session state for conversation history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display conversation history
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Add feedback buttons for assistant messages
                if message["role"] == "assistant" and "feedback" not in message:
                    col1, col2, col3 = st.columns([1, 1, 10])
                    with col1:
                        if st.button("👍", key=f"thumbs_up_{i}", help="Good answer"):
                            st.session_state.messages[i]["feedback"] = "positive"
                            # Save feedback to file
                            message_id = f"msg_{i}_{len(st.session_state.messages)}"
                            user_query = (
                                st.session_state.messages[i - 1]["content"]
                                if i > 0
                                else "Unknown"
                            )
                            feedback_manager.add_feedback(
                                message_id=message_id,
                                user_query=user_query,
                                response=message["content"],
                                feedback="positive",
                            )
                            st.rerun()
                    with col2:
                        if st.button("👎", key=f"thumbs_down_{i}", help="Poor answer"):
                            st.session_state.messages[i]["feedback"] = "negative"
                            # Save feedback to file
                            message_id = f"msg_{i}_{len(st.session_state.messages)}"
                            user_query = (
                                st.session_state.messages[i - 1]["content"]
                                if i > 0
                                else "Unknown"
                            )
                            feedback_manager.add_feedback(
                                message_id=message_id,
                                user_query=user_query,
                                response=message["content"],
                                feedback="negative",
                            )
                            st.rerun()

                # Show feedback status if given
                if "feedback" in message:
                    if message["feedback"] == "positive":
                        st.success("✅ Feedback: Good answer")
                    elif message["feedback"] == "negative":
                        st.error("❌ Feedback: Poor answer")

        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = chatbot.process_query(prompt)
                        st.markdown(response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )

        # Sidebar for document upload and analytics
        with st.sidebar:
            st.header("Document Management")

            # Initialize document processor
            doc_processor = DocumentProcessor()
            supported_types = list(doc_processor.get_supported_extensions())

            uploaded_file = st.file_uploader(
                "Upload a document",
                type=supported_types,
                help=f"Supported formats: {', '.join(supported_types)}",
            )

            if uploaded_file is not None:
                # Show file info
                st.info(f"File: {uploaded_file.name} ({uploaded_file.size} bytes)")

                if st.button("Process Document"):
                    try:
                        # Read file content
                        file_content = uploaded_file.read()

                        # Process the document
                        result = doc_processor.process_file(
                            file_content, uploaded_file.name
                        )

                        # Store in chatbot
                        doc_id = chatbot.store_information(
                            result["text"], result["metadata"]
                        )

                        st.success(
                            f"Document processed and stored successfully! (ID: {doc_id[:8]}...)"
                        )
                        st.info(
                            f"Extracted {result['metadata']['text_length']} characters"
                        )

                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                        st.info(
                            "Make sure the file is not corrupted and is in a supported format."
                        )

            # st.divider()
            # st.caption("Looking for analytics and exports?")
            # st.info(
            #     "Open the Feedback page from the sidebar navigation to view analytics and manage feedback data."
            # )

    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        st.info("Please check your configuration and ensure Ollama is running.")


if __name__ == "__main__":
    main()
