# Retrieval-Augmented Generation (RAG) Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot using Streamlit for the user interface, Ollama for language models, and ChromaDB for a local vector database.

## Project Structure

```
rag-chatbot
├── src
│   ├── app.py                # Main entry point of the application
│   ├── chatbot
│   │   ├── __init__.py       # Initializes the chatbot package
│   │   ├── rag.py            # RAG logic implementation
│   │   ├── ollama_client.py   # Interacts with Ollama language models
│   │   └── chromadb_client.py # Manages ChromaDB connections
│   └── ui
│       └── streamlit_ui.py   # Streamlit user interface code
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── .env                       # Environment variables
```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-chatbot
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the root directory and add your API keys and configuration settings.

5. **Run the application**:
   ```bash
   streamlit run src/app.py
   ```

## Features

- **User Interface**: A simple and interactive UI built with Streamlit.
- **Retrieval-Augmented Generation**: Combines retrieval of relevant information with language generation for enhanced responses.
- **Ollama Integration**: Utilizes Ollama's language models for generating responses.
- **ChromaDB**: A local vector database for efficient storage and retrieval of embeddings.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.