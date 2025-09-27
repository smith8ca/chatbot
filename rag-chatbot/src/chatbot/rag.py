class RAGChatbot:
    def __init__(self, ollama_client, chromadb_client):
        self.ollama_client = ollama_client
        self.chromadb_client = chromadb_client

    def process_query(self, user_query):
        # Retrieve relevant information from ChromaDB
        relevant_info = self.chromadb_client.retrieve_vector(user_query)
        
        # Generate a response using the Ollama language model
        response = self.ollama_client.generate_response(user_query, relevant_info)
        
        return response

    def store_information(self, information):
        # Store the information in ChromaDB for future retrieval
        self.chromadb_client.store_vector(information)