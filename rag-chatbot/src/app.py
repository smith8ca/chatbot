from streamlit import st
from chatbot.rag import RAGChatbot


def main():
    st.title("RAG Chatbot")
    st.write("Welcome to the Retrieval-Augmented Generation Chatbot!")

    chatbot = RAGChatbot()

    user_input = st.text_input("You: ", "")

    if st.button("Send"):
        if user_input:
            response = chatbot.process_query(user_input)
            st.text_area(
                "Chatbot:", value=response, height=200, max_chars=None, key=None
            )
        else:
            st.warning("Please enter a message.")


if __name__ == "__main__":
    main()
