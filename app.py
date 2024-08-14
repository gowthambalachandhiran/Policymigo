import streamlit as st
from vector_store import VectorStore
from chat_bot import ChatBot

def main():
    try:
        
        
        # Initialize the vector store and load/create the vectorstore
        vector_store = VectorStore()
        vector_store.create_or_load_vectorstore()
        
        # Initialize the chatbot and run the interface
        chat_bot = ChatBot(vector_store)
        chat_bot.run_chat_interface()
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()  # Stop the execution to avoid further errors

if __name__ == "__main__":
    main()
