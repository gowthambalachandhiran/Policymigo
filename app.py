import streamlit as st
from vector_store import VectorStore
from chat_bot import ChatBot

def main():
    vector_store = VectorStore()
    vector_store.create_or_load_vectorstore()
    chat_bot = ChatBot(vector_store.vectorstore)
    chat_bot.run_chat_interface()

if __name__ == "__main__":
    main()
