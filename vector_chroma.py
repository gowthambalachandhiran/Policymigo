# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:17:52 2024

@author: gowtham.balachan
"""

import os
import shutil
import uuid
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time

# Generate a unique directory name
CHROMA_PATH = f"chroma_{uuid.uuid4()}"

class VectorStore:
    def __init__(self, pdf_directory="PDFs for DB"):
        self.pdf_directory = pdf_directory
        
        # Initialize Google's embedding model
        self.embed_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # Initialize Chroma database
        self.vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embed_model)

    def create_or_load_vectorstore(self):
        # Clear out the existing database directory if it exists
        self._clear_chroma_path()

        # Load PDF documents and split into chunks
        pdf_loader = DirectoryLoader(self.pdf_directory, glob="*.pdf", loader_cls=PyPDFLoader)
        pdf_documents = pdf_loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        docs = splitter.split_documents(pdf_documents)
        
        # Add documents to the vector store
        self.vectorstore.add_documents(docs)
        self.vectorstore.persist()
        print(f"Saved {len(docs)} chunks to {CHROMA_PATH}.")

    def _clear_chroma_path(self):
        # Attempt to delete the directory
        if os.path.exists(CHROMA_PATH):
            retries = 3
            while retries > 0:
                try:
                    shutil.rmtree(CHROMA_PATH)
                    print(f"Cleared Chroma path: {CHROMA_PATH}")  # Debug information
                    break
                except PermissionError as e:
                    print(f"PermissionError: {e}, retrying...")
                    time.sleep(1)  # Wait before retrying
                    retries -= 1
                except Exception as e:
                    print(f"Error deleting directory: {e}")
                    break

    def get_retriever(self):
        return self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
