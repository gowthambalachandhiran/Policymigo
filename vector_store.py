import os
import pickle
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import numpy as np

class VectorStore:
    def __init__(self, persist_directory="faiss_index", pdf_directory="PDFs for DB"):
        self.persist_directory = persist_directory
        self.pdf_directory = pdf_directory
        self.vectorstore = None
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

    def create_or_load_vectorstore(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        
        if self.database_exists():
            try:
                self.load_existing_vectorstore()
            except:
                print("Error loading existing vectorstore. Creating a new one.")
                self.create_new_vectorstore()
        else:
            self.create_new_vectorstore()
    
        print(f"Vectorstore created/loaded. Type: {type(self.vectorstore)}")

    def database_exists(self):
        return os.path.exists(os.path.join(self.persist_directory, "faiss_index.pkl"))

    def load_existing_vectorstore(self):
        with open(os.path.join(self.persist_directory, "faiss_index.pkl"), "rb") as f:
            vectorstore = pickle.load(f)
        
        if hasattr(vectorstore, 'embedding_function') and callable(vectorstore.embedding_function):
            self.vectorstore = vectorstore
        else:
            raise ValueError("Incompatible vectorstore format")

    def create_new_vectorstore(self):
        pdf_loader = DirectoryLoader(self.pdf_directory, glob="./*.pdf", loader_cls=PyPDFLoader)
        pdf_documents = pdf_loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(pdf_documents)
        
        texts = [doc.page_content for doc in docs]
        embeddings = self.get_embeddings(texts)
        
        self.vectorstore = FAISS.from_embeddings(
            list(zip(texts, embeddings)), 
            embedding=self.get_embeddings,
            normalize_L2=True
        )
        
        with open(os.path.join(self.persist_directory, "faiss_index.pkl"), "wb") as f:
            pickle.dump(self.vectorstore, f)
            
    def get_embeddings(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            embedding = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            print(f"Embedding shape: {len(embedding['embedding'])}")
            embeddings.append(np.array(embedding['embedding'], dtype=np.float32))
        return embeddings

    def get_retriever(self):
        return self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
