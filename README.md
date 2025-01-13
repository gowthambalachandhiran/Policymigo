# PolicyPal: A RAG-based Underwriter Assistant

PolicyPal is a Streamlit application designed to assist underwriters by answering queries based on policy guidelines. Powered by a Retrieval-Augmented Generation (RAG) framework, the application leverages a FAISS vector database to retrieve context-relevant information, which is then processed by a large language model (LLM) to provide accurate and contextual answers.

---

## Features
- **Policy Query Resolution**: Provides answers to underwriter queries by referencing specific policy guidelines.
- **RAG Framework**: Combines retrieval of relevant documents with generative AI for accurate responses.
- **FAISS Vector Database**: Efficient and scalable retrieval of context from policy documents.
- **Streamlit Interface**: Intuitive UI for seamless interaction.

---

## How It Works
1. **Document Retrieval**: Policy documents are embedded and stored in a FAISS vector database.  
2. **Query Matching**: When an underwriter asks a question, relevant context is retrieved from the vector database.  
3. **Response Generation**: The LLM generates a response based on the retrieved context and query.  
4. **Interactive Output**: The response is displayed in the Streamlit app for the underwriter.  

---

## Prerequisites
- Python 3.8 or later
- Streamlit
- FAISS
- LLM (e.g., OpenAI GPT, Gemini)

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/gowthambalachandhiran/Policymigo.git
