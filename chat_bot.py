import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, CombinedMemory
import time

class ChatBot:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", 
            temperature=0, 
            max_output_tokens=1024, 
            timeout=None, 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        self.memory = self.initialize_memory()

    def initialize_memory(self):
        human_memory = ConversationBufferMemory(input_key="human_input", memory_key="chat_history")
        context_memory = ConversationBufferMemory(input_key="context", memory_key="context_history")
        return CombinedMemory(memories=[human_memory, context_memory])

    def get_response(self, query):
        prompt_template = """
        You are a helpful underwriter. Using the context and chat history, answer questions regarding underwriter practice.
        
        Context History: {context_history}
        Chat History: {chat_history}
        Human: {human_input}
        Assistant: """
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context_history", "chat_history", "human_input"]
        )
        chain = LLMChain(llm=self.llm, prompt=PROMPT, memory=self.memory)
        
        try:
            # Use the retriever to get the documents
            retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
            docs = retriever.invoke(query)
            
            # Combine the document contents into a context string
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        except Exception as e:
            print(f"Error during retrieval: {e}")
            context = "No relevant context found due to an error in retrieval."
            docs = []
        
        # Invoke the chain to get the response
        result = chain.invoke({"context": context, "human_input": query})
        print(result)
        return result['text'], docs

    


    def run_chat_interface(self):
        st.markdown("""
            <div style='text-align: center;'>
                <h1>Welcome to PolicyPal</h1>
                <h3>Ask your questions regarding underwriting!!!!</h3>
            </div>
            """, unsafe_allow_html=True)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        query = st.chat_input("Say something: ")
        
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                typing_indicator = st.empty()
                typing_indicator.markdown("Bot is typing...")
                
                response, docs = self.get_response(query)
                
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                
                typing_indicator.empty()
                message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # with st.expander("View Reference Documents"):
            #     if docs:
            #         st.write("### Reference Documents Used:\n")
            #         for i, doc in enumerate(docs):
            #             st.write(f"**Document {i+1}:**")
            #             st.write(doc.page_content)
            #             st.write("---")
            #     else:
            #         st.write("No reference documents found.")