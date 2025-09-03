import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from models import Models
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import pickle

# Building a streamlit app
st.set_page_config(page_title="APT Assistant", page_icon="🔬")
st.title("🔬 AP Protocol Assistant")

# Initialize models and vector store
@st.cache_resource
def initialize_system():
    models = Models()
    vector_store = Chroma(
    collection_name="biomedical_embeddings",
    embedding_function=models.embedding_medembed_small,
    persist_directory="./db/chroma_biomedical_db",
    collection_metadata={"hnsw:space": "cosine"}
    )
    return models, vector_store

models, vector_store = initialize_system()

# System prompt with proper chat history handling
SYSTEM_PROMPT = """
    You are a helpful AI assistant specialized in answering questions about Anatomical Pathology lab protocols and techniques.
    - Answer in Portuguese (PT/PT).
    - Answer strictly based on the context provided and chat history.
    - Always reference the context when answering.
    - If no relevant context: apologize and say you don't have that information.
    - Use general knowledge only for basic definitions.
    - Keep you writing style simple and concise.
    - Keep your answers faithful to the context.
    - Answer directly.
    - Do not include warnings, notes, or unnecessary extras. Stick to the requested output.
    - Do not reveal your internal reasoning.
    
    Chat History:
    {chat_history}

    Relevant Context:
    {context}

    User Question:
    {input}
    """

# LLM
llm = models.model_llama_3_1 

# Prompt
prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

# Chunks
with open("chunks/bio_big_chunks_512.pkl", "rb") as f:
    bio_big_chunks_512 = pickle.load(f)

# Hybrid Search Retriever
similarity_retreiver = vector_store.as_retriever(search_kwargs={"k": 3})
keyword_retriever = BM25Retriever.from_documents(bio_big_chunks_512)
keyword_retriever.k =  3
    
retriever = EnsembleRetriever(
    retrievers=[similarity_retreiver, keyword_retriever],
    weights=[0.3, 0.7]
)

# Chain setup with proper document handling
combine_docs_chain = create_stuff_documents_chain(
    llm,
    prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Chat UI
# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    role = "Human" if isinstance(message, HumanMessage) else "AI"
    with st.chat_message(role):
        st.markdown(message.content)

# Chat processing
if user_query := st.chat_input("Your message"):
    # Add user message to history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    # Display user message
    with st.chat_message("Human"):
        st.markdown(user_query)

    # Generate AI response
    with st.chat_message("AI"):
        response_container = st.empty()
        full_response = ""
        
        # Stream the response
        for chunk in retrieval_chain.stream({
            "input": user_query,
            "chat_history": [
                f"{msg.type}: {msg.content}" 
                for msg in st.session_state.chat_history
            ]
        }):
            full_response += chunk.get("answer", "")
            response_container.markdown(full_response + "▌")
        
        response_container.markdown(full_response)
    
    # Add AI response to history
    st.session_state.chat_history.append(AIMessage(content=full_response))
    
# streamlit run app.py