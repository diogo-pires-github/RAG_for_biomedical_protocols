from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

model_kwargs = {'device': 'cpu'}
# model_kwargs = {'device': 'cuda:0'}
encode_kwargs = {'normalize_embeddings': True}

class Models:
    def __init__(self):
        
        ### Emebeddings ###                
        self.embedding_paraphrase = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",   # size: 66M  | output: 384
            model_kwargs=model_kwargs,
			encode_kwargs=encode_kwargs
        )
        
        self.embedding_medembed_small = HuggingFaceEmbeddings(
            model_name="abhinand/MedEmbed-small-v0.1",                                  # size: 33M  | output: 384
			model_kwargs=model_kwargs,
			encode_kwargs=encode_kwargs
        )
        
        self.embedding_medembed_large = HuggingFaceEmbeddings(
            model_name="abhinand/MedEmbed-large-v0.1",                                  # size: 350M | output: 1024
			model_kwargs=model_kwargs,
			encode_kwargs=encode_kwargs
        )
        
        self.embedding_bge = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",                                                   # size: 330M | output: 1024
            model_kwargs=model_kwargs,
			encode_kwargs=encode_kwargs
        )
        
        ### LLMs ###
        self.model_llama_3_1 = ChatOllama(
            model="llama3.1:8b",
            seed=42,
            temperature=0.5
        )
        
        self.model_deepseek = ChatOllama(
            model="deepseek-r1:8b",
            seed=42
        )
        
        self.model_gemma = ChatOllama(
            model="gemma3:4b",
            seed=42,
            temperature=0.5
        )