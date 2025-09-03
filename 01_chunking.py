import os
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from uuid import uuid4
import pickle
import time

from models import Models
model = Models()
llm = model.model_llama_3_1
general_embedding = model.embedding_paraphrase

# Initialize Chroma vector stores for each chunking experiment
vector_store_small_chunks = Chroma(
    collection_name="fixed_256_chunks",
    embedding_function=general_embedding,
    persist_directory="./db/chroma_256_chunks_db",
    collection_metadata={"hnsw:space": "cosine"}
)
vector_store_big_chunks = Chroma(
    collection_name="fixed_512_chunks",
    embedding_function=general_embedding,
    persist_directory="./db/chroma_512_chunks_db",
    collection_metadata={"hnsw:space": "cosine"}
)
vector_store_semantic_chunks = Chroma(
    collection_name="semantic_chunks",
    embedding_function=general_embedding,
    persist_directory="./db/chroma_semantic_chunks_db",
    collection_metadata={"hnsw:space": "cosine"}
)

# Chunking method 1: Smaller chunks (256 tokens)
small_chunk_size = 256
small_chunk_overlap = 64
separators = ["\n\n", "\n", " ", "."]

small_chunks_256 = []
start_time = time.time()
# Load preprocessed documents
with open("./data/processed/parsed_documents.pkl", "rb") as f:
    documents = pickle.load(f)

for doc in documents:
    small_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=small_chunk_size,
        chunk_overlap=small_chunk_overlap,
        separators=separators
    )
    small_chunks = small_text_splitter.split_documents(doc)
    # Normalize whitespace in each chunk
    for chunk in small_chunks:
        chunk.page_content = " ".join(chunk.page_content.split())
    small_chunks_256.extend(small_chunks)
    uuids = [str(uuid4()) for _ in range(len(small_chunks))]
    vector_store_small_chunks.add_documents(documents=small_chunks, ids=uuids)

end_time = time.time()
print(f"✂️ {len(documents)} documents split into {len(small_chunks_256)} small chunks in {(end_time - start_time) / 60:.2f} minutes")

# Save small chunks to file
os.makedirs("chunks", exist_ok=True)
with open("chunks/small_chunks_256.pkl", "wb") as f:
    pickle.dump(small_chunks_256, f)
print("Small chunks saved!")

# Chunking method 2: Bigger chunks (512 tokens)
big_chunk_size = 512
big_chunk_overlap = 128

big_chunks_512 = []
start_time = time.time()

for doc in documents:
    big_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=big_chunk_size,
        chunk_overlap=big_chunk_overlap,
        separators=separators
    )
    big_chunks = big_text_splitter.split_documents(doc)
    for chunk in big_chunks:
        chunk.page_content = " ".join(chunk.page_content.split())
    big_chunks_512.extend(big_chunks)
    uuids = [str(uuid4()) for _ in range(len(big_chunks))]
    vector_store_big_chunks.add_documents(documents=big_chunks, ids=uuids)

end_time = time.time()
print(f"✂️ {len(documents)} documents split into {len(big_chunks_512)} big chunks in {(end_time - start_time) / 60:.2f} minutes")

# Save big chunks to file
with open("chunks/big_chunks_512.pkl", "wb") as f:
    pickle.dump(big_chunks_512, f)
print("Big chunks saved!")

# Chunking method 3: Semantic chunking
semantic_text_splitter = SemanticChunker(
    general_embedding,
    breakpoint_threshold_type="gradient",
    min_chunk_size=128
)

semantic_chunks = []
doc_counter = 1
start_time = time.time()

for doc in documents:
    for page in doc:
        # Split each page into semantic chunks
        semantic_list = semantic_text_splitter.create_documents([page.page_content])
        for chunk in semantic_list:
            chunk.page_content = " ".join(chunk.page_content.split())
        semantic_chunks.extend(semantic_list)
        uuids = [str(uuid4()) for _ in range(len(semantic_chunks))]
        vector_store_semantic_chunks.add_documents(documents=semantic_chunks, ids=uuids)
    print(f"💾 Document {doc_counter} stored in the vector store")
    doc_counter += 1

end_time = time.time()
print(f"✂️ {len(documents)} documents split into {len(semantic_chunks)} semantic documents in {(end_time - start_time) / 60:.2f} minutes")

# Save semantic chunks to file
with open("chunks/semantic_chunks.pkl", "wb") as f:
    pickle.dump(semantic_chunks, f)
print("Semantic chunks saved!")