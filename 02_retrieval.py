import os
import pandas as pd
import pickle
import time
from langchain_chroma import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from models import Models
model = Models()
llm = model.model_llama_3_1
general_embedding = model.embedding_paraphrase

# Load the test Q&A dataset for evaluation
test_dataset = pd.read_csv("evaluation/QA_test_dataset.csv")

# Reinitialize vector stores from persisted databases
vector_store_small_chunks = Chroma(
    collection_name="fixed_256_chunks",
    embedding_function=general_embedding,
    persist_directory="./db/chroma_256_chunks_db"
)
vector_store_big_chunks = Chroma(
    collection_name="fixed_512_chunks",
    embedding_function=general_embedding,
    persist_directory="./db/chroma_512_chunks_db"
)
vector_store_semantic_chunks = Chroma(
    collection_name="semantic_chunks",
    embedding_function=general_embedding,
    persist_directory="./db/chroma_semantic_chunks_db"
)

# Define the prompt for answer generation
SYSTEM_PROMPT = """
    You are a helpful AI assistant specialized in answering questions about Anatomical Pathology lab protocols and techniques.
    - Answer in portuguese (PT/PT).
    - Answer strictly based on the context provided.
    - Always reference the context when answering.
    - If no relevant context: apologize and say you don't have that information.
    - Use general knowledge only for basic definitions.
    - Keep your writing style simple and concise.
    - Keep your answers faithful to the context.
    - Answer directly.
    - Do not include warnings, notes, or unnecessary extras. Stick to the requested output.
    - Do not reveal your internal reasoning.

    Relevant Context:
    {context}

    User Question:
    {input}
    """
prompt = PromptTemplate(template=SYSTEM_PROMPT, input_variables=["context", "input"])

def rag_pipeline(experiment_name, test_dataset, retriever):
    # Create retrieval and answer generation chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    results = []
    start_time = time.time()
    for index, row in test_dataset.iterrows():
        query = row["Question"]
        reference_answer = row["Ground_Truth"]
        reference_contexts = row["Context"]
        answer = retrieval_chain.invoke({"input": query})
        results.append({
            "experiment": experiment_name,
            "user_input": query,
            "response": answer['answer'],
            "reference": reference_answer,
            "reference_contexts": [reference_contexts],
            "retrieved_contexts": [doc.page_content for doc in answer["context"]]
        })
    results_df = pd.DataFrame(results)
    end_time = time.time()
    print(f"✅ Generated {len(results_df)} answers in {(end_time - start_time) / 60:.2f} minutes")
    return results_df

# Naive
def naive_retriever(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever

# Reranking
def reranking_threshold_retriever(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.4})
    return retriever

# Hybrid search
def hybrid_search(vector_store, chunks):
    similarity_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 3
    hybrid_search_retriever = EnsembleRetriever(
        retrievers=[similarity_retriever, keyword_retriever],
        weights=[0.3, 0.7]
    )
    return hybrid_search_retriever

# Load chunk lists for hybrid search (BM25 requires direct access to documents)
with open("chunks/small_chunks_256.pkl", "rb") as f:
    small_chunks_256 = pickle.load(f)
with open("chunks/big_chunks_512.pkl", "rb") as f:
    big_chunks_512 = pickle.load(f)
with open("chunks/semantic_chunks.pkl", "rb") as f:
    semantic_chunks = pickle.load(f)

# Run retrieval experiments for each combination of chunking method and retriever

# 1. Naive retrieval (vector similarity) ----------------------------------------
print("Retrieving naive small chunks...")
naive_small_chunks_results = rag_pipeline(
    experiment_name="naive_small_chunks",
    test_dataset=test_dataset,
    retriever=naive_retriever(vector_store_small_chunks)
)

print("Retrieving naive big chunks...")
naive_big_chunks_results = rag_pipeline(
    experiment_name="naive_big_chunks",
    test_dataset=test_dataset,
    retriever=naive_retriever(vector_store_big_chunks)
)

print("Retrieving naive semantic chunks...")
naive_semantic_chunks_results = rag_pipeline(
    experiment_name="naive_semantic_chunks",
    test_dataset=test_dataset,
    retriever=naive_retriever(vector_store_semantic_chunks)
)

# 2. Reranking with similarity score threshold ---------------------------------
print("Retrieving reranking small chunks...")
reranking_threshold_small_chunks_results = rag_pipeline(
    experiment_name="reranking_threshold_small_chunks",
    test_dataset=test_dataset,
    retriever=reranking_threshold_retriever(vector_store_small_chunks)
)

print("Retrieving reranking big chunks...")
reranking_threshold_big_chunks_results = rag_pipeline(
    experiment_name="reranking_threshold_big_chunks",
    test_dataset=test_dataset,
    retriever=reranking_threshold_retriever(vector_store_big_chunks)
)

print("Retrieving reranking semantic chunks...")
reranking_threshold_semantic_chunks_results = rag_pipeline(
    experiment_name="reranking_threshold_semantic_chunks",
    test_dataset=test_dataset,
    retriever=reranking_threshold_retriever(vector_store_semantic_chunks)
)

# 3. Hybrid search (embedding + keyword search) -------------------------------
print("Retrieving hybrid search small chunks...")
hybrid_search_small_chunks_results = rag_pipeline(
    experiment_name="hybrid_search_small_chunks",
    test_dataset=test_dataset,
    retriever=hybrid_search(vector_store_small_chunks, small_chunks_256)
)

print("Retrieving hybrid search big chunks...")
hybrid_search_big_chunks_results = rag_pipeline(
    experiment_name="hybrid_search_big_chunks",
    test_dataset=test_dataset,
    retriever=hybrid_search(vector_store_big_chunks, big_chunks_512)
)

print("Retrieving hybrid search semantic chunks...")
hybrid_search_semantic_chunks_results = rag_pipeline(
    experiment_name="hybrid_search_semantic_chunks",
    test_dataset=test_dataset,
    retriever=hybrid_search(vector_store_semantic_chunks, semantic_chunks)
)

# Combine results from all experiments into one DataFrame ---------------------
evaluation_df = pd.concat(
    [
        # Naive retrieval
        naive_small_chunks_results,
        naive_big_chunks_results,
        naive_semantic_chunks_results,
        # Reranking retrieval
        reranking_threshold_small_chunks_results,
        reranking_threshold_big_chunks_results,
        reranking_threshold_semantic_chunks_results,
        # Hybrid search retrieval
        hybrid_search_small_chunks_results,
        hybrid_search_big_chunks_results,
        hybrid_search_semantic_chunks_results,
    ],
    ignore_index=True
)

# Save the combined evaluation dataset to CSV
os.makedirs("evaluation", exist_ok=True)
evaluation_df.to_csv("evaluation/evaluation_dataset.csv", index=False)
print(f"Combined evaluation dataset contains {len(evaluation_df)} QA pairs from all experiments.")