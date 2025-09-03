import re
import unicodedata
import pickle
import pandas as pd
import time

from langchain_chroma import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from models import Models
model = Models()

llm = model.model_llama_3_1
biomedical_embedding_small = model.embedding_medembed_small

# Load test dataset
test_dataset = pd.read_csv("evaluation/QA_test_dataset.csv")

# Load biomedical chunks
with open("chunks/bio_big_chunks_512.pkl", "rb") as f:
    bio_big_chunks_512 = pickle.load(f)

# Biomedical vector store
vector_store_biomedical = Chroma(
    collection_name="biomedical_embeddings",
    embedding_function=biomedical_embedding_small,
    persist_directory="./db/chroma_biomedical_db",
    collection_metadata={"hnsw:space": "cosine"}
)

# Hybrid Search retrieval
def hybrid_search(vector_store, chunks, k):
    similarity_retreiver = vector_store.as_retriever(search_kwargs={"k": k})
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k =  k
    
    hybrid_search_retriever = EnsembleRetriever(
        retrievers=[similarity_retreiver, keyword_retriever],
        weights=[0.3, 0.7]
    )
    
    return hybrid_search_retriever


# RAG Pipeline (same as before)
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

# Define k values for evaluation
k_values = [1, 2, 4, 8]
all_experiments_dfs = []

for k in k_values:
    print(f"Starting experiment for k = {k}")
    top_k_evaluation_df = rag_pipeline(
        experiment_name=f"k={k}", 
        test_dataset=test_dataset, 
        retriever=hybrid_search(vector_store_biomedical, bio_big_chunks_512, k=k)
    )
    all_experiments_dfs.append(top_k_evaluation_df)
    print(f"✅ Completed experiment for k = {k}")

top_k_biomedical_hybrid_search_512_results = pd.concat(all_experiments_dfs, ignore_index=True)
top_k_biomedical_hybrid_search_512_results.to_csv("evaluation/top_k_evaluation_dataset.csv", index=False)

# Top-k evaluation dataset
top_k_evaluation_df = top_k_biomedical_hybrid_search_512_results.copy()

# Top-k evaluation
def parse_list(x):
    if isinstance(x, str):
        return re.findall(r"'([^']*)'", x)
    elif isinstance(x, list):
        return x
    else:
        return []

def normalize_text(s: str) -> str:
    """Lowercase, decompose unicode, and strip all diacritic marks."""
    s = s.lower()
    s = unicodedata.normalize('NFD', s)
    return ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')

def has_long_ngram_match(ref_norm: str, doc_norm: str, n: int = 10) -> bool:
    tokens = ref_norm.split()
    if len(tokens) < n:
        return False
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i : i + n])
        if ngram in doc_norm:
            return True
    return False

def precision_recall_and_relevant(refs, retrieved, ngram=10):
    # 1) Normalize all refs
    refs_norm = [normalize_text(r) for r in refs]
    
    # 2) Track which refs get matched at least once
    matched_ref_idxs = set()
    relevant_docs = []
    
    # 3) For each retrieved doc, decide if it’s relevant
    for doc in retrieved:
        doc_norm = normalize_text(doc)
        is_relevant = False
        
        # a) exact‐substring match
        for idx, r in enumerate(refs_norm):
            if r in doc_norm:
                matched_ref_idxs.add(idx)
                is_relevant = True
        
        # b) n‐gram fallback
        if not is_relevant:
            for idx, r in enumerate(refs_norm):
                if has_long_ngram_match(r, doc_norm, ngram):
                    matched_ref_idxs.add(idx)
                    is_relevant = True
                    break
        
        if is_relevant:
            relevant_docs.append(doc)
    
    # 4) Compute metrics
    num_retrieved = len(retrieved)            # K
    num_refs = len(refs_norm)                 # Number of reference contexts
    num_matched_docs = len(relevant_docs)     # Number of relevant documents retrieved
    num_matched_refs = len(matched_ref_idxs)  # Number of matched references
    
    precision = num_matched_docs / num_retrieved if num_retrieved else 0.0
    recall = num_matched_refs / num_refs if num_refs else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1_score, relevant_docs

# Evaluate
top_k_evaluation_df['reference_contexts'] = top_k_evaluation_df['reference_contexts'].apply(parse_list)
top_k_evaluation_df['retrieved_contexts'] = top_k_evaluation_df['retrieved_contexts'].apply(parse_list)

top_k_evaluation_df[['precision_at_k', 'recall_at_k', 'f1_score', 'relevant_docs']] = top_k_evaluation_df.apply(
    lambda row: precision_recall_and_relevant(
        row['reference_contexts'],
        row['retrieved_contexts'],
        ngram = 5
    ),
    axis=1,
    result_type='expand'
)

top_k_evaluation_df_summary = top_k_evaluation_df[['experiment', 'precision_at_k', 'recall_at_k', 'f1_score']].groupby('experiment').mean().round(2)

top_k_evaluation_df.to_csv("top_k_results/top_k_evaluation.csv", index=False)
top_k_evaluation_df_summary.to_csv("top_k_results/top_k_evaluation_summary.csv", index=False)