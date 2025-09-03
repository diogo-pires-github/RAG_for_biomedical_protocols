import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import pickle
import time

from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.retrievers import BM25Retriever, EnsembleRetriever

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy, Faithfulness, FactualCorrectness
import ragas

from models import Models
model = Models()
llm = model.model_llama_3_1
biomedical_embedding_small = model.embedding_medembed_small

# Load preprocessed documents from file
with open("./data/processed/parsed_documents.pkl", "rb") as f:
    loaded_documents = pickle.load(f)

# Initialize a Chroma vector store for biomedical embeddings
vector_store_biomedical = Chroma(
    collection_name="biomedical_embeddings",
    embedding_function=biomedical_embedding_small,
    persist_directory="./db/chroma_biomedical_db",
    collection_metadata={"hnsw:space": "cosine"}
)

# Chunk the documents using the biomedical embedding (512 token chunks)
big_chunk_size = 512
big_chunk_overlap = 128
separators = ["\n\n", "\n", " ", "."]

bio_big_chunks_512 = []
start_time = time.time()
for doc in loaded_documents:
    big_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=big_chunk_size,
        chunk_overlap=big_chunk_overlap,
        separators=separators
    )
    bio_big_chunks = big_text_splitter.split_documents(doc)
    for chunk in bio_big_chunks:
        chunk.page_content = " ".join(chunk.page_content.split())
    bio_big_chunks_512.extend(bio_big_chunks)
    uuids = [str(uuid4()) for _ in range(len(bio_big_chunks))]
    vector_store_biomedical.add_documents(documents=bio_big_chunks, ids=uuids)
end_time = time.time()
print(f"✂️ {len(loaded_documents)} documents split into {len(bio_big_chunks_512)} big chunks in {(end_time - start_time) / 60:.2f} minutes")

# Save biomedical chunks to file
os.makedirs("chunks", exist_ok=True)
with open("chunks/bio_big_chunks_512.pkl", "wb") as f:
    pickle.dump(bio_big_chunks_512, f)
print("Documents saved to bio_big_chunks_512.pkl")

# Load the evaluation questions dataset
test_dataset = pd.read_csv("evaluation/QA_test_dataset.csv")

# Prepare prompt and retrieval chain for answer generation
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

# RAG pipeline
def rag_pipeline(experiment_name, test_dataset, retriever):
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

# Perform retrieval using hybrid search on biomedical chunks
biomedical_hybrid_search_big_chunks_results = rag_pipeline(
    experiment_name="biomedical_hybrid_search_big_chunks",
    test_dataset=test_dataset,
    retriever=hybrid_search(vector_store_biomedical, bio_big_chunks_512)
)

# Save the biomedical experiment results
os.makedirs("evaluation", exist_ok=True)
biomedical_hybrid_search_big_chunks_results.to_csv("evaluation/biomedical_evaluation_dataset.csv", index=False)

# Evaluate the biomedical experiment results with RAGAS metrics
evaluator_llm = LangchainLLMWrapper(langchain_llm=llm)
evaluator_embedding = LangchainEmbeddingsWrapper(embeddings=biomedical_embedding_small)
metrics = [
    ResponseRelevancy(),
    Faithfulness(),
    LLMContextRecall(),
    LLMContextPrecisionWithReference(),
    FactualCorrectness()
]

results_df = biomedical_hybrid_search_big_chunks_results.copy()
results_df["answer_relevancy"] = None
results_df["faithfulness"] = None
results_df["context_recall"] = None
results_df["context_precision"] = None
results_df["f1_score"] = None
results_df["duration"] = None

for idx, record in results_df.iterrows():
    start = time.time()
    sample_ds = ragas.Dataset.from_pandas(pd.DataFrame([record]))
    result = ragas.evaluate(
        sample_ds,
        llm=evaluator_llm,
        embeddings=evaluator_embedding,
        metrics=metrics,
        raise_exceptions=False
    )
    scores = result.to_pandas().iloc[0]
    results_df.at[idx, "answer_relevancy"] = scores["answer_relevancy"]
    results_df.at[idx, "faithfulness"] = scores["faithfulness"]
    results_df.at[idx, "context_recall"] = scores["context_recall"]
    results_df.at[idx, "context_precision"] = scores["llm_context_precision_with_reference"]
    results_df.at[idx, "f1_score"] = scores["factual_correctness"]
    results_df.at[idx, "duration"] = time.time() - start

    current_exp = results_df.loc[idx, "experiment"]
    if idx == len(results_df) - 1 or results_df.loc[idx + 1, "experiment"] != current_exp:
        print(f"🎯 Completed evaluation for experiment: {current_exp}")

results_df = results_df.fillna(0)
summary_df = results_df.groupby("experiment").agg({
    "answer_relevancy": "mean",
    "faithfulness": "mean",
    "context_recall": "mean",
    "context_precision": "mean",
    "f1_score": "mean",
    "duration": "sum"
}).reset_index()
summary_df["total_duration_minutes"] = round(summary_df["duration"] / 60, 2)
summary_df["total_duration_hours"] = round(summary_df["total_duration_minutes"] / 60, 2)
summary_df.drop(columns=["duration"], inplace=True)

os.makedirs("results", exist_ok=True)
results_filename = os.path.join("results", "evaluation_results_biomedical_v1.csv")
summary_filename = os.path.join("results", "evaluation_summary_biomedical_v1.csv")
results_df.to_csv(results_filename, index=False)
summary_df.to_csv(summary_filename, index=False)
print("✅ Evaluation complete!!")
