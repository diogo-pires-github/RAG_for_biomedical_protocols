# RAG Assistant for Anatomical Pathology Labs

## Project Overview
This repository implements a Retrieval-Augmented Generation (RAG) pipeline for Anatomical Pathology (AP) laboratory protocols, as introduced in the accompanying research article. The goal is to turn static AP protocol documents into a dynamic, question-answering assistant that helps lab technicians quickly find accurate, context-grounded answers to protocol-related queries. In AP labs, up to 70% of medical decisions depend on lab diagnoses, yet protocols are often stored in cumbersome PDFs or binders. This project addresses that gap by using RAG to combine a Large Language Model (LLM) with a curated protocol knowledge base, enabling precise, reliable answers drawn from official lab procedures.

## Key Features:

**AP Protocols Knowledge Base:** A custom dataset of 99 AP lab protocols (from a Portuguese healthcare institution) and 323 Q&A pairs for evaluation, available on Hugging Face (see Dataset below). This forms the knowledge source that the RAG assistant retrieves from.

**Retrieval-Augmented QA:** The system retrieves relevant protocol chunks and feeds them to an LLM, which generates answers grounded in the retrieved text. This ensures answers remain faithful to official procedures.

**Multiple Retrieval Strategies:** We implement three retrieval methods – dense embedding-based search, a reranking retrieval approach, and hybrid search (combining semantic and keyword-based retrieval) – to evaluate which best suits AP protocols.

**Evaluation with RAGAS:** The quality of answers is measured with the RAGAS framework, using metrics like Answer Relevance, Faithfulness, and Context Recall. The pipeline can reproduce the experiments from the paper, demonstrating how different chunking strategies, retrieval modes, and embedding models impact these metrics.

## Repo structure
```
├── data/
│   ├── AP_protocols/                     # Corpus in HuggingFace Datasets format
|   |   ├── data-00000-of-00001.arrow
|   |   ├── dataset_info.json
│   │   └── state.json
│   ├── processed/
│   │   └── parsed_documents.pkl          # Processed corpus (list format)
├── evaluation/
│   └── QA_test_dataset.csv               # Evaluation QA pairs
├── 00_load_from_HF.py                    # For HuggingFace format
├── 01_chunking.py                        # Create chunks and store them in a vector store
├── 02_retrieval.py                       # Retrieves contexts and answers the test QA
├── 03_evaluation.py                      # Evaluates the generated answers using RAGAS
├── 04_biomedical_experiment.py           # RAG pipeline + evaluation using biomedical-specific model
├── 05_top_k_evaluation.py                # Top-k evaluation
├── app.py                                # Streamlit chat application
├── models.py                             # LLM and embedding models
└── requirements.txt
```
## Installation
```
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```
Python 3.10+ recommended. GPU is advised for LLM inference.

## Usage Guide
#### 1. Prepare data
   - Option A: Use the `parsed_documents.pkl` data file and start with `01_chunking.py`. (recommended)
   - Option B: Use the data directly from Hugging Face and start with `00_load_from_HF.py`.
#### 2. Run pipeline
```
python 01_chunking.py
python 02_retrieval.py
python 03_evaluation.py
```
#### 3. Biomedical embeddings (optional)
```
python biomedical_experiment.py
```
#### 4. Streamlit app (optional)
```
streamlit run app.py
```

## References
Pires, D., Perezhohin, Y., & Castelli, M. (2025). *RAG Assistant for Anatomical Pathology Laboratories*.
Preprint in Artificial Intelligence in Medicine.

**Dataset:** [AP_Lab_Protocols](https://huggingface.co/datasets/diogofmp/AP_Lab_Protocols) on Hugging Face



