# RAG Assistant for Anatomical Pathology Laboratories

This repository contains the reference implementation of a **Retrieval-Augmented Generation (RAG) assistant for Anatomical Pathology (AP) laboratories**, as presented in the accompanying research paper:

**Pires, D., Perezhohin, Y., & Castelli, M. (2025).**
*Retrieval-Augmented Generation Assistant for Anatomical Pathology Laboratories.*
Emerging Science Journal, 9(6).
**DOI:** [https://doi.org/10.28991/ESJ-2025-09-06-013](https://doi.org/10.28991/ESJ-2025-09-06-013)

---

## Motivation

In Anatomical Pathology laboratories, **up to 70% of medical decisions depend on laboratory diagnoses**, yet technicians often rely on fragmented, outdated, and hard-to-search documentation. This creates inefficiencies and increases the risk of procedural errors.

This project addresses that challenge by:

* Grounding a **Large Language Model (LLM)** in official laboratory protocols
* Using **Retrieval-Augmented Generation (RAG)** to reduce hallucinations
* Providing **traceable, protocol-backed answers** to technician queries

---

## Key Features

* **Anatomical Pathology Knowledge Base**

  * 99 real AP laboratory protocols from a Portuguese healthcare institution
  * 323 protocol-derived question–answer pairs for systematic evaluation
  * Publicly available in Hugging Face Datasets format

* **Retrieval-Augmented Question Answering**

  * Retrieves relevant protocol chunks and conditions the LLM on them
  * Ensures answers are faithful to official laboratory procedures

* **Multiple Retrieval Strategies**

  * Dense semantic retrieval
  * Reranking-based retrieval
  * Hybrid search (semantic + keyword / BM25)

* **Reproducible Evaluation Pipeline**

  * Uses the **RAGAS framework** to evaluate:

    * Answer Relevance
    * Faithfulness (hallucination control)
    * Context Recall
  * Includes deterministic **Top-k retrieval evaluation**

* **Interactive Interface**

  * Streamlit-based chat application for real-time querying of protocols

---

## Repository Structure

```
├── data/
│   ├── AP_protocols/                     # Corpus in HuggingFace Datasets format
│   │   ├── data-00000-of-00001.arrow
│   │   ├── dataset_info.json
│   │   └── state.json
│   ├── processed/
│   │   └── parsed_documents.pkl          # Preprocessed corpus (recommended entry point)
│
├── evaluation/
│   └── QA_test_dataset.csv               # Ground-truth QA pairs
│
├── 00_load_from_HF.py                    # Load dataset directly from Hugging Face
├── 01_chunking.py                        # Chunking + vector store creation
├── 02_retrieval.py                       # Context retrieval + answer generation
├── 03_evaluation.py                      # RAGAS-based evaluation
├── 04_biomedical_experiment.py           # Biomedical embedding experiment
├── 05_top_k_evaluation.py                # Top-k retrieval evaluation
│
├── app.py                                # Streamlit chat application
├── models.py                             # LLM and embedding model definitions
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

**Requirements**

* Python **3.10+**
* GPU strongly recommended for local LLM inference (tested with Llama 3.1 8B)
* Ollama installed for local model serving

---

## Usage Guide

### 1️. Data Preparation

Choose **one** of the following:

* **Option A (Recommended)**
  Use the preprocessed corpus:

  ```bash
  python 01_chunking.py
  ```

* **Option B**
  Load the dataset directly from Hugging Face:

  ```bash
  python 00_load_from_HF.py
  ```

---

### 2️. Run the RAG Pipeline

```bash
python 01_chunking.py
python 02_retrieval.py
python 03_evaluation.py
```

This will:

* Build the vector store
* Retrieve contexts for all test questions
* Generate answers using the LLM
* Evaluate results using RAGAS metrics

---

### 3️. Biomedical Embedding Experiment (Optional)

Reproduce the paper’s biomedical embedding experiment:

```bash
python 04_biomedical_experiment.py
```

---

### 4️. Top-k Retrieval Evaluation (Optional)

```bash
python 05_top_k_evaluation.py
```

---

### 5️. Run the Interactive Chat App (Optional)

```bash
streamlit run app.py
```

This launches a local web interface for querying AP protocols in natural language.

---

## Reproducibility

This repository allows full reproduction of the experiments reported in the paper, including:

* Chunking strategy comparisons
* Retrieval method comparisons
* General vs. biomedical embedding models
* RAGAS and Top-k evaluation metrics

All parameters and configurations are explicitly defined in the code.

---

## Dataset

* **AP Laboratory Protocols Dataset**
  👉 [https://huggingface.co/datasets/diogofmp/AP_Lab_Protocols](https://huggingface.co/datasets/diogofmp/AP_Lab_Protocols)


## Reference

If you use this repository, please cite the paper:

```bibtex
@article{pires2025rag_ap,
  title   = {Retrieval-Augmented Generation Assistant for Anatomical Pathology Laboratories},
  author  = {Pires, Diogo and Perezhohin, Yuriy and Castelli, Mauro},
  journal = {Emerging Science Journal},
  volume  = {9},
  number  = {6},
  year    = {2025},
  doi     = {10.28991/ESJ-2025-09-06-013}
}
```

📄 **Paper link:** [https://doi.org/10.28991/ESJ-2025-09-06-013](https://doi.org/10.28991/ESJ-2025-09-06-013)

