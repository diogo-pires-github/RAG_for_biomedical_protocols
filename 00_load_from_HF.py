import pickle
from collections import defaultdict
from datasets import load_from_disk

# If the data is in HuggingFace Dataset format
hf_dataset = load_from_disk("data/AP_protocols")

# Convert documents to the original PyMuPDF format (or use parsed_documents.pkl)
try:
    from langchain.schema import Document
except ImportError:
    from langchain.docstore.document import Document  # older LangChain

loaded_documents_flat = [
    Document(
        page_content=row["text"],
        metadata={k: v for k, v in row.items() if k != "text"}
    )
    for row in hf_dataset  # datasets.Dataset is row-iterable
]

possible_keys = ["source", "file_path", "path", "filename", "doc_id"]
group_key = next((k for k in possible_keys if k in hf_dataset.column_names), None)

if group_key is not None:
    buckets = defaultdict(list)
    for row in hf_dataset:
        key = row.get(group_key, "__ungrouped__")
        buckets[key].append(
            Document(
                page_content=row["text"],
                metadata={k: v for k, v in row.items() if k != "text"}
            )
        )
    loaded_documents = list(buckets.values())  # List[List[Document]]
else:
    # Fallback
    loaded_documents = [loaded_documents_flat]

# Save
with open("./data/processed/parsed_documents.pkl", "wb") as f:
    pickle.dump(loaded_documents, f)

print("Documents saved to parsed_documents.pkl") 