import pandas as pd
import os
import time
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy, Faithfulness, FactualCorrectness
import ragas

from models import Models
model = Models()
llm = model.model_llama_3_1
general_embedding = model.embedding_paraphrase

# Load the combined evaluation results from retrieval stage
evaluation_df = pd.read_csv("evaluation/evaluation_dataset.csv")

# Prepare LLM and embedding wrappers for evaluation
evaluator_llm = LangchainLLMWrapper(langchain_llm=llm)
evaluator_embedding = LangchainEmbeddingsWrapper(embeddings=general_embedding)

# Define RAGAS metrics to use
metrics = [
    ResponseRelevancy(),
    Faithfulness(),
    LLMContextRecall(),
    LLMContextPrecisionWithReference(),
    FactualCorrectness()
]

def evaluate_dataset(evaluation_dataset, evaluator_llm, evaluator_embedding, version="v1"):
    """
    Evaluate a dataset using RAGAS metrics and produce detailed and summary results.
    """
    results_df = evaluation_dataset.copy()
    # Initialize metric score columns
    results_df["answer_relevancy"] = None
    results_df["faithfulness"] = None
    results_df["context_recall"] = None
    results_df["context_precision"] = None
    results_df["f1_score"] = None
    results_df["duration"] = None

    # Evaluate each Q&A pair
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
    # Compute average metrics per experiment
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
    # Save detailed and summary results to CSV
    os.makedirs("results", exist_ok=True)
    results_filename = os.path.join("results", f"evaluation_results_{version}.csv")
    summary_filename = os.path.join("results", f"evaluation_summary_{version}.csv")
    results_df.to_csv(results_filename, index=False)
    summary_df.to_csv(summary_filename, index=False)
    print("✅ Evaluation complete!!")
    return results_df, summary_df

# Run evaluation on the combined dataset (version "v1")
all_experiments_results_df, all_experiments_summary_df = evaluate_dataset(evaluation_df, evaluator_llm, evaluator_embedding, version="v1")