from rag.models import Company
from rag.evaluation.dataset import EVALUATION_DATASET
from rag.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
)
from rag.services.retrieval import retrieve_chunks


def evaluate_stage(relevant_ids, retrieved_ids, k):
    """
    Evaluate one retrieval stage for a single query.
    """

    relevance_scores = [
        1 if chunk_id in relevant_ids else 0
        for chunk_id in retrieved_ids
    ]

    return {
        "precision": precision_at_k(relevant_ids, retrieved_ids, k),
        "recall": recall_at_k(relevant_ids, retrieved_ids, k),
        "reciprocal_rank": reciprocal_rank(relevant_ids, retrieved_ids),
        "ndcg": ndcg_at_k(
            relevance_scores,
            k,
            len(relevant_ids),
        ),
    }


def run_evaluation(dataset=None):

    if dataset is None:
        dataset = EVALUATION_DATASET

    evaluation_results = []

    for item in dataset:

        query = item["query"]
        company = Company.objects.get(id=item["company_id"])
        department = item["department"]
        relevant_ids = item["relevant_chunk_ids"]

        # Run the existing retrieval pipeline.
        results = retrieve_chunks(
            query=query,
            company=company,
            department=department,
            top_k=5,
        )

        # Extract chunk IDs from each stage.
        stages = {
            "vector": {
                "ids": [r["id"] for r in results["vector_results"]],
                "k": 10,
            },
            "keyword": {
                "ids": [r["id"] for r in results["keyword_results"]],
                "k": 10,
            },
            "hybrid": {
                "ids": [r["id"] for r in results["hybrid_results"]],
                "k": 10,
            },
            "reranked": {
                "ids": [r["id"] for r in results["reranked_results"]],
                "k": 5,
            },
        }

        stage_metrics = {}

        for stage_name, stage_data in stages.items():

            stage_metrics[stage_name] = evaluate_stage(
                relevant_ids=relevant_ids,
                retrieved_ids=stage_data["ids"],
                k=stage_data["k"],
            )

        evaluation_results.append({
            "query": query,
            "relevant_chunk_ids": relevant_ids,
            "retrieved_ids": {
                stage: stage_data["ids"]
                for stage, stage_data in stages.items()
            },
            "stage_metrics": stage_metrics,
        })

    return evaluation_results

def aggregate_metrics(evaluation_results):

    if not evaluation_results:
        return {}

    stages = ["vector", "keyword", "hybrid", "reranked"]

    metrics = [
        "precision",
        "recall",
        "reciprocal_rank",
        "ndcg",
    ]

    summary = {}

    for stage in stages:

        summary[stage] = {}

        for metric in metrics:

            values = [
                result["stage_metrics"][stage][metric]
                for result in evaluation_results
            ]

            summary[stage][metric] = sum(values) / len(values)

    return summary