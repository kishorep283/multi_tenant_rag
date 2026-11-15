def recall_at_k(relevant_ids, retrieved_ids, k):
    """
    Calculate Recall@K.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids[:k])
    intersection = relevant_set.intersection(retrieved_set)
    recall = len(intersection) / len(relevant_set) if relevant_set else 0.0
    return recall

def precision_at_k(relevant_ids, retrieved_ids, k):
    """
    Calculate Precision@K.
    """
    if k < 0:
        raise ValueError("k must be non-negative")

    if k == 0:
        return 0.0

    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    relevant_retrieved = sum(
        1 for chunk_id in top_k
        if chunk_id in relevant_set
    )

    precision = relevant_retrieved / k

    return precision

def reciprocal_rank(relevant_ids, retrieved_ids):
    """
    Calculate Reciprocal Rank.
    """
    relevant_set = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0

def mean_reciprocal_rank(evaluation_results):
    """
    Calculate Mean Reciprocal Rank across multiple queries.
    """
    if not evaluation_results:
        return 0.0

    total_reciprocal_rank = sum(
        reciprocal_rank(
            result["relevant_chunk_ids"],
            result["retrieved_chunk_ids"]
        )
        for result in evaluation_results
    )

    return total_reciprocal_rank / len(evaluation_results)

import math
def dcg_at_k(relevance_scores, k):
    """
    Calculate Discounted Cumulative Gain at K.
    """
    if k < 0:
        raise ValueError("k must be non-negative")

    dcg = 0.0
    for i in range(min(k, len(relevance_scores))):
        rel = relevance_scores[i]
        dcg += (2 ** rel - 1) / math.log2(i + 2)  # i + 2 because rank starts at 1

    return dcg

def idcg_at_k(relevance_scores, k):
    """
    Calculate Ideal Discounted Cumulative Gain at K.
    """
    if k < 0:
        raise ValueError("k must be non-negative")

    sorted_relevance = sorted(relevance_scores, reverse=True)
    return dcg_at_k(sorted_relevance, k)

def ndcg_at_k(relevance_scores, k, total_relevant):
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    """

    if k < 0:
        raise ValueError("k must be non-negative")

    dcg = dcg_at_k(relevance_scores, k)

    ideal_relevance = [1] * total_relevant

    idcg = idcg_at_k(ideal_relevance, k)

    return dcg / idcg if idcg > 0 else 0.0

print(ndcg_at_k([1, 0, 0], 3, 1))
print(ndcg_at_k([0, 0, 1], 3, 1))
print(ndcg_at_k([0, 0, 0], 3, 1))