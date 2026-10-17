from sentence_transformers import CrossEncoder


reranker_model = CrossEncoder(
    "BAAI/bge-reranker-base"
)

def rerank_results(query, results, top_k=5):
    pairs = [
        [query, result["content"]]
        for result in results
    ]

    scores = reranker_model.predict(pairs)

    reranked_results = []

    for result, score in zip(results, scores):
        reranked_results.append({
            **result,
            "reranker_score": float(score),
        })

    reranked_results.sort(
        key=lambda result: result["reranker_score"],
        reverse=True,
    )

    return reranked_results[:top_k]