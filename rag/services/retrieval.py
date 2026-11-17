
import re

from pgvector.django import CosineDistance
from django.contrib.postgres.search import (
    SearchVector,
    SearchQuery,
    SearchRank,
)

from rag.models import DocumentChunk
from rag.services.embedding import embed_query
from rag.services.llm import (
    generate_queries,
    reciprocal_rank_fusion_multiple,
)
from rag.services.reranker import rerank_results


def merge_search_results(rankings, score_field, descending=False):
    """
    Merge duplicate chunks across multiple search rankings.
    Keep the best score for each chunk.
    """
    merged = {}

    for ranking in rankings:
        for result in ranking:
            chunk_id = result["id"]

            if chunk_id not in merged:
                merged[chunk_id] = dict(result)
                continue

            current_score = merged[chunk_id].get(score_field)
            new_score = result.get(score_field)

            if current_score is None or new_score is None:
                continue

            if descending:
                merged[chunk_id][score_field] = max(
                    current_score, new_score
                )
            else:
                merged[chunk_id][score_field] = min(
                    current_score, new_score
                )

    return sorted(
        merged.values(),
        key=lambda item: item.get(score_field, 0),
        reverse=descending,
    )


def retrieve_chunks(query, company, department, top_k=5):

    # 1. Generate original + alternative queries
    queries = generate_queries(query)

    # Remove duplicate queries while preserving order
    queries = list(dict.fromkeys(
        q.strip() for q in queries if q and q.strip()
    ))

    # Common tenant and department filters
    base_queryset = DocumentChunk.objects.filter(
        company=company,
        metadata__department=department,
    )

    vector_rankings = []
    keyword_rankings = []

    # 2. Run vector and keyword search for every query
    for search_query_text in queries:

        query_embedding = embed_query(search_query_text)

        # Vector search
        vector_chunks = (
            base_queryset
            .filter(embedding__isnull=False)
            .annotate(
                distance=CosineDistance("embedding", query_embedding)
            )
            .order_by("distance")[:10]
        )

        vector_results = [
            {
                "id": chunk.id,
                "parent_id": chunk.parent_id,
                "source_filename": chunk.source_filename,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "distance": float(chunk.distance),
            }
            for chunk in vector_chunks
        ]

        vector_rankings.append(vector_results)

        # Keyword search
        search_vector = SearchVector("content", config="english")
        search_terms = re.findall(r"\w+", search_query_text)

        keyword_results = []

        if search_terms:
            search_query = SearchQuery(
                search_terms[0],
                config="english",
            )

            for term in search_terms[1:]:
                search_query |= SearchQuery(
                    term,
                    config="english",
                )

            keyword_chunks = (
                base_queryset
                .annotate(
                    search=search_vector,
                    rank=SearchRank(search_vector, search_query),
                )
                .filter(search=search_query)
                .order_by("-rank")[:10]
            )

            keyword_results = [
                {
                    "id": chunk.id,
                    "parent_id": chunk.parent_id,
                    "source_filename": chunk.source_filename,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "keyword_score": float(chunk.rank),
                }
                for chunk in keyword_chunks
            ]

        keyword_rankings.append(keyword_results)

    # 3. Merge results for inspection/evaluation
    vector_results = merge_search_results(
        vector_rankings,
        score_field="distance",
    )

    keyword_results = merge_search_results(
        keyword_rankings,
        score_field="keyword_score",
        descending=True,
    )

    # 4. RRF across all vector and keyword rankings
    all_rankings = vector_rankings + keyword_rankings

    hybrid_results = reciprocal_rank_fusion_multiple(
        all_rankings
    )

    # 5. Cross-encoder reranking using original question
    reranked_results = rerank_results(
        query,
        hybrid_results,
        top_k=top_k,
    )

    return {
        "vector_results": vector_results,
        "keyword_results": keyword_results,
        "hybrid_results": hybrid_results,
        "reranked_results": reranked_results,
    }
