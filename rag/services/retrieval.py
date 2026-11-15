
from django.db.models import F
from pgvector.django import CosineDistance
from django.contrib.postgres.search import SearchVector, SearchQuery, SearchRank

from rag.models import DocumentChunk
from rag.services.embedding import embed_query
from rag.services.llm import reciprocal_rank_fusion
from rag.services.reranker import rerank_results
from django.contrib.postgres.search import (
    SearchVector,
    SearchQuery,
    SearchRank,
)
import re
def retrieve_chunks(query, company, department, top_k=5):

    # --------------------------------------------------
    # 1. Generate query embedding
    # --------------------------------------------------

    query_embedding = embed_query(query)

    # Common tenant and department filters
    base_queryset = DocumentChunk.objects.filter(
        company=company,
        metadata__department=department,
    )

    # --------------------------------------------------
    # 2. Vector Search
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 3. Keyword Search (PostgreSQL Full-Text Search)
    # --------------------------------------------------

    search_vector = SearchVector("content", config="english")

    search_terms = re.findall(r"\w+", query)

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

    # --------------------------------------------------
    # 4. Reciprocal Rank Fusion
    # --------------------------------------------------

    hybrid_results = reciprocal_rank_fusion(
        vector_results,
        keyword_results,
    )

    # --------------------------------------------------
    # 5. Cross-Encoder Reranking
    # --------------------------------------------------

    reranked_results = rerank_results(
        query,
        hybrid_results,
        top_k=top_k,
    )

    # Return individual stages for retrieval evaluation
    return {
        "vector_results": vector_results,
        "keyword_results": keyword_results,
        "hybrid_results": hybrid_results,
        "reranked_results": reranked_results,
    }
