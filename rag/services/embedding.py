import math

from langchain_huggingface import HuggingFaceEmbeddings

print(">>> BEFORE EMBEDDING MODEL")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
)
print(">>> AFTER EMBEDDING MODEL")

def embed_documents(documents):
    """Generate and validate embeddings for LangChain Documents."""

    if not documents:
        raise ValueError("Cannot generate embeddings for an empty document list.")

    texts = [document.page_content for document in documents]

    try:
        embeddings = embedding_model.embed_documents(texts)

        validate_embeddings(
            embeddings=embeddings,
            expected_count=len(texts),
        )

        return embeddings

    except Exception as e:
        raise RuntimeError("Document embedding generation failed.") from e

def embed_query(query: str):
    """Generate and validate an embedding for a query."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    try:
        embeddings = embedding_model.embed_query(query)

        validate_embeddings(
            embeddings=[embeddings],
            expected_count=1,
        )

        return embeddings

    except Exception as e:
        raise RuntimeError("Query embedding generation failed.") from e


def validate_embeddings(
    embeddings: list[list[float]],
    expected_count: int,
    expected_dimensions: int = 384,
) -> None:
    """
    Validate embedding count, vector dimensionality, and numeric integrity.
    Raises ValueError if validation fails.
    """
    if len(embeddings) != expected_count:
        raise ValueError(
            f"Embedding count mismatch: expected {expected_count}, got {len(embeddings)}."
        )

    for i, vector in enumerate(embeddings):
        if not isinstance(vector, (list, tuple)):
            raise ValueError(
                f"Invalid embedding vector type at index {i}: expected list/tuple, got {type(vector).__name__}."
            )

        if len(vector) != expected_dimensions:
            raise ValueError(
                f"Dimension mismatch at index {i}: expected {expected_dimensions}, got {len(vector)}."
            )

        for j, val in enumerate(vector):
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(
                    f"Non-numeric, NaN, or infinite value detected at vector {i}, dimension {j}: {val}."
                )
