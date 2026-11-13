import math

from langchain_huggingface import HuggingFaceEmbeddings

print(">>> BEFORE EMBEDDING MODEL")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
)
print(">>> AFTER EMBEDDING MODEL")

def embed_documents(documents):
    """Generate embeddings for a list of LangChain Documents."""

    texts = [document.page_content for document in documents]

    return embedding_model.embed_documents(texts)

def embed_query(query: str):
    return embedding_model.embed_query(query)


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