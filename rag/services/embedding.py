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