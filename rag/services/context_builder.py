import tiktoken
parent_documents = [
    {
        "id": 37,
        "source_filename": "leave_policy.pdf",
        "content": "Eligible employees receive 12 days..."
    },
    {
        "id": 38,
        "source_filename": "leave_policy.pdf",
        "content": "Employees should submit leave requests..."
    }
]

def build_context(parent_documents, max_context_tokens=2000):
    context_parts = []
    total_tokens = 0

    for doc in parent_documents:
        content = doc.get("content", "")

        if not content:
            continue

        source = doc.get("source_filename", "Unknown document")
        parent_id = doc.get("id", "Unknown")

        document_context = (
            f"Document: {source}\n"
            f"Parent ID: {parent_id}\n\n"
            f"Content:\n{content}"
        )

        document_tokens = count_tokens(document_context)

        if total_tokens + document_tokens > max_context_tokens:
            continue

        context_parts.append(document_context)
        total_tokens += document_tokens

    context = "\n\n--------------------------------\n\n".join(context_parts)

    return context, total_tokens

def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))