import re

from langchain_core.documents import Document

def clean_text(text: str) -> str:
    """Clean extracted PDF text without changing its meaning."""

    # Normalize Windows/Linux line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing spaces from each line
    text = "\n".join(line.strip() for line in text.splitlines())

    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_documents(documents: list[Document]) -> list[Document]:
    """Clean the page content while preserving LangChain metadata."""

    cleaned_documents = []

    for document in documents:
        cleaned_content = clean_text(document.page_content)

        # Ignore completely empty pages
        if not cleaned_content:
            continue

        cleaned_documents.append(
            Document(
                page_content=cleaned_content,
                metadata=document.metadata.copy(),
            )
        )

    return cleaned_documents