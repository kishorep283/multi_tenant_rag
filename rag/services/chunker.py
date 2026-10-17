# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter


# def chunk_documents(
#     documents: list[Document],
#     chunk_size: int = 1000,
#     chunk_overlap: int = 200,
# ) -> list[Document]:
#     """Split cleaned documents into overlapping chunks for embedding."""

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#     )
#     return splitter.split_documents(documents)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document],
    department: str,
    parent_chunk_size: int = 2000,
    parent_chunk_overlap: int = 200,
    child_chunk_size: int = 500,
    child_chunk_overlap: int = 100,
):
    """Create parent and child chunks while preserving metadata."""

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )

    parent_documents = parent_splitter.split_documents(documents)

    parent_child_documents = []

    for parent in parent_documents:

        # Add department to parent metadata
        parent_metadata = parent.metadata.copy()
        parent_metadata["department"] = department

        parent_document = Document(
            page_content=parent.page_content,
            metadata=parent_metadata,
        )

        # Split this parent into children
        children = child_splitter.split_documents([parent_document])

        parent_child_documents.append({
            "parent": parent_document,
            "children": children,
        })

    return parent_child_documents