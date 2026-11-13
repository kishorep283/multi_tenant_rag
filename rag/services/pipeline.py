import logging
from pathlib import Path

from django.db import transaction
from langchain_community.document_loaders import PyMuPDFLoader

from rag.models import Document, DocumentChunk, DocumentParent
from rag.services.chunker import chunk_documents
from rag.services.cleaner import clean_documents
from rag.services.embedding import embed_documents, validate_embeddings

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Custom exception raised when document ingestion fails."""
    pass


def load_pdf(file_path: str | Path):
    """Load a PDF into LangChain documents, one document per page."""
    return PyMuPDFLoader(str(file_path)).load()


def process_document(document: Document) -> dict:
    """
    Orchestrate the complete ingestion lifecycle for a persisted Document.

    Stages:
    1. Validate document persistence and tenant ownership.
    2. Transition status to PROCESSING.
    3. Extract text from the stored PDF.
    4. Clean extracted text.
    5. Split into parent and child chunks.
    6. Generate embeddings for child chunks.
    7. Validate embedding count, dimensions, and numeric values.
    8. Atomically replace previous records for this document (with row-level lock)
       and commit replacement parents, chunks, and READY status together.
    9. On failure, mark FAILED, preserve uploaded file, and raise PipelineError.
    """
    if not isinstance(document, Document) or not document.pk:
        raise PipelineError("process_document requires a persisted Document instance with an ID.")

    if not document.company_id:
        raise PipelineError(f"Document {document.pk} has no associated tenant company.")

    company = document.company
    department = document.metadata.get("department", "")
    document_type = document.metadata.get("document_type", "")

    # Transition to PROCESSING before running heavy stages
    try:
        document.status = Document.Status.PROCESSING
        document.error_message = ""
        document.save(update_fields=["status", "error_message", "updated_at"])
    except Exception as exc:
        logger.error("Failed to update document %s status to PROCESSING: %s", document.pk, exc)
        raise PipelineError(f"Database error initializing processing for document {document.pk}.") from exc

    try:
        # Stage 1: Text extraction from stored file
        file_path = document.file.path
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Stored document file not found at path: {file_path}")

        raw_documents = load_pdf(file_path)
        if not raw_documents:
            raise ValueError("No readable content could be extracted from the document.")

        # Stage 2: Text cleaning
        cleaned_documents = clean_documents(raw_documents)
        if not cleaned_documents:
            raise ValueError("All document pages were empty after text cleaning.")

        # Stage 3: Parent-child chunking
        parent_child_documents = chunk_documents(
            cleaned_documents,
            department=department,
        )
        if not parent_child_documents:
            raise ValueError("Document chunking yielded zero parent chunks.")

        child_documents = []
        for item in parent_child_documents:
            child_documents.extend(item["children"])

        if not child_documents:
            raise ValueError("Document chunking yielded zero child chunks.")

        # Stage 4: Embedding generation
        embeddings = embed_documents(child_documents)

        # Stage 5: Embedding validation
        validate_embeddings(
            embeddings,
            expected_count=len(child_documents),
            expected_dimensions=384,
        )

        # Stage 6: Atomic persistence with row-level lock and safe replacement
        with transaction.atomic():
            # Acquire row-level lock on the document to serialize concurrent attempts
            doc = Document.objects.select_for_update().get(pk=document.pk)

            # Safely remove previous records belonging strictly to this exact document
            DocumentChunk.objects.filter(document=doc).delete()
            DocumentParent.objects.filter(document=doc).delete()

            # Insert replacement parent and child records
            embedding_index = 0
            for item in parent_child_documents:
                parent_doc = item["parent"]
                children = item["children"]

                parent_record = DocumentParent.objects.create(
                    document=doc,
                    company=company,
                    source_filename=doc.original_filename,
                    content=parent_doc.page_content,
                    metadata={
                        **parent_doc.metadata,
                        "department": department,
                        "document_type": document_type,
                    },
                )

                for child in children:
                    embedding = embeddings[embedding_index]
                    embedding_index += 1

                    DocumentChunk.objects.create(
                        document=doc,
                        company=company,
                        parent=parent_record,
                        source_filename=doc.original_filename,
                        content=child.page_content,
                        metadata={
                            **child.metadata,
                            "department": department,
                            "document_type": document_type,
                        },
                        embedding=embedding,
                    )

            # Commit READY status atomically with the new chunk records
            doc.status = Document.Status.READY
            doc.error_message = ""
            doc.save(update_fields=["status", "error_message", "updated_at"])

        return {
            "parents_stored": len(parent_child_documents),
            "children_stored": len(child_documents),
        }

    except Exception as exc:
        logger.error("Ingestion failed for document %s: %s", document.pk, exc, exc_info=True)

        # Attempt to record failure status and sanitized error message on Document
        safe_error = "Document processing failed during extraction, chunking, or embedding."
        status_updated = False
        try:
            document.status = Document.Status.FAILED
            document.error_message = safe_error
            document.save(update_fields=["status", "error_message", "updated_at"])
            status_updated = True
        except Exception as db_exc:
            logger.error(
                "Could not persist FAILED status for document %s due to database error: %s",
                document.pk,
                db_exc,
            )

        if not status_updated:
            raise PipelineError(
                f"Document processing failed and the failure status could not be saved to the database: {exc}"
            ) from exc

        raise PipelineError(safe_error) from exc

