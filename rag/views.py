import hashlib
from pathlib import Path

import pymupdf
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from django.db import transaction
from langchain_community.document_loaders import PyMuPDFLoader
from pgvector.django import CosineDistance
from rest_framework import generics, status
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from rag.services.parent_retrieval import retrieve_parent_documents
from .models import Company, Document, DocumentChunk, DocumentParent, UserProfile
from .serializers import CompanySerializer, UserCreateSerializer
from .services.chunker import chunk_documents
from .services.cleaner import clean_documents
from .services.embedding import embed_documents, embed_query
from .services.llm import generate_answer, reciprocal_rank_fusion
from .services.pipeline import PipelineError, process_document
from .services.reranker import rerank_results


def load_pdf(file_path: str | Path):
    """Load a PDF into LangChain documents, one document per page."""
    return PyMuPDFLoader(str(file_path)).load()


class PDFUploadView(APIView):
    """Receive a PDF, securely validate tenant ownership, and process document chunks."""

    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        # 1. Resolve tenant ownership from UserProfile
        if not hasattr(request.user, "profile") or request.user.profile.company is None:
            return Response(
                {"detail": "Authenticated user has no associated company profile."},
                status=status.HTTP_403_FORBIDDEN,
            )

        company = request.user.profile.company

        # 2. Extract and validate required metadata
        department = request.data.get("department")
        document_type = request.data.get("document_type")

        if not department:
            return Response(
                {"detail": "The 'department' field is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not document_type:
            return Response(
                {"detail": "The 'document_type' field is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 3. Validate uploaded file
        uploaded_file = request.FILES.get("file")
        if uploaded_file is None:
            return Response(
                {"detail": "Upload a PDF using the 'file' field."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not uploaded_file.name.lower().endswith(".pdf"):
            return Response(
                {"detail": "Only PDF files are supported."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        max_size = getattr(settings, "MAX_UPLOAD_SIZE_BYTES", 15 * 1024 * 1024)
        if uploaded_file.size > max_size:
            return Response(
                {
                    "detail": f"File size exceeds the maximum allowed limit ({max_size // (1024 * 1024)}MB)."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Read first bytes to verify %PDF- signature and compute hash
        sha256 = hashlib.sha256()
        header = None
        for chunk in uploaded_file.chunks():
            if header is None:
                header = chunk[:5]
            sha256.update(chunk)
        file_hash = sha256.hexdigest()

        if not header or not header.startswith(b"%PDF-"):
            return Response(
                {"detail": "Invalid PDF file signature."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        uploaded_file.seek(0)

        # Validate parseability with PyMuPDF
        try:
            pdf_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            doc_test = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            if doc_test.page_count == 0:
                doc_test.close()
                return Response(
                    {"detail": "The uploaded PDF contains no readable pages."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            doc_test.close()
        except Exception:
            uploaded_file.seek(0)
            return Response(
                {"detail": "The uploaded file could not be parsed as a valid PDF."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 4. Create Document record in PROCESSING status and persist file
        document = Document.objects.create(
            company=company,
            uploaded_by=request.user,
            original_filename=uploaded_file.name,
            file=uploaded_file,
            file_hash=file_hash,
            status=Document.Status.PROCESSING,
            metadata={
                "department": department,
                "document_type": document_type,
            },
        )

        # 5. Ingestion execution delegated to pipeline service
        try:
            result = process_document(document)
            document.refresh_from_db()  # Sync stale in-memory status with the committed DB value

            return Response(
                {
                    "document_id": document.id,
                    "status": document.status,
                    "filename": uploaded_file.name,
                    "department": department,
                    "document_type": document_type,
                    "parents_stored": result["parents_stored"],
                    "children_stored": result["children_stored"],
                },
                status=status.HTTP_200_OK,
            )

        except PipelineError:
            return Response(
                {"detail": "The uploaded file could not be processed."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception:
            return Response(
                {"detail": "The uploaded file could not be processed."},
                status=status.HTTP_400_BAD_REQUEST,
            )




class ChunksRetrieveView(APIView):
    """Retrieve document chunks from PostgreSQL based on department and document type.

    Requires authentication. All database queries are scoped to the authenticated
    user's company at the ORM level before results enter RRF or reranking.
    Legacy records with company=NULL are excluded automatically by the equality
    filter (SQL equality never matches NULL).
    """

    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        # 0. Resolve tenant ownership from UserProfile -- never trust a client-supplied ID
        if not hasattr(request.user, "profile") or request.user.profile.company is None:
            return Response(
                {"detail": "Authenticated user has no associated company profile."},
                status=status.HTTP_403_FORBIDDEN,
            )

        company = request.user.profile.company

        department = request.query_params.get("department")
        query = request.query_params.get("query")

        if not department:
            return Response(
                {"detail": "The 'department' query parameter is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not query:
            return Response(
                {"detail": "The 'query' query parameter is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 1. Convert user's query into an embedding
        query_embedding = embed_query(query)

        # 2. Vector / semantic search -- scoped to the authenticated user's company.
        #    Chunks from other companies and legacy records (company=NULL) are
        #    excluded at the database level before any scoring occurs.
        vector_chunks = (
            DocumentChunk.objects
            .filter(
                company=company,
                metadata__department=department,
                embedding__isnull=False,
            )
            .annotate(
                distance=CosineDistance("embedding", query_embedding)
            )
            .order_by("distance")[:10]
        )

        # 3. Keyword / full-text search -- same company scope as vector search.
        search_query = SearchQuery(query)

        keyword_chunks = (
            DocumentChunk.objects
            .filter(
                company=company,
                metadata__department=department,
                embedding__isnull=False,
            )
            .annotate(
                search_vector=SearchVector("content"),
                rank=SearchRank(
                    SearchVector("content"),
                    search_query,
                ),
            )
            .filter(search_vector=search_query)
            .order_by("-rank")[:10]
        )

        # 4. Serialize vector search results
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

        # 5. Serialize keyword search results
        keyword_results = [
            {
                "id": chunk.id,
                "parent_id": chunk.parent_id,
                "source_filename": chunk.source_filename,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "rank": float(chunk.rank),
            }
            for chunk in keyword_chunks
        ]
        reranked_results = reciprocal_rank_fusion(
            vector_results,
            keyword_results,
        )
        reranked_results = rerank_results(
            query=query,
            results=reranked_results,
            top_k=5,
        )
        parent_documents = retrieve_parent_documents(
            reranked_results
        )
        context = "\n\n".join(
            parent["content"]
            for parent in parent_documents
        )
        answer = generate_answer(query,context)
        return Response(
            {
                "query": query,
                "department": department,
                "answer": answer,
                "reranked_results": reranked_results,
                "parent_documents": parent_documents,
            },
            status=status.HTTP_200_OK,
        )

class RerankerTestView(APIView):

    def post(self, request, *args, **kwargs):

        query = request.data.get("query")
        results = request.data.get("results")

        if not query:
            return Response(
                {"detail": "The 'query' field is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not results:
            return Response(
                {"detail": "The 'results' field is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        reranked_results = rerank_results(
            query=query,
            results=results,
            top_k=5,
        )

        return Response(
            {
                "query": query,
                "reranked_results": reranked_results,
            },
            status=status.HTTP_200_OK,
        )

class ParentDocumentView(APIView):
    """Retrieve parent documents based on child results."""

    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        # --- Input validation (does not require company context) ---
        child_ids = request.data.get("child_ids")

        if not isinstance(child_ids, list) or not child_ids:
            return Response(
                {
                    "detail": (
                        "Provide a non-empty 'child_ids' list."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        if any(
            isinstance(child_id, bool)
            or not isinstance(child_id, int)
            or child_id <= 0
            for child_id in child_ids
        ):
            return Response(
                {
                    "detail": (
                        "All child IDs must be positive integers."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # --- Resolve tenant company from authenticated user ---
        if not hasattr(request.user, "profile") or request.user.profile.company is None:
            return Response(
                {"detail": "Authenticated user has no associated company profile."},
                status=status.HTTP_403_FORBIDDEN,
            )
        company = request.user.profile.company

        # Load only child records belonging to this tenant.
        children = DocumentChunk.objects.filter(
            id__in=child_ids, company=company
        ).select_related("parent")

        child_results = [
            {
                "id": child.id,
                "parent_id": child.parent_id,
            }
            for child in children
        ]

        # Retrieve the corresponding unique parent documents.
        parents = retrieve_parent_documents(child_results)

        return Response(
            {
                "requested_child_ids": child_ids,
                "children_found": len(child_results),
                "parents_found": len(parents),
                "parents": parents,
            },
            status=status.HTTP_200_OK,
        )

class CompanyListCreateView(generics.ListCreateAPIView):
    queryset = Company.objects.all().order_by("id")
    serializer_class = CompanySerializer
    permission_classes = [AllowAny]

User = get_user_model()
class UserCreateView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserCreateSerializer
    permission_classes = [AllowAny]
