import tempfile
from pathlib import Path

from aiohttp import request

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import generics
from rest_framework.permissions import AllowAny
from langchain_community.document_loaders import PyMuPDFLoader

from .services.cleaner import clean_documents
from .services.chunker import chunk_documents
from .services.embedding import embed_documents,embed_query
from .models import DocumentChunk, DocumentParent,Company
from pgvector.django import CosineDistance
from .services.llm import generate_answer,reciprocal_rank_fusion
from .services.reranker import rerank_results
from rag.services.parent_retrieval import retrieve_parent_documents
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from .serializers import CompanySerializer, UserCreateSerializer
from django.contrib.auth import get_user_model
def load_pdf(file_path: str | Path):
    """Load a PDF into LangChain documents, one document per page."""
    return PyMuPDFLoader(str(file_path)).load()


class PDFUploadView(APIView):
    """Receive a PDF from the dashboard and load it into LangChain documents."""

    
    def post(self, request, *args, **kwargs):
        uploaded_file = request.FILES.get("file")
        department = request.data.get("department")
        document_type = request.data.get("document_type")

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

        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False
            ) as temporary_file:

                temporary_path = Path(temporary_file.name)

                for file_chunk in uploaded_file.chunks():
                    temporary_file.write(file_chunk)

            # 1. Load PDF
            documents = load_pdf(temporary_path)

            # 2. Clean extracted text
            documents = clean_documents(documents)

            # 3. Split into parent and child chunks
            parent_child_documents = chunk_documents(
                documents,
                department=department,
            )

            # 4. Extract child documents
            child_documents = []

            for item in parent_child_documents:
                child_documents.extend(item["children"])

            # 5. Generate embeddings for children only
            embeddings = embed_documents(child_documents)

            # 6. Create metadata for this document
            metadata = {
                "department": department,
                "document_type": document_type,
            }

            # 7. Store parents, children, and embeddings
            embedding_index = 0

            for item in parent_child_documents:

                parent_document = item["parent"]
                children = item["children"]

                # Create parent
                parent = DocumentParent.objects.create(
                    source_filename=uploaded_file.name,
                    content=parent_document.page_content,
                    metadata={
                        **parent_document.metadata,
                        **metadata,
                    },
                )

                # Create children
                for child in children:

                    embedding = embeddings[embedding_index]
                    embedding_index += 1

                    DocumentChunk.objects.create(
                        parent=parent,
                        source_filename=uploaded_file.name,
                        content=child.page_content,
                        metadata={
                            **child.metadata,
                            **metadata,
                        },
                        embedding=embedding,
                    )

        except Exception as e:
            print("Exception:", str(e))

            return Response(
                {"detail": "The uploaded file could not be processed."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return Response(
            {
                "filename": uploaded_file.name,
                "department": department,
                "document_type": document_type,
                "parents_stored": len(parent_child_documents),
                "children_stored": len(child_documents),
            },
            status=status.HTTP_200_OK,
        )


class ChunksRetrieveView(APIView):
    """Retrieve document chunks from PostgreSQL based on department and document type."""

    def get(self, request, *args, **kwargs):
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

        # 2. Vector / semantic search
        vector_chunks = (
            DocumentChunk.objects
            .filter(
                metadata__department=department,
                embedding__isnull=False,
            )
            .annotate(
                distance=CosineDistance("embedding", query_embedding)
            )
            .order_by("distance")[:10]
        )

        # 3. Keyword / full-text search
        search_query = SearchQuery(query)

        keyword_chunks = (
            DocumentChunk.objects
            .filter(
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

    def post(self, request, *args, **kwargs):
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

        # Load the actual child records from the database.
        children = DocumentChunk.objects.filter(
            id__in=child_ids
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
