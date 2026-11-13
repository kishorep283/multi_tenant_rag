import hashlib
import math
from io import BytesIO
from unittest.mock import patch

import pymupdf
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient

from rag.models import Company, Document, DocumentChunk, DocumentParent, UserProfile
from rag.services.embedding import validate_embeddings
from rag.services.pipeline import PipelineError, process_document

User = get_user_model()


def create_sample_pdf_bytes(text: str = "Company Policy Handbook Content for Testing.") -> bytes:
    """Create a minimal valid PDF in memory using pymupdf."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class SecureUploadAPITests(TestCase):

    def setUp(self):
        self.client = APIClient()

        # Create Tenant Company A
        self.company_a = Company.objects.create(name="Acme Corp")
        self.user_a = User.objects.create_user(
            username="alice",
            password="securepassword123",
            email="alice@acme.com",
        )
        self.profile_a = UserProfile.objects.create(
            user=self.user_a,
            company=self.company_a,
            role=UserProfile.Role.ADMIN,
        )
        self.token_a = Token.objects.create(user=self.user_a)

        # Create Tenant Company B
        self.company_b = Company.objects.create(name="Beta LLC")
        self.user_b = User.objects.create_user(
            username="bob",
            password="securepassword123",
            email="bob@beta.com",
        )
        self.profile_b = UserProfile.objects.create(
            user=self.user_b,
            company=self.company_b,
            role=UserProfile.Role.MEMBER,
        )
        self.token_b = Token.objects.create(user=self.user_b)

        # User without profile
        self.user_no_profile = User.objects.create_user(
            username="noprofile",
            password="securepassword123",
        )
        self.token_no_profile = Token.objects.create(user=self.user_no_profile)

    def test_unauthenticated_upload_rejected(self):
        """Unauthenticated requests must be rejected with 401 Unauthorized."""
        pdf_bytes = create_sample_pdf_bytes()
        file = SimpleUploadedFile("policy.pdf", pdf_bytes, content_type="application/pdf")

        response = self.client.post(
            "/rag/documents/upload/",
            {"file": file, "department": "HR", "document_type": "Policy"},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_user_without_company_profile_rejected(self):
        """Authenticated users without a valid company profile must receive 403 Forbidden."""
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {self.token_no_profile.key}")

        pdf_bytes = create_sample_pdf_bytes()
        file = SimpleUploadedFile("policy.pdf", pdf_bytes, content_type="application/pdf")

        response = self.client.post(
            "/rag/documents/upload/",
            {"file": file, "department": "HR", "document_type": "Policy"},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertIn("no associated company profile", response.data["detail"])

    def test_token_generation_endpoint(self):
        """Users can exchange valid credentials for an authentication token."""
        response = self.client.post(
            "/rag/auth/token/",
            {"username": "alice", "password": "securepassword123"},
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["token"], self.token_a.key)

        # Invalid password rejected
        response_bad = self.client.post(
            "/rag/auth/token/",
            {"username": "alice", "password": "wrongpassword"},
        )
        self.assertEqual(response_bad.status_code, status.HTTP_400_BAD_REQUEST)

    def test_client_supplied_company_id_ignored(self):
        """Tenant ownership is derived solely from UserProfile, ignoring client company_id."""
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {self.token_a.key}")

        pdf_bytes = create_sample_pdf_bytes("Acme confidential engineering guidelines.")
        file = SimpleUploadedFile("guide.pdf", pdf_bytes, content_type="application/pdf")

        response = self.client.post(
            "/rag/documents/upload/",
            {
                "file": file,
                "department": "Engineering",
                "document_type": "Guideline",
                "company_id": self.company_b.id,  # Malicious client attempts cross-tenant upload
            },
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        doc = Document.objects.get(id=response.data["document_id"])
        self.assertEqual(doc.company, self.company_a)
        self.assertNotEqual(doc.company, self.company_b)

    def test_non_pdf_file_rejected(self):
        """Non-PDF extensions are rejected with 400 Bad Request."""
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {self.token_a.key}")

        file = SimpleUploadedFile("notes.txt", b"Plain text notes", content_type="text/plain")
        response = self.client.post(
            "/rag/documents/upload/",
            {"file": file, "department": "HR", "document_type": "Policy"},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data["detail"], "Only PDF files are supported.")

    def test_invalid_pdf_signature_rejected(self):
        """Files ending in .pdf but lacking '%PDF-' magic bytes are rejected."""
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {self.token_a.key}")

        fake_pdf = SimpleUploadedFile("fake.pdf", b"NOT_A_REAL_PDF_HEADER", content_type="application/pdf")
        response = self.client.post(
            "/rag/documents/upload/",
            {"file": fake_pdf, "department": "HR", "document_type": "Policy"},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data["detail"], "Invalid PDF file signature.")

    @override_settings(MAX_UPLOAD_SIZE_BYTES=100)
    def test_oversized_file_rejected(self):
        """Files exceeding the configured maximum size are rejected."""
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {self.token_a.key}")

        large_bytes = create_sample_pdf_bytes("A" * 500)
        file = SimpleUploadedFile("large.pdf", large_bytes, content_type="application/pdf")
        response = self.client.post(
            "/rag/documents/upload/",
            {"file": file, "department": "HR", "document_type": "Policy"},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("File size exceeds", response.data["detail"])

    def test_successful_upload_and_persistence(self):
        """A valid upload creates Document(READY), persists the file, and links parents and chunks."""
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {self.token_a.key}")

        content = "Security Policy: Password rotation every 90 days is strictly enforced."
        pdf_bytes = create_sample_pdf_bytes(content)
        file = SimpleUploadedFile("security_policy.pdf", pdf_bytes, content_type="application/pdf")

        response = self.client.post(
            "/rag/documents/upload/",
            {"file": file, "department": "Security", "document_type": "Policy"},
            format="multipart",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Check response contract preservation
        self.assertIn("document_id", response.data)
        self.assertEqual(response.data["status"], "READY")
        self.assertEqual(response.data["filename"], "security_policy.pdf")
        self.assertEqual(response.data["department"], "Security")
        self.assertEqual(response.data["document_type"], "Policy")
        self.assertGreater(response.data["parents_stored"], 0)
        self.assertGreater(response.data["children_stored"], 0)

        # Check Document model integrity
        doc = Document.objects.get(id=response.data["document_id"])
        self.assertEqual(doc.status, Document.Status.READY)
        self.assertEqual(doc.company, self.company_a)
        self.assertEqual(doc.uploaded_by, self.user_a)
        self.assertEqual(doc.original_filename, "security_policy.pdf")
        self.assertEqual(doc.file_hash, hashlib.sha256(pdf_bytes).hexdigest())
        self.assertTrue(doc.file.storage.exists(doc.file.name))

        # Check Parent and Chunk relationships
        parents = DocumentParent.objects.filter(document=doc)
        self.assertEqual(parents.count(), response.data["parents_stored"])
        for parent in parents:
            self.assertEqual(parent.company, self.company_a)
            self.assertEqual(parent.document, doc)

        chunks = DocumentChunk.objects.filter(document=doc)
        self.assertEqual(chunks.count(), response.data["children_stored"])
        for chunk in chunks:
            self.assertEqual(chunk.company, self.company_a)
            self.assertEqual(chunk.document, doc)
            self.assertIsNotNone(chunk.embedding)
            self.assertEqual(len(chunk.embedding), 384)

    def test_ingestion_failure_handling(self):
        """Ingestion failure sets FAILED status, preserves file, and leaves no partial chunks."""
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {self.token_a.key}")

        pdf_bytes = create_sample_pdf_bytes()
        file = SimpleUploadedFile("faulty.pdf", pdf_bytes, content_type="application/pdf")

        with patch("rag.services.pipeline.embed_documents", side_effect=RuntimeError("Embedding model failure")):
            response = self.client.post(
                "/rag/documents/upload/",
                {"file": file, "department": "Finance", "document_type": "Policy"},
                format="multipart",
            )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data["detail"], "The uploaded file could not be processed.")

        # Document record must be marked FAILED
        doc = Document.objects.filter(original_filename="faulty.pdf").first()
        self.assertIsNotNone(doc)
        self.assertEqual(doc.status, Document.Status.FAILED)
        self.assertIn("failed", doc.error_message.lower())

        # Permanent file must be preserved for debugging
        self.assertTrue(doc.file.storage.exists(doc.file.name))

        # No partial parents or chunks persisted
        self.assertEqual(DocumentParent.objects.filter(document=doc).count(), 0)
        self.assertEqual(DocumentChunk.objects.filter(document=doc).count(), 0)


class EmbeddingValidationTests(TestCase):
    """Unit tests for validate_embeddings helper."""

    def test_valid_embeddings_pass(self):
        embeddings = [[0.1] * 384, [0.2] * 384]
        # Should complete without raising
        validate_embeddings(embeddings, expected_count=2, expected_dimensions=384)

    def test_count_mismatch_raises(self):
        embeddings = [[0.1] * 384]
        with self.assertRaises(ValueError) as ctx:
            validate_embeddings(embeddings, expected_count=2, expected_dimensions=384)
        self.assertIn("Embedding count mismatch", str(ctx.exception))

    def test_dimension_mismatch_raises(self):
        embeddings = [[0.1] * 128]
        with self.assertRaises(ValueError) as ctx:
            validate_embeddings(embeddings, expected_count=1, expected_dimensions=384)
        self.assertIn("Dimension mismatch", str(ctx.exception))

    def test_nan_value_raises(self):
        embeddings = [[0.1] * 383 + [float("nan")]]
        with self.assertRaises(ValueError) as ctx:
            validate_embeddings(embeddings, expected_count=1, expected_dimensions=384)
        self.assertIn("NaN, or infinite", str(ctx.exception))

    def test_infinite_value_raises(self):
        embeddings = [[0.1] * 383 + [float("inf")]]
        with self.assertRaises(ValueError) as ctx:
            validate_embeddings(embeddings, expected_count=1, expected_dimensions=384)
        self.assertIn("NaN, or infinite", str(ctx.exception))

    def test_non_numeric_element_raises(self):
        embeddings = [[0.1] * 383 + ["not_a_number"]]
        with self.assertRaises(ValueError) as ctx:
            validate_embeddings(embeddings, expected_count=1, expected_dimensions=384)
        self.assertIn("Non-numeric", str(ctx.exception))


class DocumentPipelineServiceTests(TestCase):
    """Unit tests for rag/services/pipeline.py."""

    def setUp(self):
        self.company = Company.objects.create(name="Pipeline Test Co")
        self.user = User.objects.create_user(username="testpipeline", password="pw")
        self.profile = UserProfile.objects.create(user=self.user, company=self.company)

        pdf_bytes = create_sample_pdf_bytes("Testing pipeline orchestration and atomic replace.")
        self.uploaded_file = SimpleUploadedFile("pipeline_doc.pdf", pdf_bytes, content_type="application/pdf")

        self.document = Document.objects.create(
            company=self.company,
            uploaded_by=self.user,
            original_filename="pipeline_doc.pdf",
            file=self.uploaded_file,
            file_hash=hashlib.sha256(pdf_bytes).hexdigest(),
            status=Document.Status.UPLOADED,
            metadata={"department": "Engineering", "document_type": "Manual"},
        )

    def test_unpersisted_document_raises_pipeline_error(self):
        unsaved_doc = Document(original_filename="unsaved.pdf")
        with self.assertRaises(PipelineError) as ctx:
            process_document(unsaved_doc)
        self.assertIn("persisted Document instance", str(ctx.exception))

    def test_document_without_company_raises_pipeline_error(self):
        """Pipeline rejects a document whose company relationship is absent."""
        from unittest.mock import MagicMock

        mock_doc = MagicMock(spec=Document)
        mock_doc.pk = 9999  # Looks like a persisted document
        mock_doc.company_id = None  # No tenant attached

        with self.assertRaises(PipelineError) as ctx:
            process_document(mock_doc)
        self.assertIn("no associated tenant company", str(ctx.exception))

    def test_process_document_success_with_mocked_embeddings(self):
        """Pipeline runs through all stages and sets document READY."""
        # 1 child chunk produced, mock returns 1 vector of 384 dimensions
        mock_vector = [0.05] * 384
        with patch("rag.services.pipeline.embed_documents", return_value=[mock_vector]):
            result = process_document(self.document)

        self.assertEqual(result["parents_stored"], 1)
        self.assertEqual(result["children_stored"], 1)

        self.document.refresh_from_db()
        self.assertEqual(self.document.status, Document.Status.READY)
        self.assertEqual(self.document.error_message, "")

        # Verify DB records
        parents = DocumentParent.objects.filter(document=self.document)
        self.assertEqual(parents.count(), 1)
        parent = parents.first()
        self.assertEqual(parent.company, self.company)
        self.assertEqual(parent.source_filename, "pipeline_doc.pdf")

        chunks = DocumentChunk.objects.filter(document=self.document)
        self.assertEqual(chunks.count(), 1)
        chunk = chunks.first()
        self.assertEqual(chunk.company, self.company)
        self.assertEqual(chunk.parent, parent)
        self.assertEqual(len(chunk.embedding), 384)

    def test_process_document_embedding_validation_failure(self):
        """When embedding dimension is wrong, pipeline marks document FAILED and raises PipelineError."""
        # Return vector with dimension 128 instead of 384
        mock_bad_vector = [0.05] * 128
        with patch("rag.services.pipeline.embed_documents", return_value=[mock_bad_vector]):
            with self.assertRaises(PipelineError):
                process_document(self.document)

        self.document.refresh_from_db()
        self.assertEqual(self.document.status, Document.Status.FAILED)
        self.assertIn("failed", self.document.error_message.lower())

        # Zero chunks or parents persisted
        self.assertEqual(DocumentParent.objects.filter(document=self.document).count(), 0)
        self.assertEqual(DocumentChunk.objects.filter(document=self.document).count(), 0)

    def test_process_document_db_failure_rolls_back(self):
        """Database exception inside transaction rolls back all chunks and marks document FAILED."""
        mock_vector = [0.05] * 384
        with patch("rag.services.pipeline.embed_documents", return_value=[mock_vector]):
            with patch("rag.models.DocumentChunk.objects.create", side_effect=RuntimeError("DB write fail")):
                with self.assertRaises(PipelineError):
                    process_document(self.document)

        self.document.refresh_from_db()
        self.assertEqual(self.document.status, Document.Status.FAILED)

        # Atomic rollback ensures no partial parent records remain
        self.assertEqual(DocumentParent.objects.filter(document=self.document).count(), 0)
        self.assertEqual(DocumentChunk.objects.filter(document=self.document).count(), 0)

    def test_reprocessing_replaces_records_without_duplicates(self):
        """Reprocessing the same document replaces old records without duplicating chunks."""
        mock_vector = [0.05] * 384
        with patch("rag.services.pipeline.embed_documents", return_value=[mock_vector]):
            # First processing
            res1 = process_document(self.document)
            self.assertEqual(res1["parents_stored"], 1)
            self.assertEqual(res1["children_stored"], 1)

            # Second processing (retry / update)
            res2 = process_document(self.document)
            self.assertEqual(res2["parents_stored"], 1)
            self.assertEqual(res2["children_stored"], 1)

        # Confirm exactly 1 parent and 1 chunk exist (NOT 2)
        self.assertEqual(DocumentParent.objects.filter(document=self.document).count(), 1)
        self.assertEqual(DocumentChunk.objects.filter(document=self.document).count(), 1)

    def test_reprocessing_isolation_to_exact_document(self):
        """Reprocessing Document 1 does not alter or delete Document 2's chunks or parents."""
        company_other = Company.objects.create(name="Other Co")
        pdf_bytes2 = create_sample_pdf_bytes("Doc 2 content for another company.")
        doc2 = Document.objects.create(
            company=company_other,
            original_filename="doc2.pdf",
            file=SimpleUploadedFile("doc2.pdf", pdf_bytes2, content_type="application/pdf"),
            file_hash="hash2",
            status=Document.Status.UPLOADED,
        )

        mock_vector = [0.05] * 384
        with patch("rag.services.pipeline.embed_documents", return_value=[mock_vector]):
            process_document(self.document)
            process_document(doc2)

            self.assertEqual(DocumentParent.objects.filter(document=doc2).count(), 1)
            self.assertEqual(DocumentChunk.objects.filter(document=doc2).count(), 1)

            # Reprocess self.document
            process_document(self.document)

            # Confirm doc2's records are untouched
            self.assertEqual(DocumentParent.objects.filter(document=doc2).count(), 1)
            self.assertEqual(DocumentChunk.objects.filter(document=doc2).count(), 1)


class RetrievalTenantIsolationTests(TestCase):
    """
    Verify that ChunksRetrieveView enforces authentication and scopes every
    database query to the authenticated user's company.

    External calls (embed_query, rerank_results, generate_answer) are mocked
    so tests do not require network access or model inference.
    """

    # A fixed 384-dimensional vector used as both chunk embedding and mock query
    # embedding so cosine distance == 0 (perfect match) for all test chunks.
    EMBEDDING = [0.1] * 384

    def setUp(self):
        self.client = APIClient()

        # --- Company A -------------------------------------------------
        self.company_a = Company.objects.create(name="Isolation Alpha Corp")
        self.user_a = User.objects.create_user(
            username="isolation_alice", password="pw"
        )
        UserProfile.objects.create(user=self.user_a, company=self.company_a)
        self.token_a = Token.objects.create(user=self.user_a)

        # --- Company B -------------------------------------------------
        self.company_b = Company.objects.create(name="Isolation Beta LLC")
        self.user_b = User.objects.create_user(
            username="isolation_bob", password="pw"
        )
        UserProfile.objects.create(user=self.user_b, company=self.company_b)
        self.token_b = Token.objects.create(user=self.user_b)

        # --- User without a profile ------------------------------------
        self.user_no_profile = User.objects.create_user(
            username="isolation_noprofile", password="pw"
        )
        self.token_no_profile = Token.objects.create(user=self.user_no_profile)

        # --- DocumentChunks --------------------------------------------
        # Company A — HR department
        self.chunk_a_hr = DocumentChunk.objects.create(
            company=self.company_a,
            source_filename="alpha_hr.pdf",
            content="Alpha Corp salary policy: employees receive annual bonuses.",
            metadata={"department": "HR"},
            embedding=self.EMBEDDING,
        )

        # Company B — HR department (identical content to prove isolation)
        self.chunk_b_hr = DocumentChunk.objects.create(
            company=self.company_b,
            source_filename="beta_hr.pdf",
            content="Alpha Corp salary policy: employees receive annual bonuses.",
            metadata={"department": "HR"},
            embedding=self.EMBEDDING,
        )

        # Company A — Finance department (must not appear in HR queries)
        self.chunk_a_finance = DocumentChunk.objects.create(
            company=self.company_a,
            source_filename="alpha_finance.pdf",
            content="Alpha Corp quarterly revenue report with salary figures.",
            metadata={"department": "Finance"},
            embedding=self.EMBEDDING,
        )

        # Legacy chunk — company=NULL (must be invisible to all tenants)
        self.chunk_legacy = DocumentChunk.objects.create(
            company=None,
            source_filename="legacy.pdf",
            content="Alpha Corp salary policy: employees receive annual bonuses.",
            metadata={"department": "HR"},
            embedding=self.EMBEDDING,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retrieve(self, token, department="HR", query="salary"):
        """Authenticated GET to the retrieve endpoint with all external calls mocked."""
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {token.key}")
        with (
            patch("rag.views.embed_query", return_value=self.EMBEDDING),
            patch(
                "rag.views.rerank_results",
                side_effect=lambda query, results, top_k: results[:top_k],
            ),
            patch("rag.views.generate_answer", return_value="Mocked LLM answer."),
        ):
            return self.client.get(
                "/rag/documents/retrieve/",
                {"department": department, "query": query},
            )

    def _result_ids(self, response):
        return {r["id"] for r in response.data.get("reranked_results", [])}

    # ------------------------------------------------------------------
    # Authentication tests
    # ------------------------------------------------------------------

    def test_unauthenticated_retrieve_rejected(self):
        """Unauthenticated request must receive 401 Unauthorized."""
        self.client.credentials()  # clear any credentials
        response = self.client.get(
            "/rag/documents/retrieve/",
            {"department": "HR", "query": "salary"},
        )
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_user_without_profile_rejected(self):
        """Authenticated user with no UserProfile must receive 403 Forbidden."""
        self.client.credentials(
            HTTP_AUTHORIZATION=f"Token {self.token_no_profile.key}"
        )
        response = self.client.get(
            "/rag/documents/retrieve/",
            {"department": "HR", "query": "salary"},
        )
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertIn("no associated company profile", response.data["detail"])

    # ------------------------------------------------------------------
    # Tenant isolation — vector + keyword search
    # ------------------------------------------------------------------

    def test_company_a_retrieves_own_hr_chunks(self):
        """Company A sees its own HR chunk in results."""
        response = self._retrieve(self.token_a)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn(self.chunk_a_hr.id, self._result_ids(response))

    def test_company_a_cannot_see_company_b_chunks(self):
        """Company A query must not return any chunk belonging to Company B,
        even when the content is byte-for-byte identical and the embedding
        distance is zero."""
        response = self._retrieve(self.token_a)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertNotIn(self.chunk_b_hr.id, self._result_ids(response))

    def test_company_b_cannot_see_company_a_chunks(self):
        """Symmetric: Company B query must not return Company A's chunks."""
        response = self._retrieve(self.token_b)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertNotIn(self.chunk_a_hr.id, self._result_ids(response))

    def test_legacy_null_company_chunks_excluded(self):
        """Chunks with company=NULL must never appear for any authenticated tenant."""
        response = self._retrieve(self.token_a)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertNotIn(self.chunk_legacy.id, self._result_ids(response))

    def test_department_filter_respected_within_company(self):
        """Company A's Finance chunk must not appear in an HR query."""
        response = self._retrieve(self.token_a, department="HR", query="salary")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertNotIn(self.chunk_a_finance.id, self._result_ids(response))

    def test_response_structure_preserved(self):
        """The existing response contract must still be present after securing the view."""
        response = self._retrieve(self.token_a)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for key in ("query", "department", "answer", "reranked_results", "parent_documents"):
            self.assertIn(key, response.data)
        self.assertEqual(response.data["query"], "salary")
        self.assertEqual(response.data["department"], "HR")


class ParentDocumentViewTests(TestCase):
    """
    Phase 4 Step 2: ParentDocumentView must require authentication
    and scope chunk lookups exclusively to the authenticated user's company.
    """

    def setUp(self):
        # --- Companies ---
        self.company_a = Company.objects.create(name="Alpha Corp")
        self.company_b = Company.objects.create(name="Beta Corp")

        # --- Users and tokens for Company A ---
        self.user_a = get_user_model().objects.create_user(
            username="alice_pdv", password="pass"
        )
        UserProfile.objects.create(user=self.user_a, company=self.company_a)
        self.token_a = Token.objects.create(user=self.user_a)

        # --- Users and tokens for Company B ---
        self.user_b = get_user_model().objects.create_user(
            username="bob_pdv", password="pass"
        )
        UserProfile.objects.create(user=self.user_b, company=self.company_b)
        self.token_b = Token.objects.create(user=self.user_b)

        # --- User without a profile ---
        self.user_no_profile = get_user_model().objects.create_user(
            username="noprofile_pdv", password="pass"
        )
        self.token_no_profile = Token.objects.create(user=self.user_no_profile)

        # --- Parent documents ---
        self.parent_a = DocumentParent.objects.create(
            company=self.company_a,
            content="Alpha parent content",
        )
        self.parent_b = DocumentParent.objects.create(
            company=self.company_b,
            content="Beta parent content",
        )

        # --- Chunks: Company A ---
        self.chunk_a1 = DocumentChunk.objects.create(
            company=self.company_a,
            parent=self.parent_a,
            content="Alpha chunk 1",
            embedding=[0.1] * 384,
        )
        self.chunk_a2 = DocumentChunk.objects.create(
            company=self.company_a,
            parent=self.parent_a,
            content="Alpha chunk 2",
            embedding=[0.2] * 384,
        )

        # --- Chunks: Company B ---
        self.chunk_b1 = DocumentChunk.objects.create(
            company=self.company_b,
            parent=self.parent_b,
            content="Beta chunk 1",
            embedding=[0.3] * 384,
        )

        # --- Legacy chunk (company=NULL) ---
        self.chunk_legacy = DocumentChunk.objects.create(
            company=None,
            content="Legacy chunk",
            embedding=[0.4] * 384,
        )

        self.url = "/rag/documents/parent"

    def _post(self, token=None, child_ids=None):
        if child_ids is None:
            child_ids = [self.chunk_a1.id]
        headers = {}
        if token:
            headers["HTTP_AUTHORIZATION"] = f"Token {token.key}"
        return self.client.post(
            self.url,
            data={"child_ids": child_ids},
            content_type="application/json",
            **headers,
        )

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def test_unauthenticated_rejected(self):
        """No credentials -> 401."""
        response = self._post(token=None)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_user_without_profile_rejected(self):
        """Authenticated but no UserProfile -> 403."""
        response = self._post(token=self.token_no_profile)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    # ------------------------------------------------------------------
    # Tenant isolation
    # ------------------------------------------------------------------

    def test_company_a_retrieves_own_parents(self):
        """Company A can retrieve parents via its own chunk IDs."""
        response = self._post(
            token=self.token_a,
            child_ids=[self.chunk_a1.id, self.chunk_a2.id],
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["children_found"], 2)
        self.assertEqual(response.data["parents_found"], 1)
        parent_ids = [p["id"] for p in response.data["parents"]]
        self.assertIn(self.parent_a.id, parent_ids)

    def test_company_a_cannot_use_company_b_chunk_ids(self):
        """Company A supplying Company B's chunk IDs gets silently empty results."""
        response = self._post(
            token=self.token_a,
            child_ids=[self.chunk_b1.id],
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["children_found"], 0)
        self.assertEqual(response.data["parents_found"], 0)
        self.assertEqual(response.data["parents"], [])

    def test_legacy_null_company_chunks_cannot_retrieve_parents(self):
        """Chunks with company=NULL cannot be accessed by any tenant."""
        response = self._post(
            token=self.token_a,
            child_ids=[self.chunk_legacy.id],
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["children_found"], 0)
        self.assertEqual(response.data["parents_found"], 0)

    def test_cross_company_chunk_ids_mixed_with_own(self):
        """Mixed IDs: only the caller's own chunks are counted."""
        response = self._post(
            token=self.token_a,
            child_ids=[self.chunk_a1.id, self.chunk_b1.id, self.chunk_legacy.id],
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # Only chunk_a1 belongs to company_a
        self.assertEqual(response.data["children_found"], 1)
        self.assertEqual(response.data["parents_found"], 1)

    # ------------------------------------------------------------------
    # Input validation (these run before company resolution)
    # ------------------------------------------------------------------

    def test_invalid_child_ids_rejected(self):
        """Non-integer IDs must return 400."""
        response = self._post(token=self.token_a, child_ids=["abc"])
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_boolean_child_ids_rejected(self):
        """Boolean values must be rejected even though bool is a subclass of int."""
        response = self._post(token=self.token_a, child_ids=[True])
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_zero_child_id_rejected(self):
        """Non-positive integers must return 400."""
        response = self._post(token=self.token_a, child_ids=[0])
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_empty_child_ids_rejected(self):
        """An empty list must return 400."""
        response = self._post(token=self.token_a, child_ids=[])
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_nonexistent_chunk_ids_handled_safely(self):
        """IDs that don't exist at all return 200 with zero results."""
        response = self._post(token=self.token_a, child_ids=[999999])
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["children_found"], 0)
        self.assertEqual(response.data["parents_found"], 0)

    def test_duplicate_chunk_ids_deduplicated(self):
        """Duplicate IDs in the request must not produce duplicate parent entries."""
        response = self._post(
            token=self.token_a,
            child_ids=[self.chunk_a1.id, self.chunk_a1.id],
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # DB deduplicates via QuerySet; only one chunk returned
        self.assertEqual(response.data["children_found"], 1)
        self.assertEqual(response.data["parents_found"], 1)

    def test_response_structure_preserved(self):
        """All four response keys must always be present."""
        response = self._post(token=self.token_a)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for key in ("requested_child_ids", "children_found", "parents_found", "parents"):
            self.assertIn(key, response.data)
        self.assertEqual(response.data["requested_child_ids"], [self.chunk_a1.id])
