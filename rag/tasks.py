"""
Celery tasks for the rag app.

Week 1: Contains only a simple health-check / smoke-test task.
         The actual asynchronous ingestion task will be added in Week 2.
"""

from celery import shared_task

from rag.models import Document
from rag.services.pipeline import process_document, PipelineError
@shared_task(bind=True, name="rag.tasks.health_check")
def health_check(self, payload: str = "ping") -> dict:
    """
    Lightweight smoke-test task.

    Use this to verify that:
    - The Celery worker has started successfully.
    - The worker can connect to the Redis broker.
    - Task serialization (JSON) works end-to-end.
    - Results are stored in the Redis result backend.

    Usage (Django shell or test):
        from rag.tasks import health_check
        result = health_check.delay(payload="hello")
        print(result.get(timeout=10))
        # => {'status': 'ok', 'echo': 'hello', 'worker': '<hostname>'}
    """
    return {
        "status": "ok",
        "echo": payload,
        "worker": self.request.hostname,
    }

@shared_task(bind=True, name="rag.tasks.ingest_document")
def ingest_document(self, document_id: int) -> dict:
    """
    Process a persisted PDF document asynchronously.
    """
    try:
        document = Document.objects.get(pk=document_id)

    except Document.DoesNotExist:
        return {
            "document_id": document_id,
            "status": "NOT_FOUND",
        }

    try:
        result = process_document(document)

        return {
            "document_id": document_id,
            "status": "READY",
            **result,
        }

    except PipelineError:
        return {
            "document_id": document_id,
            "status": "FAILED",
        }

