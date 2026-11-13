"""
Celery application for the rag-project-2 Django project.

This module creates and configures the Celery application instance.
It is imported by config/__init__.py so that Django auto-discovers it
on startup and the @shared_task decorator works in all reusable apps.
"""

import os

from celery import Celery

# Tell Celery which Django settings module to use.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("rag_project")

# Read Celery configuration from Django settings using the CELERY_ namespace.
# All Celery-related settings must be prefixed with CELERY_ in settings.py.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Auto-discover tasks in INSTALLED_APPS (looks for tasks.py in each app).
app.autodiscover_tasks()


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    """Introspection task — prints the current request context."""
    print(f"Request: {self.request!r}")

