from django.urls import path

from .views import ChunksRetrieveView, PDFUploadView, RerankerTestView, ParentDocumentView,CompanyListCreateView,UserCreateView


urlpatterns = [
	path("documents/upload/", PDFUploadView.as_view(), name="documents-upload"),
    path("documents/retrieve/", ChunksRetrieveView.as_view(), name="documents-retrieve"),
    path("documents/reranker", RerankerTestView.as_view(), name="documents-retrieve-by-id"),
    path("documents/parent", ParentDocumentView.as_view(), name="documents-parent"),
    path("companies/", CompanyListCreateView.as_view(), name="companies"),
    path("users/create/", UserCreateView.as_view(), name="users-create"),

]
