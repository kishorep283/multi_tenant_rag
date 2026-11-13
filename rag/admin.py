from django.contrib import admin
from .models import Company, Document, DocumentParent, DocumentChunk, UserProfile


@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "created_at")
    search_fields = ("name",)


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "original_filename",
        "company",
        "uploaded_by",
        "status",
        "created_at",
    )
    list_filter = ("status", "company", "created_at")
    search_fields = ("original_filename", "file_hash")
    readonly_fields = ("created_at", "updated_at")


@admin.register(DocumentParent)
class DocumentParentAdmin(admin.ModelAdmin):
    list_display = ("id", "source_filename", "company", "document", "created_at")
    list_filter = ("company",)
    search_fields = ("source_filename", "content")


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "source_filename",
        "company",
        "document",
        "parent",
        "created_at",
    )
    list_filter = ("company",)
    search_fields = ("source_filename", "content")


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "company", "role", "created_at")
    list_filter = ("role", "company")
    search_fields = ("user__username", "user__email", "company__name")

