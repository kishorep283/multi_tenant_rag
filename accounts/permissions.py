
from rest_framework.permissions import BasePermission
from rag.models import UserProfile

class IsCompanyOwnerOrAdmin(BasePermission):
    """
    Allow access only to authenticated users
    with OWNER or ADMIN roles.
    """

    def has_permission(self, request, view):
        user = request.user

        if not user or not user.is_authenticated:
            return False

        if not hasattr(user, "profile"):
            return False

        return user.profile.role in [
            UserProfile.Role.OWNER,
            UserProfile.Role.ADMIN,
        ]

class CanManageCompanyDocuments(BasePermission):
    def has_permission(self, request, view):
        user = request.user

        if not user or not user.is_authenticated:
            return False

        if not hasattr(user, "profile"):
            return False

        return user.profile.role in [
            UserProfile.Role.OWNER,
            UserProfile.Role.ADMIN,
            UserProfile.Role.SUBADMIN,
        ]