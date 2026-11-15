
from rest_framework import generics
from rest_framework.permissions import AllowAny
from rest_framework.authentication import TokenAuthentication, SessionAuthentication
from rest_framework.permissions import IsAuthenticated
from .serializers import CompanyRegistrationSerializer
from .permissions import IsCompanyOwnerOrAdmin
from .serializers import UserCreateSerializer

class CompanyRegistrationView(generics.CreateAPIView):
    serializer_class = CompanyRegistrationSerializer
    permission_classes = [AllowAny]

class UserCreateView(generics.CreateAPIView):
    serializer_class = UserCreateSerializer

    authentication_classes = [
        TokenAuthentication,
        SessionAuthentication,
    ]

    permission_classes = [
        IsAuthenticated,
        IsCompanyOwnerOrAdmin,
    ]
