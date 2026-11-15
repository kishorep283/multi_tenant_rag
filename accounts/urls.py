from django.urls import path
from .views import CompanyRegistrationView,UserCreateView
from rest_framework.authtoken.views import obtain_auth_token
urlpatterns = [
    path(
        "register/",
        CompanyRegistrationView.as_view(),
        name="company-register",
    ),
    path("login/", obtain_auth_token, name="login"),
    path("users/", UserCreateView.as_view(), name="user-create"),
]