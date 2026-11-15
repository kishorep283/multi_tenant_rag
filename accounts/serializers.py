
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.db import transaction
from rest_framework import serializers

from rag.models import Company, UserProfile

User = get_user_model()


class CompanyRegistrationSerializer(serializers.Serializer):
    company_name = serializers.CharField(max_length=255)

    username = serializers.CharField(max_length=150)

    email = serializers.EmailField(
        required=False,
        allow_blank=True,
    )

    password = serializers.CharField(
        write_only=True,
        min_length=8,
        trim_whitespace=False,
    )

    def validate_company_name(self, value):
        value = value.strip()
        if not value:
            raise serializers.ValidationError(
                "Company name cannot be empty."
            )

        if Company.objects.filter(name=value).exists():
            raise serializers.ValidationError(
                "A company with this name already exists."
            )

        return value
    

    def validate_username(self, value):
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError(
                "A user with this username already exists."
            )
        return value

    def validate_password(self, value):
        validate_password(value)
        return value

    def create(self, validated_data):
        company_name = validated_data.pop("company_name")

        with transaction.atomic():

            # 1. Create company
            company = Company.objects.create(
                name=company_name
            )

            # 2. Create the first user
            user = User.objects.create_user(
                **validated_data
            )

            # 3. Assign the user as company OWNER
            UserProfile.objects.create(
                user=user,
                company=company,
                role=UserProfile.Role.OWNER,
            )

        return user

    def to_representation(self, instance):
        return {
            "user_id": instance.id,
            "username": instance.username,
            "email": instance.email,
            "company": {
                "id": instance.profile.company_id,
                "name": instance.profile.company.name,
            },
            "role": instance.profile.role,
        }




class UserCreateSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)

    email = serializers.EmailField(
        required=False,
        allow_blank=True,
    )

    password = serializers.CharField(
        write_only=True,
        min_length=8,
        trim_whitespace=False,
    )

    role = serializers.ChoiceField(
        choices=UserProfile.Role.choices,
        default=UserProfile.Role.MEMBER,
    )

    def validate_username(self, value):
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError(
                "A user with this username already exists."
            )
        return value

    def validate_password(self, value):
        validate_password(value)
        return value

    def validate_role(self, value):
        requesting_user = self.context["request"].user
        requester_role = requesting_user.profile.role

        allowed_roles = {
            UserProfile.Role.OWNER: [
                UserProfile.Role.ADMIN,
                UserProfile.Role.SUBADMIN,
                UserProfile.Role.MEMBER,
            ],
            UserProfile.Role.ADMIN: [
                UserProfile.Role.SUBADMIN,
                UserProfile.Role.MEMBER,
            ],
            UserProfile.Role.SUBADMIN: [
                UserProfile.Role.MEMBER,
            ],
        }

        if value not in allowed_roles.get(requester_role, []):
            raise serializers.ValidationError(
                "You are not allowed to assign this role."
            )

        return value

    def create(self, validated_data):
        company = self.context["request"].user.profile.company

        role = validated_data.pop("role")

        with transaction.atomic():
            user = User.objects.create_user(
                **validated_data
            )

            UserProfile.objects.create(
                user=user,
                company=company,
                role=role,
            )

        return user

    def to_representation(self, instance):
        return {
            "user_id": instance.id,
            "username": instance.username,
            "email": instance.email,
            "company_id": instance.profile.company_id,
            "role": instance.profile.role,
        }
