from rest_framework import serializers
from .models import Company,UserProfile
from django.contrib.auth import get_user_model
from django.db import transaction

User = get_user_model()

class CompanySerializer(serializers.ModelSerializer):
    class Meta:
        model = Company
        fields = ["id", "name", "created_at"]
        read_only_fields = ["id", "created_at"]


class UserCreateSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    email = serializers.EmailField(required=False, allow_blank=True)
    password = serializers.CharField(
        write_only=True,
        min_length=8,
        trim_whitespace=False,
    )
    company_id = serializers.PrimaryKeyRelatedField(
        queryset=Company.objects.all(),
        source="company",
    )

    def validate_username(self, value):
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError(
                "A user with this username already exists."
            )
        return value

    def create(self, validated_data):
        company = validated_data.pop("company")

        with transaction.atomic():
            user = User.objects.create_user(**validated_data)

            UserProfile.objects.create(
                user=user,
                company=company,
                role=UserProfile.Role.MEMBER,
            )

        return user

    def to_representation(self, instance):
        return {
            "id": instance.id,
            "username": instance.username,
            "email": instance.email,
            "company_id": instance.profile.company_id,
            "role": instance.profile.role,
        }
