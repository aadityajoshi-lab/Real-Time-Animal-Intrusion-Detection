from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import UserProfile


class RegisterForm(UserCreationForm):
    phone = forms.CharField(
        max_length=15,
        required=True,
        widget=forms.TextInput(attrs={"placeholder": "9XXXXXXXXX"})
    )

    class Meta:
        model = User
        fields = ("username", "phone", "password1", "password2")

    def clean_phone(self):
        phone = self.cleaned_data.get("phone", "").strip()
        if UserProfile.objects.filter(phone=phone).exists():
            raise forms.ValidationError("This phone number is already registered.")
        return phone

    def save(self, commit=True):
        user = super().save(commit=True)
        UserProfile.objects.create(
            user=user,
            phone=self.cleaned_data["phone"],
        )
        return user
