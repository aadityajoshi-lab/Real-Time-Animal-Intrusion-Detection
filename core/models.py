from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone = models.CharField(max_length=15, unique=True)
    ntfy_subscribed = models.BooleanField(default=False)
    telegram_chat_id = models.CharField(max_length=64, blank=True, default="")

    def __str__(self):
        return f"{self.user.username} ({self.phone})"


class CameraStream(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="streams")
    name = models.CharField(max_length=100)
    source_url = models.CharField(max_length=500)
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    last_started_at = models.DateTimeField(null=True, blank=True)
    last_detection_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ("user", "name")
        ordering = ["-created_at"]

    def __str__(self):
        status = "active" if self.is_active else "idle"
        return f"{self.name} ({status}) - {self.user.username}"


class DetectionEvent(models.Model):
    stream = models.ForeignKey(
        CameraStream, on_delete=models.CASCADE, related_name="detections", null=True, blank=True
    )
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="detections")
    label = models.CharField(max_length=50, db_index=True)
    confidence = models.FloatField()
    camera_id = models.CharField(max_length=100, blank=True, default="")
    image_path = models.CharField(max_length=255, blank=True, default="")
    source_url = models.CharField(max_length=500, blank=True, default="")
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    feedback = models.CharField(max_length=20, blank=True, default="")

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.label} ({self.confidence:.0%}) @ {self.timestamp:%Y-%m-%d %H:%M}"
