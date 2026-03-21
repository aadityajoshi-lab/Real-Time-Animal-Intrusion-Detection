"""
db_helpers.py - Database access for FastAPI backend.

Uses Django ORM by bootstrapping Django settings.
This avoids raw SQL and keeps things consistent with the Django models.
"""
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "detection.settings")

import django

django.setup()

from django.contrib.auth.models import User
from core.models import CameraStream, DetectionEvent, UserProfile
from django.utils import timezone


# ── CameraStream CRUD ──────────────────────────────

def get_user_streams(user_id: int):
    return list(
        CameraStream.objects.filter(user_id=user_id)
        .values("id", "name", "source_url", "is_active", "created_at", "last_started_at", "last_detection_at")
    )


def get_all_active_streams():
    return list(
        CameraStream.objects.filter(is_active=True)
        .values("id", "user_id", "name", "source_url", "last_started_at")
    )


def add_stream(user_id: int, name: str, source_url: str):
    stream, created = CameraStream.objects.update_or_create(
        user_id=user_id,
        name=name,
        defaults={"source_url": source_url},
    )
    return {
        "id": stream.id,
        "name": stream.name,
        "source_url": stream.source_url,
        "is_active": stream.is_active,
        "created": created,
    }


def delete_stream(user_id: int, stream_id: int):
    deleted, _ = CameraStream.objects.filter(id=stream_id, user_id=user_id).delete()
    return deleted > 0


def delete_stream_by_name(user_id: int, name: str):
    deleted, _ = CameraStream.objects.filter(name=name, user_id=user_id).delete()
    return deleted > 0


def set_stream_active(stream_id: int, active: bool):
    CameraStream.objects.filter(id=stream_id).update(
        is_active=active,
        last_started_at=timezone.now() if active else None,
    )


def set_stream_active_by_name(name: str, user_id: int, active: bool):
    CameraStream.objects.filter(name=name, user_id=user_id).update(
        is_active=active,
        last_started_at=timezone.now() if active else None,
    )


def get_stream_by_name(user_id: int, name: str):
    try:
        s = CameraStream.objects.get(user_id=user_id, name=name)
        return {
            "id": s.id, "name": s.name, "source_url": s.source_url,
            "is_active": s.is_active, "user_id": s.user_id,
        }
    except CameraStream.DoesNotExist:
        return None


def get_user_stream_count(user_id: int) -> int:
    return CameraStream.objects.filter(user_id=user_id).count()


# ── DetectionEvent CRUD ────────────────────────────

def save_detection(
    user_id: int,
    label: str,
    confidence: float,
    camera_id: str = "",
    image_path: str = "",
    source_url: str = "",
    stream_id: int = None,
):
    event = DetectionEvent.objects.create(
        user_id=user_id,
        stream_id=stream_id,
        label=label,
        confidence=confidence,
        camera_id=camera_id,
        image_path=image_path,
        source_url=source_url,
    )
    if stream_id:
        CameraStream.objects.filter(id=stream_id).update(last_detection_at=timezone.now())
    return event.id


def get_detection_history(user_id: int = None, limit: int = 50, offset: int = 0):
    qs = DetectionEvent.objects.all()
    if user_id:
        qs = qs.filter(user_id=user_id)
    total = qs.count()
    records = list(
        qs[offset : offset + limit].values(
            "id", "label", "confidence", "camera_id",
            "image_path", "source_url", "timestamp", "feedback",
        )
    )
    for r in records:
        r["time_str"] = r["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if r["timestamp"] else ""
        r["image"] = r.pop("image_path", "")
    return {"total": total, "records": records}


def clear_detection_history(user_id: int = None):
    qs = DetectionEvent.objects.all()
    if user_id:
        qs = qs.filter(user_id=user_id)
    count, _ = qs.delete()
    return count


def delete_detection(event_id: int, user_id: int = None):
    qs = DetectionEvent.objects.filter(id=event_id)
    if user_id:
        qs = qs.filter(user_id=user_id)
    deleted, _ = qs.delete()
    return deleted > 0


# ── User helpers ───────────────────────────────────

def get_user_by_id(user_id: int):
    try:
        u = User.objects.get(id=user_id)
        return {"id": u.id, "username": u.username}
    except User.DoesNotExist:
        return None


def get_user_id_by_api_key(api_key: str):
    """Placeholder for future API key auth. For now, returns None."""
    return None
