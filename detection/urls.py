from django.contrib import admin
from django.urls import path, include
from core.api_proxy import api_proxy
from core.media_views import serve_repellent_sound, serve_detection_image

urlpatterns = [
    path("admin/", admin.site.urls),
    # Direct media serving (no auth required, parent pages are protected)
    path("api/repellent_sounds/<str:filename>", serve_repellent_sound, name="serve_sound"),
    path("api/detection_images/<str:filename>", serve_detection_image, name="serve_image"),
    # API proxy for other endpoints
    path("api/<path:path>", api_proxy, name="api_proxy"),
    path("api/", api_proxy, name="api_root"),
    path("", include("core.urls")),
]
