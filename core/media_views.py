"""
Direct views for serving media files without authentication.
These views are safe because the parent pages require login.
"""
from django.http import FileResponse, Http404, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.cache import cache_control
from pathlib import Path
import os


@csrf_exempt
@cache_control(public=True, max_age=3600)
def serve_repellent_sound(request, filename):
    """
    Serve repellent sound files directly from FastAPI backend or local storage.
    No authentication required since the control page is already protected.
    """
    # Try to proxy to FastAPI first
    import requests
    fastapi_url = "http://127.0.0.1:8001"
    
    try:
        # Proxy to FastAPI backend
        resp = requests.get(
            f"{fastapi_url}/repellent_sounds/{filename}",
            timeout=10,
            stream=True
        )
        
        if resp.status_code == 200:
            response = HttpResponse(
                content=resp.content,
                content_type="audio/mpeg"
            )
            response['Accept-Ranges'] = 'bytes'
            return response
        else:
            return HttpResponse(
                '{"error": "Sound not found"}',
                status=404,
                content_type="application/json"
            )
    except Exception as e:
        return HttpResponse(
            f'{{"error": "Audio service unavailable: {str(e)}"}}',
            status=503,
            content_type="application/json"
        )


@csrf_exempt
@cache_control(public=True, max_age=3600)
def serve_detection_image(request, filename):
    """
    Serve detection images directly from FastAPI backend.
    No authentication required since the control page is already protected.
    """
    import requests
    fastapi_url = "http://127.0.0.1:8001"
    
    try:
        resp = requests.get(
            f"{fastapi_url}/detection_images/{filename}",
            timeout=10,
            stream=True
        )
        
        if resp.status_code == 200:
            response = HttpResponse(
                content=resp.content,
                content_type=resp.headers.get("Content-Type", "image/jpeg")
            )
            return response
        else:
            raise Http404("Image not found")
    except Exception as e:
        return HttpResponse(
            f'{{"error": "Image service unavailable: {str(e)}"}}',
            status=503,
            content_type="application/json"
        )
