import asyncio
import httpx
import requests
from django.http import HttpResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt

FASTAPI_URL = "http://127.0.0.1:8001"


@csrf_exempt
def api_proxy(request, path=""):
    url = f"{FASTAPI_URL}/{path}"
    headers = {k: v for k, v in request.META.items()
               if k.startswith("HTTP_") and k not in ("HTTP_HOST", "HTTP_COOKIE")}
    fwd_headers = {}
    for k, v in headers.items():
        header_name = k[5:].replace("_", "-").title()
        fwd_headers[header_name] = v

    if "CONTENT_TYPE" in request.META:
        fwd_headers["Content-Type"] = request.META["CONTENT_TYPE"]

    method = request.method.lower()
    kwargs = {
        "headers": fwd_headers,
        "params": request.GET.dict(),
        "timeout": 120,
    }

    if method in ("post", "put", "patch"):
        kwargs["data"] = request.body

    # Video streaming - return redirect hint or buffered proxy
    if path == "video" or path.startswith("video?"):
        kwargs["stream"] = True
        kwargs["timeout"] = None
        try:
            resp = getattr(requests, method)(url, **kwargs)

            def gen():
                try:
                    for chunk in resp.iter_content(chunk_size=4096):
                        yield chunk
                except Exception:
                    pass
                finally:
                    resp.close()

            streaming = StreamingHttpResponse(
                gen(),
                content_type=resp.headers.get("Content-Type", "application/octet-stream"),
            )
            streaming.status_code = resp.status_code
            return streaming
        except Exception as e:
            return HttpResponse(f'{{"error": "{e}"}}', status=502, content_type="application/json")

    # Detection images
    if path == "detection_images" or path.startswith("detection_images/"):
        kwargs["stream"] = True
        try:
            resp = getattr(requests, method)(url, **kwargs)
            django_resp = HttpResponse(
                content=resp.content,
                status=resp.status_code,
                content_type=resp.headers.get("Content-Type", "application/octet-stream"),
            )
            return django_resp
        except Exception as e:
            return HttpResponse(f'{{"error": "{e}"}}', status=502, content_type="application/json")

    # Repellent sounds
    if path == "repellent_sounds" or path.startswith("repellent_sounds/"):
        kwargs["stream"] = True
        try:
            resp = getattr(requests, method)(url, **kwargs)
            
            # If FastAPI returned an error, return proper HTTP status without redirect
            if resp.status_code >= 400:
                return HttpResponse(
                    content=resp.content,
                    status=resp.status_code,
                    content_type="application/json"
                )
            
            django_resp = HttpResponse(
                content=resp.content,
                status=resp.status_code,
                content_type=resp.headers.get("Content-Type", "audio/mpeg"),
            )
            return django_resp
        except requests.exceptions.ConnectionError:
            # FastAPI backend not available - return 503 instead of redirecting
            return HttpResponse(
                '{"error": "Audio service temporarily unavailable. Please wait a moment and try again."}',
                status=503,
                content_type="application/json"
            )
        except Exception as e:
            return HttpResponse(f'{{"error": "{e}"}}', status=502, content_type="application/json")

    try:
        resp = getattr(requests, method)(url, **kwargs)
        django_resp = HttpResponse(
            content=resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "application/json"),
        )
        return django_resp
    except requests.exceptions.ConnectionError:
        return HttpResponse(
            '{"error": "FastAPI backend not available"}',
            status=502,
            content_type="application/json",
        )
