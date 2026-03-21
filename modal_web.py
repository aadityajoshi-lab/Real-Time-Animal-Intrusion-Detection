"""
modal_web.py - Deploy FarmGuard Django + FastAPI on Modal.com

The Django app serves HTML pages and proxies /api/* requests to the
FastAPI backend, both running inside the same Modal container.

Deploy:   cd detection && modal deploy modal_web.py
"""
import modal
import os
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "libpq-dev")
    .pip_install(
        "django==5.2.10",
        "gunicorn>=21.2.0",
        "dj-database-url>=2.1.0",
        "whitenoise>=6.6.0",
        "psycopg2-binary>=2.9.9",
        "fastapi==0.128.0",
        "uvicorn[standard]==0.40.0",
        "python-dotenv>=1.2.0",
        "requests>=2.32.0",
        "httpx>=0.27.0",
        "numpy>=2.0.0",
        "pillow>=12.0.0",
        "PyYAML>=6.0",
        "psutil>=7.0",
        "yt-dlp>=2025.12.0",
        "ultralytics>=8.4.0",
    )
    .add_local_dir(
        str(THIS_DIR),
        "/app",
        ignore=lambda p: any(
            x in str(p) for x in [
                "__pycache__", ".pyc", "db.sqlite3", "venv", ".venv",
                "telegram_state.json", "telegram_pending.json",
                "alert_feedback.json", "detection_history.json",
                "staticfiles", ".git", "best.pt", "bestn.pt",
            ]
        ),
    )
)

app = modal.App("farmguard-web", image=web_image)

detections_vol = modal.Volume.from_name(
    "farmguard-detections-volume", create_if_missing=True
)

# Add a persistent volume for SQLite database
db_vol = modal.Volume.from_name(
    "farmguard-db-volume", create_if_missing=True
)

# Add a volume for repellent sounds
repellent_sounds_vol = modal.Volume.from_name(
    "farmguard-repellent-sounds-volume", create_if_missing=True
)


@app.function(
    secrets=[modal.Secret.from_name("farmguard-secrets")],
    volumes={
        "/data/detections": detections_vol,
        "/data/db": db_vol,  # Persistent database
        "/data/repellent_sounds": repellent_sounds_vol,  # Repellent sounds
    },
    scaledown_window=300,
    timeout=3600,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="farmguard-app")
def web():
    import sys
    import subprocess
    import threading

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/backend_fastapi")

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "detection.settings")
    os.environ.setdefault("DJANGO_DEBUG", "false")
    os.environ.setdefault("DJANGO_ALLOWED_HOSTS", "*")
    
    # Use persistent SQLite database on Modal volume
    os.environ.setdefault("DATABASE_URL", "sqlite:////data/db/farmguard.db")

    import django
    django.setup()

    from django.core.management import call_command
    try:
        call_command("migrate", "--noinput", verbosity=0)
        print("[OK] Migrations applied")
        # Create cache table for sessions
        try:
            call_command("createcachetable", verbosity=0)
            print("[OK] Cache table created")
        except Exception:
            pass  # Table might already exist
        # Commit database volume after migrations
        db_vol.commit()
    except Exception as e:
        print(f"[Migration] {e}")

    # Create default user for stream management if it doesn't exist
    try:
        from django.contrib.auth.models import User
        if not User.objects.filter(username="admin").exists():
            User.objects.create_superuser("admin", "admin@farmguard.local", "farmguard2024")
            print("[OK] Default admin user created")
            db_vol.commit()
    except Exception as e:
        print(f"[User] {e}")

    try:
        call_command("collectstatic", "--noinput", verbosity=0)
        print("[OK] Static files collected")
    except Exception as e:
        print(f"[Static] {e}")

    # Start FastAPI backend on port 8001 in background thread
    def start_fastapi():
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "main:app",
             "--host", "127.0.0.1", "--port", "8001", "--workers", "1"],
            cwd="/app/backend_fastapi",
        )

    fastapi_thread = threading.Thread(target=start_fastapi, daemon=True)
    fastapi_thread.start()

    import time
    time.sleep(5)  # Give FastAPI more time to start fully
    print("[OK] FastAPI backend started on port 8001")

    from detection.asgi import application as django_app
    return django_app
