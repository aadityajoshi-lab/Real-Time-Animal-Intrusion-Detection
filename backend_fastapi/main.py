import os
import time
import json
import base64
import threading
import uuid
from typing import Any, Dict, List, Optional
from pathlib import Path

import cv2
import numpy as np
import requests
import httpx

from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import asyncio

from dotenv import load_dotenv
from asgiref.sync import sync_to_async

# Load .env BEFORE importing modules that need env vars
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

# Bootstrap Django ORM for database access
import sys
PROJECT_DIR = BASE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "detection.settings")
import django
django.setup()

from db_helpers import (
    get_user_streams, add_stream as db_add_stream, delete_stream_by_name,
    set_stream_active_by_name, get_stream_by_name, get_user_stream_count,
    save_detection, get_detection_history, clear_detection_history,
    delete_detection, get_all_active_streams,
)

from ultralytics import YOLO
from alert_agent import dispatch_all_alerts
from telegram_bot import is_alert_suppressed

# ── Simple rate limiter ──────────────────────────────
from collections import defaultdict

class RateLimiter:
    def __init__(self):
        self._hits = defaultdict(list)

    def check(self, key: str, max_hits: int, window_seconds: int) -> bool:
        now = time.time()
        cutoff = now - window_seconds
        self._hits[key] = [t for t in self._hits[key] if t > cutoff]
        if len(self._hits[key]) >= max_hits:
            return False
        self._hits[key].append(now)
        return True

_rate = RateLimiter()

# Use Modal volume if running on Modal, otherwise local directory
if os.path.exists("/data/detections"):
    DETECTIONS_DIR = Path("/data/detections")
    print("[IMG] Using Modal volume for detections: /data/detections")
else:
    DETECTIONS_DIR = BASE_DIR / "detections"
    print("[IMG] Using local directory for detections")
    
DETECTIONS_DIR.mkdir(exist_ok=True)
HISTORY_FILE = BASE_DIR / "detection_history.json"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_IDS = os.getenv("TELEGRAM_CHAT_IDS", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").strip()
MODAL_STREAM_URL = os.getenv("MODAL_STREAM_URL", "").strip()
MODAL_WEBHOOK_SECRET = os.getenv("MODAL_WEBHOOK_SECRET", "").strip()

ALERTS_ENABLED = bool(TELEGRAM_BOT_TOKEN)

print(f"TELEGRAM configured: {bool(TELEGRAM_BOT_TOKEN)}", flush=True)
print(f"ALERTS_ENABLED: {ALERTS_ENABLED}", flush=True)
print(f"MODAL_STREAM_URL: {MODAL_STREAM_URL[:50]}..." if MODAL_STREAM_URL else "MODAL_STREAM_URL: (not set)", flush=True)
print(f"PUBLIC_BASE_URL: {PUBLIC_BASE_URL}", flush=True)


def is_probably_youtube(url: str) -> bool:
    u = (url or "").lower()
    return ("youtube.com" in u) or ("youtu.be" in u) or ("m.youtube.com" in u)


def resolve_youtube_to_direct_url(url: str) -> str:
    import yt_dlp
    ydl_opts = {
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "format": "best[ext=mp4][vcodec!=none][acodec!=none]/best[vcodec!=none]",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        direct = info.get("url")
        if direct:
            return direct
        fmts = info.get("formats") or []
        for f in reversed(fmts):
            if f.get("vcodec") == "none":
                continue
            u = f.get("url")
            if u:
                return u
        raise RuntimeError("Could not resolve YouTube URL to a direct stream URL.")


# ── Model ────────────────────────────────────────────
MODEL_FILE = BASE_DIR / "best.pt"
if not MODEL_FILE.exists():
    MODEL_FILE = BASE_DIR / "bestn.pt"
if not MODEL_FILE.exists():
    print("[WARNING] No local model found (best.pt/bestn.pt). Local inference disabled.", flush=True)
    model = None
else:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    DEVICE = 0 if HAS_CUDA else "cpu"
    print(f"CUDA available: {HAS_CUDA}, using device: {DEVICE}", flush=True)
    model = YOLO(str(MODEL_FILE))
    if HAS_CUDA:
        model.to("cuda:0")

CONF_THRES = 0.60
IOU_THRES = 0.55
IMG_SIZE = 896
FRAME_SKIP = 1
JPEG_QUALITY = 100

DANGEROUS = {
    "elephant", "tiger", "nilgai", "monkey", "bear",
    "jackal", "leopard", "wild_boar", "gaur",
}
DANGEROUS_MIN_CONF = 0.65

# Safety tips, repellent methods, and alert details per animal
ANIMAL_SAFETY_INFO = {
    "elephant": {
        "severity": 9,
        "emoji": "\U0001f418",
        "sound": "elephant_bee.mp3",
        "tips": [
            "Do NOT run \u2013 elephants can reach 40 km/h",
            "Move downwind so it cannot smell you",
            "Find a large tree or solid structure to hide behind",
            "Never come between a mother and calf",
            "Avoid direct eye contact \u2013 it is seen as a challenge",
        ],
        "repellent": (
            "Bee-hive fence (most effective), chilli-smoke bombs, "
            "loud drums/firecrackers, bright spotlights at night"
        ),
        "danger": "Can charge without warning. Crop raiding peaks at night.",
    },
    "tiger": {
        "severity": 10,
        "emoji": "\U0001f405",
        "sound": "tiger_firecracker.mp3",
        "tips": [
            "NEVER turn your back \u2013 tigers attack from behind",
            "Make yourself look big; raise arms and jacket above head",
            "Shout loudly and firmly; bang pots or tins",
            "Back away slowly facing the tiger",
            "If attacked, fight back \u2013 hit eyes and nose",
        ],
        "repellent": (
            "Firecrackers, trip-wire alarms, bright flashing lights, "
            "human-voice recordings, face masks worn on back of head"
        ),
        "danger": "Ambush predator. Most attacks happen at dawn/dusk.",
    },
    "nilgai": {
        "severity": 6,
        "emoji": "\U0001f402",
        "sound": "nilgai_scare.mp3",
        "tips": [
            "Do not approach \u2013 males can weigh up to 300 kg and charge",
            "Keep vehicle headlights on; nilgai freeze on roads at night",
            "Secure crops before dusk \u2013 they feed mostly at night",
            "Keep safe distance; they stampede when startled in herds",
            "Report recurring raids to your local Forest Department",
        ],
        "repellent": (
            "Solar-powered electric fencing (most effective), chilli-tobacco rope, "
            "bio-fencing with citrus/thorny hedges, reflective tape, loud crackers"
        ),
        "danger": "Major crop raider. Herds can destroy entire fields overnight.",
    },
    "monkey": {
        "severity": 5,
        "emoji": "\U0001f412",
        "sound": "monkey_scare.mp3",
        "tips": [
            "Do NOT feed or show food \u2013 increases aggression",
            "Hide all food and close bags/containers immediately",
            "Avoid showing teeth (smiling) \u2013 seen as a threat display",
            "If approached, stand tall and make a loud, firm noise",
            "Carry a stick or umbrella as a visual deterrent",
        ],
        "repellent": (
            "Langur-call recordings (very effective), rubber slingshots to startle, "
            "netting over fruit crops, reflective CDs, guard dogs"
        ),
        "danger": "Crop raiders. Can bite and carry rabies. Travel in large troops.",
    },
    "bear": {
        "severity": 8,
        "emoji": "\U0001f43b",
        "sound": "bear_horn.mp3",
        "tips": [
            "Stay calm; do NOT run \u2013 bears can outrun humans",
            "Speak in a low, calm voice so the bear knows you are human",
            "Slowly back away while avoiding eye contact",
            "If a sloth bear attacks, fight back; do NOT play dead",
            "Protect your face and neck with arms if knocked down",
        ],
        "repellent": (
            "Bear spray (capsaicin), air horns, electric fencing around apiaries, "
            "motion-activated lights and sirens, raised machans for watchmen"
        ),
        "danger": "Sloth bears are unpredictable. Most attacks are surprise encounters.",
    },
    "jackal": {
        "severity": 5,
        "emoji": "\U0001f43a",
        "sound": "jackal_scare.mp3",
        "tips": [
            "Do not leave small livestock or poultry unattended at night",
            "Make loud noise \u2013 jackals are naturally wary of humans",
            "Keep dustbins sealed; food waste attracts packs",
            "Walk with a torch at night in known jackal areas",
            "If approached, shout and wave arms \u2013 they will flee",
        ],
        "repellent": (
            "Guard dogs (very effective), foxlights (flashing lights), "
            "secure livestock pens at night, predator-urine sprays, loud sirens"
        ),
        "danger": "Targets poultry and small livestock. Can carry rabies.",
    },
    "leopard": {
        "severity": 9,
        "emoji": "\U0001f406",
        "sound": "leopard_scare.mp3",
        "tips": [
            "Do NOT crouch or bend down \u2013 you look like prey",
            "Make loud noise: shout, clap, bang metal objects",
            "Wave arms to appear larger",
            "Keep children and pets indoors at dusk and dawn",
            "Use a bright flashlight at night \u2013 leopards avoid light",
        ],
        "repellent": (
            "Bright floodlights, loud sirens, livestock guard dogs, "
            "thorny-hedge boma fencing, motion-sensor alarms with strobe lights"
        ),
        "danger": "Nocturnal and stealthy. Targets livestock and dogs at night.",
    },
    "wild_boar": {
        "severity": 7,
        "emoji": "\U0001f417",
        "sound": "boar_siren.mp3",
        "tips": [
            "Climb a tree or get to high ground immediately",
            "Do NOT corner or provoke \u2013 boars charge when threatened",
            "If charged, dodge sideways; boars have poor turning radius",
            "Keep dogs leashed \u2013 dogs provoke boar charges",
            "Avoid areas with fresh rooting or mud wallows",
        ],
        "repellent": (
            "Solar electric fencing, chilli-grease rope around crops, "
            "ultrasonic repellers, bright flashing lights, tin-can noise lines"
        ),
        "danger": "Destroys crops overnight. Males have razor-sharp tusks.",
    },
    "gaur": {
        "severity": 8,
        "emoji": "\U0001f402",
        "sound": "gaur_scare.mp3",
        "tips": [
            "Maintain at least 30 m distance \u2013 gaur are the largest wild cattle",
            "Do NOT block their escape path; they charge when cornered",
            "Stay inside your vehicle if you spot one on a road",
            "Avoid eye contact and move away slowly and quietly",
            "Never get between a cow and calf \u2013 mothers are highly aggressive",
        ],
        "repellent": (
            "Torch/spotlight at night, loud crackers in their direction, "
            "trenches around fields, thick bio-fences, raised machans with sirens"
        ),
        "danger": "Weighs up to 1,500 kg. Can flip vehicles if provoked.",
    },
}

# Build repellent sound map from safety info
REPELLENT_SOUNDS = {
    animal: info["sound"]
    for animal, info in ANIMAL_SAFETY_INFO.items()
    if info.get("sound")
}

# Use Modal volume if running on Modal, otherwise local directory
if os.path.exists("/data/repellent_sounds"):
    REPELLENT_SOUNDS_DIR = Path("/data/repellent_sounds")
    print("[REPELLENT] Using Modal volume for sounds: /data/repellent_sounds")
else:
    REPELLENT_SOUNDS_DIR = BASE_DIR / "repellent_sounds"
    print("[REPELLENT] Using local directory for sounds")
    
REPELLENT_SOUNDS_DIR.mkdir(exist_ok=True)
COOLDOWN_SECONDS = 900
last_alert: Dict[str, float] = {}


def can_send_alert(key: str, cooldown: int = COOLDOWN_SECONDS) -> bool:
    now = time.time()
    prev = last_alert.get(key, 0.0)
    if now - prev >= cooldown:
        last_alert[key] = now
        return True
    return False


# ── Detection history helpers ────────────────────────
def _load_history() -> List[Dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_history(history: List[Dict]):
    HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


_detection_counter: Dict[str, int] = {}


def _get_detection_number(label: str) -> int:
    """Get incrementing detection number per animal type."""
    global _detection_counter
    if not _detection_counter:
        # Initialize from existing history
        history = _load_history()
        for rec in history:
            lbl = rec.get("label", "")
            _detection_counter[lbl] = _detection_counter.get(lbl, 0) + 1
    _detection_counter[label] = _detection_counter.get(label, 0) + 1
    return _detection_counter[label]


def save_detection_record(
    label: str, confidence: float, source: str, camera_id: str,
    frame: Optional[np.ndarray] = None, image_b64: Optional[str] = None,
    modal_image_filename: Optional[str] = None,
) -> dict:
    ts = time.time()
    detection_num = _get_detection_number(label)
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
    
    # Image naming: animalname_detectionNo_time.jpg
    base_name = f"{label}_{detection_num}_{time_str}"
    img_filename = ""

    # Save image from various sources
    if frame is not None:
        img_filename = f"{base_name}.jpg"
        img_path = DETECTIONS_DIR / img_filename
        cv2.imwrite(str(img_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        print(f"[IMG] Saved from frame: {img_filename}")
    elif image_b64 and len(image_b64) > 10:  
        img_filename = f"{base_name}.jpg"
        img_path = DETECTIONS_DIR / img_filename
        try:
            img_bytes = base64.b64decode(image_b64)
            img_path.write_bytes(img_bytes)
            # Commit Modal volume if available
            if str(DETECTIONS_DIR).startswith("/data/"):
                try:
                    from modal import Volume
                    vol = Volume.from_name("farmguard-detections-volume")
                    vol.commit()
                except Exception:
                    pass
            print(f"[IMG] Saved from base64: {img_filename} ({len(img_bytes)} bytes)")
        except Exception as e:
            print(f"[!] Failed to save image: {e}")
            img_filename = ""
    elif modal_image_filename:
        # Use the Modal filename directly if it exists on disk
        modal_path = DETECTIONS_DIR / modal_image_filename
        if modal_path.exists():
            # Rename to our naming convention
            new_path = DETECTIONS_DIR / f"{base_name}.jpg"
            try:
                import shutil
                shutil.copy2(str(modal_path), str(new_path))
                img_filename = f"{base_name}.jpg"
                print(f"[IMG] Copied from Modal: {modal_image_filename} -> {img_filename}")
            except Exception:
                img_filename = modal_image_filename
                print(f"[IMG] Using Modal filename as-is: {modal_image_filename}")
        else:
            img_filename = modal_image_filename
            print(f"[IMG] Modal file reference: {modal_image_filename}")
    else:
        print(f"[IMG] No image data provided (frame={frame is not None}, b64_len={len(image_b64) if image_b64 else 0})")

    record = {
        "id": f"{label}_{detection_num}",
        "label": label,
        "confidence": round(confidence, 3),
        "source": source or "",
        "camera_id": camera_id or "",
        "image": img_filename,
        "timestamp": ts,
        "time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
    }

    history = _load_history()
    history.insert(0, record)
    history = history[:200]
    _save_history(history)
    return record


# ── Multi-camera stream manager ──────────────────────
class StreamSlot:
    def __init__(self, cam_id: str, source: str, phone: str = ""):
        self.cam_id = cam_id
        self.source = source
        self.phone = phone
        self.running = False
        self.fps = 0.0
        self.last_detections: List[Dict] = []
        self.frame_id = 0
        self.ts = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None

    def to_dict(self):
        return {
            "cam_id": self.cam_id,
            "source": self.source,
            "phone": self.phone,
            "running": self.running,
            "fps": round(self.fps, 2),
            "frame_id": self.frame_id,
            "detections": self.last_detections,
        }


streams: Dict[str, StreamSlot] = {}
streams_lock = threading.Lock()


def _open_capture(source: str) -> cv2.VideoCapture:
    src = (source or "").strip()
    if src.startswith("camera:"):
        idx = int(src.split(":", 1)[1])
        c = cv2.VideoCapture(idx)
    else:
        if is_probably_youtube(src):
            src = resolve_youtube_to_direct_url(src)
        c = cv2.VideoCapture(src)
    if not c.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return c


def _detect_on_frame(frame: np.ndarray):
    if model is None:
        return frame.copy(), []
    results = model.predict(
        source=frame, imgsz=IMG_SIZE, conf=CONF_THRES,
        iou=IOU_THRES, verbose=False, device=DEVICE,
    )
    r = results[0]
    detections = []
    annotated = frame.copy()
    if r.boxes is not None and len(r.boxes) > 0:
        names = r.names
        for b in r.boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            label = (names.get(cls_id, str(cls_id)) or "").lower().strip()
            detections.append({"label": label, "confidence": round(conf, 3), "box": xyxy})
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{label} {conf:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return annotated, detections


_alert_counter = 0


def _save_incident_mapping(incident_id: str, image_filename: str):
    """Save mapping of incident_id to image_filename for false positive handling."""
    import json
    from pathlib import Path
    
    mapping_file = Path(__file__).parent / "incident_mappings.json"
    
    try:
        if mapping_file.exists():
            with open(mapping_file, "r") as f:
                mappings = json.load(f)
        else:
            mappings = {}
        
        mappings[incident_id] = {
            "image_filename": image_filename,
            "timestamp": time.time(),
        }
        
        with open(mapping_file, "w") as f:
            json.dump(mappings, f, indent=2)
    except Exception as e:
        print(f"[INCIDENT] Failed to save incident mapping: {e}")


def _dispatch_alert_bg(label: str, confidence_pct: int, message: str,
                       image_b64: Optional[str] = None,
                       repellent_sound: Optional[str] = None,
                       severity: int = 7):
    global _alert_counter
    _alert_counter += 1
    incident_id = f"{label}_{_alert_counter}"
    
    # Check if alerts are suppressed for this incident
    if is_alert_suppressed(incident_id):
        print(f"[ALERT_BG] Alert suppressed for incident {incident_id} due to false positive feedback")
        return
    
    print(f"[ALERT_BG] Starting alert dispatch: label={label}, conf={confidence_pct}%, severity={severity}, has_image={bool(image_b64)}, sound={repellent_sound}")
    try:
        result = dispatch_all_alerts(
            incident_id=incident_id, animal=label,
            severity=severity, confidence_pct=confidence_pct,
            message=message, image_b64=image_b64,
        )
        print(f"[ALERT_BG] Alert dispatched successfully: {result}", flush=True)
    except Exception as e:
        print(f"[ALERT_BG] Alert dispatch FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()


def _stream_detect_loop(slot: StreamSlot):
    frame_count = 0
    fps_frames = 0
    fps_start = time.time()
    print(f"[{slot.cam_id}] Detection thread started for {slot.source}", flush=True)

    while not slot.stop_event.is_set():
        if slot.cap is None:
            time.sleep(0.03)
            continue
        ok, frame = slot.cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        do_infer = (FRAME_SKIP <= 1) or (frame_count % FRAME_SKIP == 0)
        if do_infer:
            annotated, detections = _detect_on_frame(frame)
        else:
            annotated, detections = frame, slot.last_detections

        fps_frames += 1
        now = time.time()
        if now - fps_start >= 1.0:
            slot.fps = fps_frames / (now - fps_start)
            fps_frames = 0
            fps_start = now

        slot.frame_id += 1
        slot.ts = int(now * 1000)
        slot.last_detections = detections

        for d in detections:
            lbl = (d.get("label") or "").lower().strip()
            conf = float(d.get("confidence") or 0)
            if lbl in DANGEROUS and conf >= DANGEROUS_MIN_CONF:
                if can_send_alert(f"{slot.cam_id}:{lbl}"):
                    conf_pct = int(round(conf * 100))
                    msg = (f"FarmGuard Alert\n"
                           f"{lbl.replace('_',' ').capitalize()} detected on camera {slot.cam_id}.\n"
                           f"Confidence: {conf_pct}%. Stay safe.")
                    img_b64 = None
                    ok_enc, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ok_enc:
                        img_b64 = base64.b64encode(jpg.tobytes()).decode("utf-8")
                    save_detection_record(lbl, conf, slot.source, slot.cam_id, frame=frame)
                    # Also save to DB with image path
                    try:
                        record = _load_history()[0]  # Most recent record
                        save_detection(
                            user_id=1, label=lbl, confidence=conf,
                            camera_id=slot.cam_id, image_path=record.get("image", ""),
                            source_url=slot.source,
                        )
                    except Exception:
                        pass
                    threading.Thread(
                        target=_dispatch_alert_bg,
                        args=(lbl, conf_pct, msg, img_b64), daemon=True,
                    ).start()

        with slot.frame_lock:
            slot.latest_frame = annotated

    if slot.cap:
        try:
            slot.cap.release()
        except Exception:
            pass
    slot.cap = None
    print(f"[{slot.cam_id}] Detection thread stopped", flush=True)


# ── FastAPI app ──────────────────────────────────────
app = FastAPI(title="Farm Intrusion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.mount("/detection_images", StaticFiles(directory=str(DETECTIONS_DIR)), name="detection_images")

# Background task for polling Telegram updates
telegram_poll_task = None

async def telegram_poll_loop():
    """Background task to poll Telegram for new messages."""
    from telegram_bot import poll_telegram_updates
    while True:
        try:
            if TELEGRAM_BOT_TOKEN:
                poll_telegram_updates()
        except Exception as e:
            print(f"[TelegramPoll] Error: {e}")
        await asyncio.sleep(2)  # Poll every 2 seconds

@app.on_event("startup")
async def startup_event():
    global telegram_poll_task
    if TELEGRAM_BOT_TOKEN:
        telegram_poll_task = asyncio.create_task(telegram_poll_loop())
        print("[TelegramBot] Polling task started", flush=True)

@app.on_event("shutdown")
async def shutdown_event():
    global telegram_poll_task
    if telegram_poll_task:
        telegram_poll_task.cancel()
        print("[TelegramBot] Polling task stopped", flush=True)

# Legacy single-stream state (kept for backward-compat with existing frontend)
state: Dict[str, Any] = {
    "running": False, "started_at": None, "fps": 0.0,
    "last_detections": [], "source": None, "phone": None,
    "frame_id": 0, "ts": 0, "active_cam": None,
}

GPU_LOCK = asyncio.Lock()
ACTIVE_KEY = None


async def claim_gpu(key: str):
    global ACTIVE_KEY
    if ACTIVE_KEY is not None and ACTIVE_KEY != key:
        raise HTTPException(status_code=429, detail={"error": "GPU is busy", "active_source": ACTIVE_KEY})
    if ACTIVE_KEY is None:
        await GPU_LOCK.acquire()
        ACTIVE_KEY = key


_GPU_LOCK_SYNC = threading.Lock()

def _claim_gpu_sync(key: str):
    global ACTIVE_KEY
    if ACTIVE_KEY is not None and ACTIVE_KEY != key:
        raise HTTPException(status_code=429, detail={"error": "GPU is busy", "active_source": ACTIVE_KEY})
    if ACTIVE_KEY is None:
        _GPU_LOCK_SYNC.acquire()
        ACTIVE_KEY = key


def release_gpu(key: str):
    global ACTIVE_KEY
    if ACTIVE_KEY == key:
        ACTIVE_KEY = None
        if GPU_LOCK.locked():
            GPU_LOCK.release()


# ── Endpoints ────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "FastAPI YOLO server running", "streams": len(streams)}


@app.post("/start")
async def start_detection(payload: Dict[str, Any] = Body(default={})):
    source = (payload.get("source") or "camera:0").strip()
    phone = (payload.get("phone") or "").strip()
    cam_id = payload.get("cam_id", "default")

    await claim_gpu(cam_id)

    if MODAL_STREAM_URL:
        state["running"] = True
        state["source"] = source
        state["phone"] = phone
        state["active_cam"] = cam_id
        state["started_at"] = time.time()
        state["frame_id"] = 0

        import urllib.parse
        modal_video_params = urllib.parse.urlencode({
            "source": source, "camera_id": cam_id, "phone": phone,
            "webhook_url": f"{PUBLIC_BASE_URL}/api/report_incident",
            "webhook_secret": MODAL_WEBHOOK_SECRET,
        })
        modal_video_url = f"{MODAL_STREAM_URL}?{modal_video_params}"

        return {
            "ok": True, "running": True, "source": source,
            "cam_id": cam_id, "mode": "modal",
            "video_url": modal_video_url,
        }

    with streams_lock:
        if cam_id in streams and streams[cam_id].running:
            return {"ok": True, "running": True, "message": "Already running", "cam_id": cam_id}

        slot = StreamSlot(cam_id, source, phone)
        slot.cap = _open_capture(source)
        slot.running = True
        slot.stop_event.clear()
        slot.thread = threading.Thread(target=_stream_detect_loop, args=(slot,), daemon=True)
        slot.thread.start()
        streams[cam_id] = slot

    state["running"] = True
    state["source"] = source
    state["phone"] = phone
    state["active_cam"] = cam_id
    state["started_at"] = time.time()
    state["frame_id"] = 0

    return {"ok": True, "running": True, "source": source, "cam_id": cam_id}


@app.post("/stop")
async def stop_detection(payload: Dict[str, Any] = Body(default={})):
    cam_id = payload.get("cam_id", "default")
    stop_modal = payload.get("stop_modal", True)

    with streams_lock:
        slot = streams.get(cam_id)
        if slot and slot.running:
            slot.running = False
            slot.stop_event.set()

    release_gpu(cam_id)

    if cam_id == state.get("active_cam"):
        state["running"] = False
        state["last_detections"] = []
        state["fps"] = 0.0

    modal_result = None
    if stop_modal and MODAL_STREAM_URL:
        try:
            resp = requests.post(
                MODAL_STOP_ALL_URL,
                params={"camera_id": cam_id},
                timeout=10,
            )
            modal_result = resp.json()
        except Exception as e:
            modal_result = {"error": str(e)}

    return {"ok": True, "running": False, "cam_id": cam_id, "modal": modal_result}


@app.get("/status")
def status():
    cam_id = state.get("active_cam", "default")
    slot = streams.get(cam_id)
    return {
        "running": state.get("running", False) or (slot.running if slot else False),
        "fps": round(slot.fps, 2) if slot else 0,
        "source": state.get("source") or (slot.source if slot else None),
        "phone": state.get("phone"),
        "frame_id": slot.frame_id if slot else 0,
        "ts": slot.ts if slot else 0,
        "cam_id": cam_id,
        "alerts_enabled": ALERTS_ENABLED,
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN),
        "dangerous": list(DANGEROUS),
        "dangerous_min_conf": DANGEROUS_MIN_CONF,
        "cooldown_seconds": COOLDOWN_SECONDS,
        "modal_configured": bool(MODAL_STREAM_URL),
        "modal_url": MODAL_STREAM_URL[:50] + "..." if len(MODAL_STREAM_URL) > 50 else MODAL_STREAM_URL,
    }


@app.get("/latest")
def latest(cam_id: str = ""):
    cid = cam_id or state.get("active_cam", "default")
    
    if MODAL_STREAM_URL and state.get("running"):
        try:
            modal_latest_url = "https://wildanimaldetection--farmguard-vision-engine-get-latest.modal.run"
            resp = requests.get(modal_latest_url, params={"camera_id": cid}, timeout=5)
            modal_data = resp.json()
            
            # Get FPS from background status
            fps = 0
            try:
                status_url = "https://wildanimaldetection--farmguard-vision-engine-get-bg-status.modal.run"
                status_resp = requests.get(status_url, params={"camera_id": cid}, timeout=3)
                status_data = status_resp.json()
                if status_data.get("status") == "running" and status_data.get("frames"):
                    elapsed = status_data.get("elapsed", 1)
                    fps = status_data["frames"] / max(elapsed, 1) if elapsed > 0 else 0
            except Exception:
                pass
            
            detections = []
            if modal_data.get("label"):
                detections.append({
                    "label": modal_data.get("label", ""),
                    "confidence": modal_data.get("conf", 0),
                    "bbox": modal_data.get("bbox", []),
                })
            
            return {
                "running": True,
                "fps": round(fps, 1),
                "detections": detections,
                "frame_id": modal_data.get("frame_count", 0),
                "ts": modal_data.get("ts", time.time()),
            }
        except Exception as e:
            print(f"[!] Modal latest error: {e}")
            return {"running": True, "fps": 0, "detections": [], "frame_id": 0, "ts": time.time()}
            
            return JSONResponse({
                "running": True,
                "fps": 15,
                "detections": detections,
                "source": state.get("source"),
                "frame_id": int(modal_data.get("ts", 0) * 1000) % 100000,
                "ts": modal_data.get("ts", 0),
                "cam_id": cid,
                "modal": True,
            })
        except Exception as e:
            pass
    
    slot = streams.get(cid)
    if slot:
        return JSONResponse({
            "running": slot.running, "fps": round(slot.fps, 2),
            "detections": slot.last_detections, "source": slot.source,
            "frame_id": slot.frame_id, "ts": slot.ts, "cam_id": cid,
        })
    return JSONResponse({
        "running": state.get("running", False), "fps": 0, "detections": [],
        "source": state.get("source"), "frame_id": 0, "ts": 0, "cam_id": cid,
    })


def _mjpeg_gen(cam_id: str):
    slot = streams.get(cam_id)
    if not slot:
        return
    while slot.running:
        with slot.frame_lock:
            frame = None if slot.latest_frame is None else slot.latest_frame.copy()
        if frame is None:
            time.sleep(0.02)
            continue
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        time.sleep(0.005)


def _modal_proxy_gen(source: str, cam_id: str = "", phone: str = ""):
    """Proxy an MJPEG stream from Modal's GPU endpoint."""
    import urllib.parse
    params = urllib.parse.urlencode({
        "source": source, "camera_id": cam_id, "phone": phone,
        "webhook_url": f"{PUBLIC_BASE_URL}/api/report_incident",
        "webhook_secret": MODAL_WEBHOOK_SECRET,
    })
    url = f"{MODAL_STREAM_URL}?{params}"
    print(f"Modal proxy -> {url[:120]}", flush=True)
    try:
        with httpx.Client(timeout=None) as client:
            with client.stream("GET", url) as resp:
                for chunk in resp.iter_bytes(chunk_size=8192):
                    yield chunk
    except Exception as e:
        print(f"Modal proxy stream ended: {e}", flush=True)


@app.get("/video")
def video(cam_id: str = ""):
    cid = cam_id or state.get("active_cam", "default")

    if MODAL_STREAM_URL and state.get("running"):
        import urllib.parse
        source = state.get("source", "")
        phone = state.get("phone", "")
        params = urllib.parse.urlencode({
            "source": source, "camera_id": cid, "phone": phone,
            "webhook_url": f"{PUBLIC_BASE_URL}/api/report_incident",
            "webhook_secret": MODAL_WEBHOOK_SECRET,
        })
        redirect_url = f"{MODAL_STREAM_URL}?{params}"
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=redirect_url)

    slot = streams.get(cid)
    if not slot or not slot.running:
        return Response(status_code=404)
    return StreamingResponse(
        _mjpeg_gen(cid),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Multi-cam management ─────────────────────────────

@app.get("/streams")
async def list_streams(user_id: int = 0):
    if user_id:
        db_streams = await sync_to_async(get_user_streams, thread_sensitive=False)(user_id)
    else:
        db_streams = await sync_to_async(get_user_streams, thread_sensitive=False)(0)

    result = []
    for s in db_streams:
        cam_id = s["name"]
        slot = streams.get(cam_id)
        result.append({
            "id": s["id"],
            "cam_id": cam_id,
            "source": s["source_url"],
            "running": (slot.running if slot else False) or s["is_active"],
            "fps": round(slot.fps, 2) if slot and slot.running else 0,
            "frame_id": slot.frame_id if slot else 0,
            "detections": slot.last_detections if slot else [],
            "is_active": s["is_active"],
        })

    with streams_lock:
        for cam_id, slot in streams.items():
            if not any(r["cam_id"] == cam_id for r in result):
                result.append(slot.to_dict())

    return {"streams": result}


MAX_STREAMS_PER_USER = 5

@app.post("/streams/add")
async def add_stream(payload: Dict[str, Any] = Body(...)):
    source = (payload.get("source") or "").strip()
    cam_id = (payload.get("cam_id") or f"cam-{len(streams)+1}").strip()
    phone = (payload.get("phone") or "").strip()
    user_id = int(payload.get("user_id", 0))

    if not source:
        raise HTTPException(400, "source is required")

    if not source.startswith(("http://", "https://", "rtsp://", "camera:")):
        raise HTTPException(400, "Invalid stream URL scheme. Use http/https/rtsp.")

    if len(cam_id) > 100:
        raise HTTPException(400, "Stream name too long (max 100 chars)")

    rate_key = f"add_stream:{user_id or 'anon'}"
    if not _rate.check(rate_key, max_hits=60, window_seconds=3600):
        raise HTTPException(429, "Rate limit exceeded. Try again later.")

    if user_id:
        # Wrap synchronous Django ORM calls with sync_to_async
        # Ensure user exists (create default if needed for Modal SQLite)
        try:
            count = await sync_to_async(get_user_stream_count, thread_sensitive=False)(user_id)
        except Exception as e:
            # If user doesn't exist, create a default one
            from django.contrib.auth.models import User
            def ensure_user():
                if not User.objects.filter(id=user_id).exists():
                    User.objects.create(id=user_id, username=f"user{user_id}", password="!")
                return get_user_stream_count(user_id)
            count = await sync_to_async(ensure_user, thread_sensitive=False)()
        
        if count >= MAX_STREAMS_PER_USER:
            raise HTTPException(400, f"Maximum {MAX_STREAMS_PER_USER} streams per user")
        try:
            db_result = await sync_to_async(db_add_stream, thread_sensitive=False)(user_id, cam_id, source)
        except Exception as e:
            raise HTTPException(500, f"Database error: {e}")
    else:
        db_result = {"id": 0, "name": cam_id, "source_url": source, "created": True}

    if MODAL_STREAM_URL:
        return {"ok": True, "cam_id": cam_id, "source": source, "db": db_result}

    await claim_gpu(cam_id)

    with streams_lock:
        if cam_id in streams and streams[cam_id].running:
            return {"ok": True, "message": "Already running", "cam_id": cam_id}
        slot = StreamSlot(cam_id, source, phone)
        slot.cap = _open_capture(source)
        slot.running = True
        slot.stop_event.clear()
        slot.thread = threading.Thread(target=_stream_detect_loop, args=(slot,), daemon=True)
        slot.thread.start()
        streams[cam_id] = slot

    return {"ok": True, "cam_id": cam_id, "source": source, "db": db_result}


@app.post("/streams/{cam_id}/stop")
async def stop_stream(cam_id: str):
    with streams_lock:
        slot = streams.get(cam_id)
        if not slot:
            raise HTTPException(404, "Stream not found")
        slot.running = False
        slot.stop_event.set()
    release_gpu(cam_id)
    return {"ok": True, "cam_id": cam_id}


@app.delete("/streams/{cam_id}")
async def remove_stream(cam_id: str, user_id: int = 0):
    with streams_lock:
        slot = streams.pop(cam_id, None)
        if slot and slot.running:
            slot.running = False
            slot.stop_event.set()
    release_gpu(cam_id)

    if user_id:
        await sync_to_async(delete_stream_by_name, thread_sensitive=False)(user_id, cam_id)

    return {"ok": True, "cam_id": cam_id, "removed": True}


# ── Detection history ────────────────────────────────

@app.get("/history")
async def get_history(limit: int = 50, offset: int = 0, user_id: int = 0):
    def _clean_image_field(records):
        """Strip modal: prefix and fix image paths for display."""
        for r in records:
            img = r.get("image", "") or ""
            if img.startswith("modal:"):
                img = img[6:]  # Strip "modal:" prefix
            r["image"] = img
        return records

    try:
        result = await sync_to_async(get_detection_history, thread_sensitive=False)(user_id=user_id or None, limit=limit, offset=offset)
        _clean_image_field(result["records"])
        return {"ok": True, "total": result["total"], "offset": offset, "records": result["records"]}
    except Exception:
        history = _load_history()
        total = len(history)
        page = history[offset:offset + limit]
        _clean_image_field(page)
        return {"ok": True, "total": total, "offset": offset, "records": page}


@app.delete("/history/{record_id}")
async def delete_history_record(record_id: str, user_id: int = 0):
    try:
        rid = int(record_id)
        deleted = await sync_to_async(delete_detection, thread_sensitive=False)(rid, user_id=user_id or None)
        if deleted:
            return {"ok": True, "deleted": record_id}
    except (ValueError, TypeError):
        pass
    history = _load_history()
    new_history = [r for r in history if r.get("id") != record_id]
    if len(new_history) == len(history):
        raise HTTPException(404, "Record not found")
    img_file = DETECTIONS_DIR / f"{record_id}.jpg"
    if img_file.exists():
        img_file.unlink()
    _save_history(new_history)
    return {"ok": True, "deleted": record_id}


@app.delete("/history")
async def clear_history(user_id: int = 0):
    try:
        await sync_to_async(clear_detection_history, thread_sensitive=False)(user_id=user_id or None)
    except Exception:
        pass
    _save_history([])
    for f in DETECTIONS_DIR.glob("*.jpg"):
        f.unlink(missing_ok=True)
    return {"ok": True, "cleared": True}


# ── Single image detect (unchanged interface) ────────

@app.post("/detect_image_json")
def detect_image_json(payload: Dict[str, Any] = Body(...)):
    url = (payload.get("url") or "").strip()
    if not url:
        return JSONResponse({"ok": False, "error": "Missing url"}, status_code=400)
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"ok": False, "error": "Could not decode image"}, status_code=400)
    _, detections = _detect_on_frame(img)
    return {"ok": True, "detections": detections}


@app.post("/detect_image_render")
def detect_image_render(payload: Dict[str, Any] = Body(...)):
    url = (payload.get("url") or "").strip()
    if not url:
        return JSONResponse({"ok": False, "error": "Missing url"}, status_code=400)
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"ok": False, "error": "Could not decode image"}, status_code=400)
    annotated, _ = _detect_on_frame(img)
    ok, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return JSONResponse({"ok": False, "error": "Could not encode result"}, status_code=500)
    return Response(content=jpg.tobytes(), media_type="image/jpeg")


# ── Webhook for Modal GPU inference ──────────────────

@app.post("/report_incident")
async def report_incident(payload: Dict[str, Any] = Body(...)):
    secret = (payload.get("secret") or "").strip()
    if MODAL_WEBHOOK_SECRET and secret != MODAL_WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    label = (payload.get("label") or "").lower().strip()
    confidence = float(payload.get("confidence") or 0)
    image_b64 = payload.get("image_b64", "")
    image_filename = payload.get("image_filename", "")  # Filename from Modal volume
    camera_id = payload.get("camera_id", "")
    timestamp = payload.get("timestamp", time.time())

    print(f"[WEBHOOK] Received: label={label}, conf={confidence}, camera={camera_id}, image_len={len(image_b64)}, filename={image_filename}")
    
    # Store the detection record and save image with proper naming
    record = save_detection_record(label, confidence, "", camera_id, image_b64=image_b64, modal_image_filename=image_filename)
    saved_image = record.get("image", "")

    # Also save to database (best-effort, don't fail the webhook)
    try:
        await sync_to_async(save_detection, thread_sensitive=False)(
            user_id=1,
            label=label,
            confidence=confidence,
            camera_id=camera_id,
            image_path=saved_image,
            source_url="",
        )
    except Exception:
        pass

    # Get repellent sound for this animal
    repellent_sound = REPELLENT_SOUNDS.get(label, "")
    safety_info = ANIMAL_SAFETY_INFO.get(label, {})
    
    response_data = {"ok": True, "label": label, "confidence": confidence}
    
    if repellent_sound:
        sound_path = REPELLENT_SOUNDS_DIR / repellent_sound
        if sound_path.exists():
            response_data["repellent_sound"] = f"/api/repellent_sounds/{repellent_sound}"
            print(f"[REPELLENT] Sound available: {repellent_sound}")
        else:
            print(f"[REPELLENT] Sound file not found: {sound_path}")

    if label in DANGEROUS and confidence >= DANGEROUS_MIN_CONF:
        if can_send_alert(label):
            conf_pct = int(round(confidence * 100))
            emoji = safety_info.get("emoji", "\U0001f6a8")
            tips = safety_info.get("tips", [])
            repellent_methods = safety_info.get("repellent", "")
            danger_note = safety_info.get("danger", "")
            sev = safety_info.get("severity", 7)

            tips_text = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(tips))

            msg = (
                f"{emoji} FarmGuard Alert\n"
                f"{label.replace('_', ' ').title()} detected!\n"
                f"Confidence: {conf_pct}%\n\n"
            )
            if danger_note:
                msg += f"\u26a0\ufe0f Danger: {danger_note}\n\n"
            if tips_text:
                msg += f"\U0001f6e1\ufe0f Safety Tips:\n{tips_text}\n\n"
            if repellent_methods:
                msg += f"\U0001f4a1 Repellent Methods:\n  {repellent_methods}\n\n"
            if repellent_sound:
                control_url = f"{PUBLIC_BASE_URL}/repellent-control/"
                msg += f"\U0001f50a Play deterrent sound:\n  {control_url}\n\n"
            msg += (
                f"Camera: {camera_id or 'default'}\n"
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Stay safe!"
            )

            print(f"[ALERT] Triggering alert for {label} (conf={conf_pct}%, image_len={len(image_b64)}, sound={repellent_sound})")
            
            # Save incident mapping for false positive handling
            # Use _alert_counter + 1 because _dispatch_alert_bg will increment it
            incident_id = f"{label}_{_alert_counter + 1}"
            _save_incident_mapping(incident_id, saved_image)
            
            threading.Thread(
                target=_dispatch_alert_bg,
                args=(label, conf_pct, msg, image_b64, repellent_sound, sev), daemon=True,
            ).start()
        else:
            print(f"[ALERT] Alert cooldown active for {label}")
    else:
        print(f"[ALERT] No alert: label={label} in DANGEROUS={label in DANGEROUS}, conf={confidence} >= {DANGEROUS_MIN_CONF}={confidence >= DANGEROUS_MIN_CONF}")

    return response_data


# ── User alert subscription ──────────────────────────

@app.post("/subscribe_alerts")
def subscribe_alerts(payload: Dict[str, Any] = Body(...)):
    """Register Telegram chat for alerts."""
    telegram_chat_id = (payload.get("telegram_chat_id") or "").strip()
    result = {}

    if telegram_chat_id and TELEGRAM_BOT_TOKEN:
        current_ids = set(c.strip() for c in TELEGRAM_CHAT_IDS.split(",") if c.strip())
        current_ids.add(telegram_chat_id)
        result["telegram"] = {
            "chat_id": telegram_chat_id,
            "status": "registered",
        }

    return {"ok": True, "result": result}


@app.get("/repellent_sounds/{filename}")
@app.head("/repellent_sounds/{filename}")
async def get_repellent_sound(filename: str):
    """Serve repellent sound files."""
    from fastapi.responses import FileResponse
    import os
    
    # Security: prevent directory traversal
    filename = os.path.basename(filename)
    sound_path = REPELLENT_SOUNDS_DIR / filename
    
    if not sound_path.exists():
        raise HTTPException(status_code=404, detail="Sound file not found")
    
    return FileResponse(sound_path, media_type="audio/mpeg")


@app.get("/animal_safety/{animal}")
async def get_animal_safety(animal: str):
    """Return safety tips and repellent info for a specific animal."""
    info = ANIMAL_SAFETY_INFO.get(animal.lower().strip())
    if not info:
        return {"found": False, "animal": animal}
    return {
        "found": True,
        "animal": animal,
        "severity": info["severity"],
        "emoji": info["emoji"],
        "tips": info["tips"],
        "repellent": info["repellent"],
        "danger": info["danger"],
        "sound": info.get("sound", ""),
    }


@app.get("/animal_safety")
async def list_animal_safety():
    """Return safety info for all dangerous animals."""
    return {
        animal: {
            "severity": info["severity"],
            "emoji": info["emoji"],
            "tips": info["tips"],
            "repellent": info["repellent"],
            "danger": info["danger"],
            "sound": info.get("sound", ""),
        }
        for animal, info in ANIMAL_SAFETY_INFO.items()
    }


@app.get("/telegram/status")
async def telegram_status():
    """Check Telegram bot configuration and subscribers."""
    from telegram_bot import get_subscriber_chat_ids, get_bot_token
    
    token = get_bot_token()
    chat_ids = get_subscriber_chat_ids()
    
    return {
        "bot_token_configured": bool(token),
        "bot_token_preview": token[:20] + "..." if token else None,
        "subscribers": chat_ids,
        "subscriber_count": len(chat_ids),
        "telegram_chat_ids_env": os.getenv("TELEGRAM_CHAT_IDS", ""),
    }


@app.post("/test_telegram")
async def test_telegram():
    """Test endpoint to send a Telegram alert."""
    try:
        msg = f"🧪 Test Alert from FarmGuard\n\nThis is a test message sent at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        result = dispatch_all_alerts(
            incident_id=0, animal="test", severity=5,
            confidence_pct=99, message=msg,
        )
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/test_alert")
def test_alert(payload: Dict[str, Any] = Body(default={})):
    msg = (payload.get("message") or "TEST ALERT from FarmGuard").strip()
    try:
        result = dispatch_all_alerts(
            incident_id=0, animal="test", severity=5,
            confidence_pct=99, message=msg,
        )
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Background Detection (Modal) ─────────────────────

MODAL_BG_START_URL = "https://wildanimaldetection--farmguard-vision-engine-start-background.modal.run"
MODAL_BG_STOP_URL = "https://wildanimaldetection--farmguard-vision-engine-stop-background.modal.run"
MODAL_BG_STATUS_URL = "https://wildanimaldetection--farmguard-vision-engine-get-bg-status.modal.run"
MODAL_STOP_ALL_URL = "https://wildanimaldetection--farmguard-vision-engine-stop-all.modal.run"
MODAL_ALL_STATUS_URL = "https://wildanimaldetection--farmguard-vision-engine-get-all-status.modal.run"


@app.post("/start_background")
def start_background_detection(payload: Dict[str, Any] = Body(...)):
    """Start background detection on Modal with live video stream (continues 24/7)."""
    if not MODAL_STREAM_URL:
        return {"ok": False, "error": "Modal not configured"}

    source = (payload.get("source") or "").strip()
    camera_id = payload.get("camera_id", "default")
    phone = (payload.get("phone") or "").strip()
    duration_hours = int(payload.get("duration_hours", 24))  # Default 24 hours

    webhook_url = f"{PUBLIC_BASE_URL}/api/report_incident"

    try:
        # Start background detection
        resp = requests.post(
            MODAL_BG_START_URL,
            params={
                "source": source,
                "camera_id": camera_id,
                "phone": phone,
                "webhook_url": webhook_url,
                "webhook_secret": MODAL_WEBHOOK_SECRET,
                "duration_hours": duration_hours,
            },
            timeout=30,
        )
        result = resp.json()
        
        # Also generate stream URL for live viewing
        if result.get("ok"):
            video_url = (
                f"{MODAL_STREAM_URL}?"
                f"source={requests.utils.quote(source)}&"
                f"camera_id={requests.utils.quote(camera_id)}&"
                f"phone={requests.utils.quote(phone)}&"
                f"webhook_url={requests.utils.quote(webhook_url)}&"
                f"webhook_secret={requests.utils.quote(MODAL_WEBHOOK_SECRET or '')}"
            )
            result["video_url"] = video_url
            result["stream_type"] = "background_with_video"
        
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/stop_background")
def stop_background_detection(payload: Dict[str, Any] = Body(default={})):
    """Stop background detection on Modal."""
    camera_id = payload.get("camera_id", "default")

    try:
        resp = requests.post(
            MODAL_BG_STOP_URL,
            params={"camera_id": camera_id},
            timeout=10,
        )
        return resp.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/stop_all_modal")
def stop_all_modal(payload: Dict[str, Any] = Body(default={})):
    """
    EMERGENCY STOP - Stops ALL Modal GPU processing immediately.
    This stops all video streams and background detections to save GPU costs.
    """
    results = {"local": [], "modal": None}

    with streams_lock:
        for cam_id, slot in list(streams.items()):
            if slot.running:
                slot.stop_event.set()
                slot.running = False
                results["local"].append(cam_id)

    state["running"] = False

    if MODAL_STREAM_URL:
        try:
            resp = requests.post(MODAL_STOP_ALL_URL, timeout=15)
            results["modal"] = resp.json()
        except Exception as e:
            results["modal"] = {"error": str(e)}

    return {"ok": True, "message": "STOP ALL executed", "results": results}


@app.post("/clear_modal_flags")
def clear_modal_flags():
    """Clear all stop flags on Modal before starting new streams."""
    if not MODAL_STREAM_URL:
        return {"ok": False, "error": "Modal not configured"}

    try:
        resp = requests.get(MODAL_ALL_STATUS_URL, params={"clear_flags": "true"}, timeout=10)
        return resp.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/modal_status")
def get_modal_status():
    """Get status of all Modal streams and background tasks."""
    if not MODAL_STREAM_URL:
        return {"ok": False, "error": "Modal not configured"}

    try:
        resp = requests.get(MODAL_ALL_STATUS_URL, timeout=10)
        return resp.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/bg_status")
def get_background_status(camera_id: str = "default"):
    """Get background detection status from Modal."""
    try:
        resp = requests.get(
            MODAL_BG_STATUS_URL,
            params={"camera_id": camera_id},
            timeout=10,
        )
        return resp.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Telegram Bot Endpoints ───────────────────────────

from telegram_bot import (
    poll_telegram_updates,
    get_all_subscribers,
    handle_telegram_update,
    send_telegram_message,
    get_subscriber_chat_ids,
)


@app.post("/telegram/webhook")
async def telegram_webhook(payload: Dict[str, Any] = Body(...)):
    """
    Webhook endpoint for Telegram bot updates.
    Set this URL in Telegram using:
    https://api.telegram.org/bot<TOKEN>/setWebhook?url=<PUBLIC_URL>/api/telegram/webhook
    """
    try:
        result = handle_telegram_update(payload)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/telegram/poll")
def telegram_poll():
    """
    Poll for Telegram updates (alternative to webhook).
    Call this periodically to process new messages.
    """
    try:
        results = poll_telegram_updates()
        return {"ok": True, "processed": results}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/telegram/subscribers")
def telegram_subscribers():
    """Get list of all Telegram subscribers."""
    try:
        subscribers = get_all_subscribers()
        return {
            "ok": True,
            "count": len(subscribers),
            "subscribers": [
                {
                    "username": s.get("username"),
                    "first_name": s.get("first_name"),
                    "active": s.get("active", True),
                }
                for s in subscribers
            ],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/telegram/setup_webhook")
def setup_telegram_webhook():
    """
    Set up Telegram webhook to receive bot messages.
    This tells Telegram to send all bot messages to your server.
    """
    if not TELEGRAM_BOT_TOKEN:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN not configured"}
    
    webhook_url = f"{PUBLIC_BASE_URL}/api/telegram/webhook"
    
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook",
            json={"url": webhook_url},
            timeout=10,
        )
        result = resp.json()
        return {
            "ok": result.get("ok", False),
            "webhook_url": webhook_url,
            "telegram_response": result,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/telegram/remove_webhook")
def remove_telegram_webhook():
    """Remove Telegram webhook (switch to polling mode)."""
    if not TELEGRAM_BOT_TOKEN:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN not configured"}
    
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook",
            timeout=10,
        )
        return resp.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}
