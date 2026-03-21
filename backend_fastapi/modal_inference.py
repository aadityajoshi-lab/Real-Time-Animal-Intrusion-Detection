"""
modal_inference.py - Serverless GPU inference engine on Modal.com.
Deploy with: modal deploy modal_inference.py
Stream URL will be printed after deployment.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git", "ffmpeg")
    .pip_install(
        "ultralytics>=8.1",
        "opencv-python-headless",
        "numpy",
        "fastapi",
        "yt-dlp",
        "requests",
    )
)

vol = modal.Volume.from_name("farmguard-weights-volume", create_if_missing=True)
detections_vol = modal.Volume.from_name("farmguard-detections-volume", create_if_missing=True)

app = modal.App("farmguard-vision-engine", image=image)

MODEL_DIR = "/mnt/weights/weights"
MODEL_PATH = f"{MODEL_DIR}/best.pt"

global_state = modal.Dict.from_name("farmguard-engine-state", create_if_missing=True)


def is_probably_youtube(url: str) -> bool:
    u = (url or "").lower()
    return ("youtube.com" in u) or ("youtu.be" in u) or ("m.youtube.com" in u)


def resolve_youtube_to_direct_url(url: str) -> str:
    import yt_dlp
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "best[ext=mp4][vcodec!=none][acodec!=none]/best[vcodec!=none]",
        "extractor_args": {"youtube": {"player_client": ["android", "ios"]}},
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return info.get("url")
        except Exception as e:
            raise RuntimeError(f"YouTube resolution failed: {e}")


DETECTIONS_DIR = "/data/detections"

@app.cls(
    gpu="T4",
    volumes={"/mnt/weights": vol, DETECTIONS_DIR: detections_vol},
    scaledown_window=60,
    max_containers=3,
    timeout=3600,
)
class VideoInferenceEngine:
    @modal.enter()
    def load_model(self):
        import os
        import torch
        from ultralytics import YOLO

        print("[*] FarmGuard Container initializing... Loading YOLO Model")
        
        # Check CUDA availability
        print(f"[+] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[+] CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"[+] CUDA version: {torch.version.cuda}")
        
        # Reload volumes to get latest files
        vol.reload()
        detections_vol.reload()
        print(f"[+] Volumes reloaded")

        # List volume contents to verify model exists
        if os.path.exists(MODEL_DIR):
            files = os.listdir(MODEL_DIR)
            print(f"[+] Files in {MODEL_DIR}: {files}")
        else:
            print(f"[!] WARNING: {MODEL_DIR} does not exist!")

        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # Size in MB
            print(f"[+] Model file found: {MODEL_PATH} ({file_size:.2f} MB)")
            self.model = YOLO(MODEL_PATH)
            print(f"[+] Custom model loaded from MOUNTED VOLUME: {MODEL_PATH}")
        else:
            # List files to debug
            import subprocess
            result = subprocess.run(["ls", "-lh", MODEL_DIR], capture_output=True, text=True)
            print(f"[DEBUG] Contents of {MODEL_DIR}:\n{result.stdout}")
            raise RuntimeError(
                f"Custom model not found at {MODEL_PATH}. "
                "Upload best.pt to Modal volume: "
                "modal volume put farmguard-weights-volume best.pt /weights/best.pt"
            )

        # Move model to GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        print(f"[+] Model moved to device: {device}")
        
        self.class_names = self.model.names or {}
        print(f"[+] Model classes: {list(self.class_names.values())[:20]}...")
        print(f"[+] Total classes: {len(self.class_names)}")
        print("[✓] Model initialization complete!")

    def _infer(
        self,
        source: str,
        camera_id: str = "",
        phone: str = "",
        webhook_url: str = "",
        webhook_secret: str = "",
    ):
        import cv2
        import time
        import requests
        import base64
        import numpy as np
        import threading
        from collections import deque

        DANGEROUS = {
            "elephant", "tiger", "nilgai", "monkey", "bear",
            "jackal", "leopard", "wild_boar", "gaur",
        }
        CONF_THRES = 0.60
        ALERT_CONF = 0.65
        COOLDOWN = 900
        last_alert = {}

        def can_alert(label):
            now = time.time()
            if label not in last_alert or (now - last_alert[label]) > COOLDOWN:
                last_alert[label] = now
                return True
            return False

        def send_webhook(label, conf, images_b64):
            # Save full-quality detection image to volume FIRST
            img_filename = ""
            if images_b64:
                try:
                    import os as _os
                    _os.makedirs(DETECTIONS_DIR, exist_ok=True)
                    img_filename = f"{camera_id}_{int(time.time())}_{label}.jpg"
                    img_path = f"{DETECTIONS_DIR}/{img_filename}"
                    img_bytes = base64.b64decode(images_b64[0])
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                    detections_vol.commit()
                    print(f"[+] Image saved to volume: {img_path}")
                except Exception as e:
                    print(f"[X] Image save error: {e}")
                    img_filename = ""

            # Send webhook with image filename (not full base64)
            if webhook_url:
                try:
                    # Create small thumbnail for webhook
                    thumbnail_b64 = ""
                    if images_b64 and len(images_b64) > 0:
                        try:
                            img_bytes = base64.b64decode(images_b64[0])
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            # Resize to max 400px width while maintaining aspect ratio
                            h, w = img.shape[:2]
                            if w > 400:
                                ratio = 400 / w
                                new_w, new_h = 400, int(h * ratio)
                                img = cv2.resize(img, (new_w, new_h))
                            # Compress heavily for thumbnail
                            _, thumb_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                            thumbnail_b64 = base64.b64encode(thumb_jpg.tobytes()).decode('utf-8')
                        except Exception as e:
                            print(f"[!] Thumbnail creation failed: {e}")
                    
                    payload = {
                        "camera_id": camera_id,
                        "phone": phone,
                        "label": label,
                        "confidence": float(conf),
                        "image_b64": thumbnail_b64,  # Small thumbnail
                        "image_filename": img_filename,  # Full-res filename on Modal
                        "timestamp": time.time(),
                    }
                    if webhook_secret:
                        payload["secret"] = webhook_secret
                    response = requests.post(webhook_url, json=payload, timeout=5)
                    if response.status_code == 200:
                        print(f"[+] Webhook sent: {label} to {webhook_url}")
                    else:
                        print(f"[!] Webhook failed: {response.status_code}")
                except Exception as e:
                    print(f"[X] Webhook error: {e}")

        src = source
        if is_probably_youtube(src):
            src = resolve_youtube_to_direct_url(src)

        print(f"[*] Starting inference for camera: {camera_id or 'unknown'}")
        print(f"[*] Source: {src[:100]}...")
        print(f"[*] Model device: {next(self.model.model.parameters()).device}")
        print(f"[*] Using GPU: {str(next(self.model.model.parameters()).device) != 'cpu'}")

        global_state[f"stream_{camera_id}"] = {"status": "running", "source": source, "ts": time.time()}
        global_state[f"stop_{camera_id}"] = False

        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[X] Failed to open stream: {src}")
            yield b"--frame\r\nContent-Type: text/plain\r\n\r\nError: Stream offline.\r\n"
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame_buffer = deque(maxlen=30)
        frame_count = 0

        try:
            while cap.isOpened():
                stop_all = global_state.get("stop_all", False)
                stop_cam = global_state.get(f"stop_{camera_id}", False)
                if stop_all or stop_cam:
                    print(f"[*] Stop signal received for camera: {camera_id}")
                    break

                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                frame_count += 1
                annotated = frame.copy()

                results = self.model.predict(
                    source=frame, imgsz=896, conf=CONF_THRES, verbose=False, device="cuda"
                )[0]

                if results.boxes is not None and len(results.boxes) > 0:
                    names = results.names or self.class_names
                    for box in results.boxes:
                        c_id = int(box.cls[0])
                        label = (names.get(c_id, str(c_id)) or "").lower().strip()
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)

                        color = (0, 255, 0)
                        if label in DANGEROUS:
                            color = (0, 255, 255)
                            if conf >= ALERT_CONF:
                                color = (0, 0, 255)

                        cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 3)
                        cv2.putText(
                            annotated,
                            f"{label.upper()} {conf:.2f}",
                            (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 4,
                        )
                        cv2.putText(
                            annotated,
                            f"{label.upper()} {conf:.2f}",
                            (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1,
                        )

                        if label in DANGEROUS and conf >= ALERT_CONF and can_alert(label):
                            print(f"[!] THREAT [{camera_id}]: {label} ({conf:.2f})")
                            snapshot_buffer = list(frame_buffer)
                            indices = [-1, -7, -14, -21, -28]
                            sequence_b64 = []

                            for idx in indices:
                                try:
                                    f_enc = snapshot_buffer[idx] if len(snapshot_buffer) > abs(idx) else snapshot_buffer[0]
                                    _, enc = cv2.imencode(".jpg", f_enc, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                                    sequence_b64.append(base64.b64encode(enc).decode("utf-8"))
                                except Exception:
                                    continue

                            if not sequence_b64:
                                _, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                                sequence_b64 = [base64.b64encode(enc).decode("utf-8")]

                            latest_info = {
                                "camera_id": camera_id,
                                "label": label,
                                "conf": conf,
                                "ts": time.time(),
                                "bbox": xyxy.tolist(),
                                "frame_count": frame_count,
                            }
                            if camera_id:
                                global_state[f"latest_{camera_id}"] = latest_info
                            global_state["latest"] = latest_info

                            threading.Thread(
                                target=send_webhook,
                                args=(label, conf, sequence_b64),
                                daemon=True,
                            ).start()

                succ, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if succ:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"

                frame_buffer.append(frame.copy())

        finally:
            cap.release()
            frame_buffer.clear()
            try:
                global_state[f"stop_{camera_id}"] = False
                global_state[f"stream_{camera_id}"] = {"status": "stopped", "ts": time.time()}
            except Exception:
                pass  # Suppress deadlock errors on cleanup
            print(f"[-] Inference stopped for camera: {camera_id or 'unknown'}")

    @modal.fastapi_endpoint(method="GET", label="farmguard-stream")
    def stream_video(
        self,
        source: str,
        camera_id: str = "",
        phone: str = "",
        webhook_url: str = "",
        webhook_secret: str = "",
    ):
        from fastapi.responses import StreamingResponse

        print(f"[*] Stream request received:")
        print(f"    Camera ID: {camera_id}")
        print(f"    Source: {source[:100]}...")
        print(f"    Webhook: {webhook_url}")

        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Camera-ID": camera_id,
        }

        return StreamingResponse(
            self._infer(source, camera_id, phone, webhook_url, webhook_secret),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers=headers,
        )

    @modal.method()
    def run_background_detection(
        self,
        source: str,
        camera_id: str = "",
        phone: str = "",
        webhook_url: str = "",
        webhook_secret: str = "",
        duration_seconds: int = 3600,
    ):
        """Run detection in background without video streaming. Sends webhooks on detection."""
        import cv2
        import time
        import requests
        import base64
        import numpy as np
        from collections import deque

        DANGEROUS = {
            "elephant", "tiger", "nilgai", "monkey", "bear",
            "jackal", "leopard", "wild_boar", "gaur",
        }
        CONF_THRES = 0.60
        ALERT_CONF = 0.65
        COOLDOWN = 900
        last_alert = {}

        def can_alert(label):
            now = time.time()
            if label not in last_alert or (now - last_alert[label]) > COOLDOWN:
                last_alert[label] = now
                return True
            return False

        def send_webhook(label, conf, images_b64):
            # Save full-quality detection image to volume FIRST
            img_filename = ""
            if images_b64:
                try:
                    import os as _os
                    _os.makedirs(DETECTIONS_DIR, exist_ok=True)
                    img_filename = f"{camera_id}_{int(time.time())}_{label}.jpg"
                    img_path = f"{DETECTIONS_DIR}/{img_filename}"
                    img_bytes = base64.b64decode(images_b64[0])
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                    detections_vol.commit()
                    print(f"[+] Image saved to volume: {img_path}")
                except Exception as e:
                    print(f"[X] Image save error: {e}")
                    img_filename = ""

            # Send webhook
            if not webhook_url:
                return
            try:
                # Send a small thumbnail (first image, compressed)
                thumbnail_b64 = ""
                if images_b64 and len(images_b64) > 0:
                    # Take first image and resize to thumbnail
                    try:
                        img_bytes = base64.b64decode(images_b64[0])
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        # Resize to max 400px width
                        h, w = img.shape[:2]
                        if w > 400:
                            ratio = 400 / w
                            new_w, new_h = 400, int(h * ratio)
                            img = cv2.resize(img, (new_w, new_h))
                        _, thumb_jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                        thumbnail_b64 = base64.b64encode(thumb_jpg.tobytes()).decode('utf-8')
                    except Exception as e:
                        print(f"[!] Thumbnail creation failed: {e}")
                
                payload = {
                    "camera_id": camera_id,
                    "phone": phone,
                    "label": label,
                    "confidence": float(conf),
                    "image_b64": thumbnail_b64,  # Small thumbnail
                    "image_filename": img_filename,  # Full-res filename on Modal
                    "timestamp": time.time(),
                }
                if webhook_secret:
                    payload["secret"] = webhook_secret
                response = requests.post(webhook_url, json=payload, timeout=10)
                print(f"[+] Webhook sent: {label} ({response.status_code})")
            except Exception as e:
                print(f"[X] Webhook error: {e}")

        src = source
        if is_probably_youtube(src):
            src = resolve_youtube_to_direct_url(src)

        print(f"[*] Background detection started for camera: {camera_id}")
        print(f"[*] Source: {src[:100]}...")
        print(f"[*] Duration: {duration_seconds}s")

        global_state[f"bg_{camera_id}"] = {
            "status": "running",
            "source": source,
            "started_at": time.time(),
            "duration": duration_seconds,
        }

        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            global_state[f"bg_{camera_id}"] = {"status": "error", "error": "Failed to open stream"}
            return {"ok": False, "error": "Failed to open stream"}

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame_buffer = deque(maxlen=30)
        frame_count = 0
        start_time = time.time()
        detections_count = 0

        try:
            while cap.isOpened() and (time.time() - start_time) < duration_seconds:
                bg_state = global_state.get(f"bg_{camera_id}", {})
                if bg_state.get("status") == "stop_requested":
                    print(f"[*] Stop requested for camera: {camera_id}")
                    break

                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                frame_count += 1
                results = self.model.predict(
                    source=frame, imgsz=896, conf=CONF_THRES, verbose=False, device="cuda"
                )[0]

                if results.boxes is not None and len(results.boxes) > 0:
                    names = results.names or self.class_names
                    for box in results.boxes:
                        c_id = int(box.cls[0])
                        label = (names.get(c_id, str(c_id)) or "").lower().strip()
                        conf = float(box.conf[0])

                        if label in DANGEROUS and conf >= ALERT_CONF and can_alert(label):
                            detections_count += 1
                            print(f"[!] THREAT [{camera_id}]: {label} ({conf:.2f})")
                            
                            snapshot_buffer = list(frame_buffer)
                            sequence_b64 = []
                            for idx in [-1, -7, -14]:
                                try:
                                    f_enc = snapshot_buffer[idx] if len(snapshot_buffer) > abs(idx) else frame
                                    _, enc = cv2.imencode(".jpg", f_enc, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                                    sequence_b64.append(base64.b64encode(enc).decode("utf-8"))
                                except Exception:
                                    continue

                            if not sequence_b64:
                                _, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                                sequence_b64 = [base64.b64encode(enc).decode("utf-8")]

                            global_state["latest"] = {
                                "camera_id": camera_id,
                                "label": label,
                                "conf": conf,
                                "ts": time.time(),
                            }
                            send_webhook(label, conf, sequence_b64)

                frame_buffer.append(frame.copy())

                global_state[f"bg_{camera_id}"] = {
                    "status": "running",
                    "source": source,
                    "started_at": start_time,
                    "elapsed": time.time() - start_time,
                    "frames": frame_count,
                    "detections": detections_count,
                }

        finally:
            cap.release()
            try:
                global_state[f"bg_{camera_id}"] = {
                    "status": "stopped",
                    "source": source,
                    "frames": frame_count,
                    "detections": detections_count,
                    "elapsed": time.time() - start_time,
                }
            except Exception:
                pass  # Suppress deadlock errors on cleanup
            print(f"[-] Background detection stopped for camera: {camera_id}")

        return {
            "ok": True,
            "camera_id": camera_id,
            "frames_processed": frame_count,
            "detections": detections_count,
            "elapsed_seconds": time.time() - start_time,
        }


@app.function()
@modal.fastapi_endpoint()
def get_latest(camera_id: str = ""):
    """Poll latest detection from the Modal global state."""
    if camera_id:
        key = f"latest_{camera_id}"
        return global_state.get(key, {"label": None, "ts": 0, "camera_id": camera_id})
    return global_state.get("latest", {"label": None, "ts": 0})


@app.function()
@modal.fastapi_endpoint()
def get_bg_status(camera_id: str = "default"):
    """Get background detection status for a camera."""
    key = f"bg_{camera_id}"
    return global_state.get(key, {"status": "not_running", "camera_id": camera_id})


@app.function()
@modal.fastapi_endpoint(method="POST")
def start_background(
    source: str,
    camera_id: str = "default",
    phone: str = "",
    webhook_url: str = "",
    webhook_secret: str = "",
    duration_hours: int = 1,
):
    """Start background detection (runs even when browser is closed)."""
    engine = VideoInferenceEngine()
    engine.run_background_detection.spawn(
        source=source,
        camera_id=camera_id,
        phone=phone,
        webhook_url=webhook_url,
        webhook_secret=webhook_secret,
        duration_seconds=duration_hours * 3600,
    )
    return {
        "ok": True,
        "message": f"Background detection started for {duration_hours} hour(s)",
        "camera_id": camera_id,
        "source": source,
    }


@app.function()
@modal.fastapi_endpoint(method="POST")
def stop_background(camera_id: str = "default"):
    """Signal to stop background detection (will stop on next iteration)."""
    key = f"bg_{camera_id}"
    current = global_state.get(key, {})
    if current.get("status") == "running":
        global_state[key] = {**current, "status": "stop_requested"}
        return {"ok": True, "message": "Stop requested", "camera_id": camera_id}
    return {"ok": False, "message": "Not running", "camera_id": camera_id}


@app.function()
@modal.fastapi_endpoint(method="POST")
def stop_all(camera_id: str = "default", clear_flags: bool = False):
    """
    POWERFUL STOP - Stops ALL running streams and background detections.
    Use this to immediately halt all GPU processing and save costs.
    
    - camera_id: specific camera to stop (also stops all if "all")
    - clear_flags: if True, also clears all stop flags for fresh start
    """
    import time as t
    
    global_state["stop_all"] = True
    
    stopped = []
    
    if camera_id and camera_id != "all":
        global_state[f"stop_{camera_id}"] = True
        global_state[f"stream_{camera_id}"] = {"status": "stop_requested", "ts": t.time()}
        stopped.append(f"stream:{camera_id}")
        
        bg_key = f"bg_{camera_id}"
        bg_current = global_state.get(bg_key, {})
        if bg_current.get("status") == "running":
            global_state[bg_key] = {**bg_current, "status": "stop_requested"}
            stopped.append(f"bg:{camera_id}")
    
    for key in list(global_state.keys()):
        if key.startswith("bg_"):
            cam_id = key[3:]
            current = global_state.get(key, {})
            if current.get("status") == "running":
                global_state[key] = {**current, "status": "stop_requested"}
                if f"bg:{cam_id}" not in stopped:
                    stopped.append(f"bg:{cam_id}")
        elif key.startswith("stream_"):
            cam_id = key[7:]
            global_state[f"stop_{cam_id}"] = True
            if f"stream:{cam_id}" not in stopped:
                stopped.append(f"stream:{cam_id}")
    
    global_state["stop_all_ts"] = t.time()
    
    if clear_flags:
        global_state["stop_all"] = False
        for key in list(global_state.keys()):
            if key.startswith("stop_") and key != "stop_all_ts":
                global_state[key] = False
    
    return {
        "ok": True,
        "message": "STOP ALL signal sent to all streams and background tasks",
        "stopped": stopped,
        "timestamp": t.time(),
    }


@app.function()
@modal.fastapi_endpoint()
def get_all_status(clear_flags: bool = False):
    """
    Get status of all streams and background tasks.
    Also can clear stop flags if clear_flags=True.
    """
    import time as t
    
    if clear_flags:
        global_state["stop_all"] = False
        for key in list(global_state.keys()):
            if key.startswith("stop_") and key != "stop_all_ts":
                global_state[key] = False
    
    streams = {}
    backgrounds = {}
    
    for key in list(global_state.keys()):
        if key.startswith("stream_"):
            cam_id = key[7:]
            streams[cam_id] = global_state.get(key, {})
        elif key.startswith("bg_"):
            cam_id = key[3:]
            backgrounds[cam_id] = global_state.get(key, {})
    
    return {
        "streams": streams,
        "backgrounds": backgrounds,
        "stop_all_active": global_state.get("stop_all", False),
        "timestamp": t.time(),
    }

