"""
alert_agent.py - Telegram alert dispatch for FarmGuard.
"""
import os
import json
import base64
from typing import Optional

import httpx

from telegram_bot import get_subscriber_chat_ids, get_bot_token

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").strip()


def send_telegram_alert(
    animal: str,
    severity: int,
    message: str,
    incident_id: int = 0,
    image_b64: Optional[str] = None,
) -> dict:
    """Send Telegram message to all registered subscribers."""
    token = get_bot_token()
    if not token:
        return {"telegram": "skipped", "reason": "no token configured"}

    # Get all registered subscriber chat IDs
    chat_ids = get_subscriber_chat_ids()
    
    # Also include any manually configured chat IDs from env
    env_chat_ids = os.getenv("TELEGRAM_CHAT_IDS", "")
    if env_chat_ids:
        for cid in env_chat_ids.split(","):
            cid = cid.strip()
            if cid and cid not in chat_ids:
                chat_ids.append(cid)
    
    if not chat_ids:
        return {"telegram": "skipped", "reason": "no subscribers"}

    base = f"https://api.telegram.org/bot{token}"

    # The message already contains emoji, tips, and repellent info
    # Just add severity badge and feedback buttons
    caption = (
        f"\U0001f6a8 *FarmGuard Alert* \u2014 Severity `{severity}/10`\n\n"
        f"{message}"
    )

    # Telegram caption limit is 1024 chars for photos
    if image_b64 and len(caption) > 1020:
        caption = caption[:1017] + "..."

    reply_markup = json.dumps({
        "inline_keyboard": [[
            {"text": "\u2705 Confirmed", "callback_data": f"feedback:{incident_id}:confirmed"},
            {"text": "\u274c False alarm", "callback_data": f"feedback:{incident_id}:false_positive"},
        ]]
    })

    results = {}
    for chat_id in chat_ids:
        try:
            if image_b64:
                img_bytes = base64.b64decode(image_b64)
                resp = httpx.post(
                    f"{base}/sendPhoto",
                    data={
                        "chat_id": chat_id,
                        "caption": caption,
                        "parse_mode": "Markdown",
                        "reply_markup": reply_markup,
                    },
                    files={"photo": ("detection.jpg", img_bytes, "image/jpeg")},
                    timeout=15,
                )
            else:
                resp = httpx.post(
                    f"{base}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": caption,
                        "parse_mode": "Markdown",
                        "reply_markup": json.loads(reply_markup),
                    },
                    timeout=15,
                )
            results[chat_id] = "sent" if resp.status_code == 200 else f"error:{resp.status_code}"
        except Exception as e:
            results[chat_id] = f"failed:{str(e)[:60]}"

    return {"telegram": results, "subscribers": len(chat_ids)}


def dispatch_all_alerts(
    incident_id: int,
    animal: str,
    severity: int,
    confidence_pct: int,
    message: str,
    image_b64: Optional[str] = None,
) -> dict:
    """
    Dispatches alerts to Telegram.
    Called from main.py when a dangerous animal is detected.
    """
    return send_telegram_alert(
        animal=animal,
        severity=severity,
        message=message,
        incident_id=incident_id,
        image_b64=image_b64,
    )
