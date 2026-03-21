"""
telegram_bot.py - Telegram bot for FarmGuard alert subscriptions.

Users must:
1. Register on the website first
2. Send /start to the bot
3. Enter their website username
4. If username matches, they receive alerts

This links Telegram to the Django UserProfile.
"""
import os
import json
import time
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
import httpx

BASE_DIR = Path(__file__).resolve().parent
# Support both local Windows path and Modal Linux path
if Path("/app/db.sqlite3").exists():
    DJANGO_DB_PATH = Path("/app/db.sqlite3")
else:
    DJANGO_DB_PATH = BASE_DIR.parent / "db.sqlite3"
PENDING_FILE = BASE_DIR / "telegram_pending.json"


def get_bot_token() -> str:
    """Get bot token from environment (called at runtime, not import time)."""
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


def _load_pending() -> Dict[str, Any]:
    """Load pending verifications (users who sent /start but haven't entered username yet)."""
    if PENDING_FILE.exists():
        try:
            with open(PENDING_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_pending(data: Dict[str, Any]):
    """Save pending verifications."""
    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Look up a user in the Django database by username."""
    if not DJANGO_DB_PATH.exists():
        print(f"[TelegramBot] Django DB not found: {DJANGO_DB_PATH}")
        return None
    
    try:
        conn = sqlite3.connect(str(DJANGO_DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Join auth_user with core_userprofile
        cursor.execute("""
            SELECT u.id, u.username, u.first_name, u.last_name, u.email,
                   p.id as profile_id, p.phone, p.telegram_chat_id
            FROM auth_user u
            LEFT JOIN core_userprofile p ON p.user_id = u.id
            WHERE LOWER(u.username) = LOWER(?)
        """, (username,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    except Exception as e:
        print(f"[TelegramBot] DB error: {e}")
        return None


def link_telegram_to_user(username: str, chat_id: int) -> bool:
    """Link a Telegram chat_id to a user's profile in Django database."""
    if not DJANGO_DB_PATH.exists():
        return False
    
    try:
        conn = sqlite3.connect(str(DJANGO_DB_PATH))
        cursor = conn.cursor()
        
        # Get user_id from username
        cursor.execute("SELECT id FROM auth_user WHERE LOWER(username) = LOWER(?)", (username,))
        user_row = cursor.fetchone()
        
        if not user_row:
            conn.close()
            return False
        
        user_id = user_row[0]
        
        # Update or check if profile exists
        cursor.execute("SELECT id FROM core_userprofile WHERE user_id = ?", (user_id,))
        profile_row = cursor.fetchone()
        
        if profile_row:
            # Update existing profile
            cursor.execute(
                "UPDATE core_userprofile SET telegram_chat_id = ? WHERE user_id = ?",
                (str(chat_id), user_id)
            )
        else:
            # This shouldn't happen if user registered properly, but handle it
            print(f"[TelegramBot] No profile found for user {username}")
            conn.close()
            return False
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[TelegramBot] DB update error: {e}")
        return False


def get_all_linked_chat_ids() -> List[str]:
    """Get all Telegram chat IDs that are linked to website users."""
    if not DJANGO_DB_PATH.exists():
        return []
    
    try:
        conn = sqlite3.connect(str(DJANGO_DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT telegram_chat_id FROM core_userprofile 
            WHERE telegram_chat_id IS NOT NULL AND telegram_chat_id != ''
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [row[0] for row in rows if row[0]]
    except Exception as e:
        print(f"[TelegramBot] DB read error: {e}")
        return []


def get_all_subscribers() -> List[Dict[str, Any]]:
    """Get all users with linked Telegram accounts."""
    if not DJANGO_DB_PATH.exists():
        return []
    
    try:
        conn = sqlite3.connect(str(DJANGO_DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT u.username, u.first_name, p.phone, p.telegram_chat_id
            FROM auth_user u
            JOIN core_userprofile p ON p.user_id = u.id
            WHERE p.telegram_chat_id IS NOT NULL AND p.telegram_chat_id != ''
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"[TelegramBot] DB read error: {e}")
        return []


def get_subscriber_chat_ids() -> List[str]:
    """Get all subscriber chat IDs for sending alerts."""
    return get_all_linked_chat_ids()


def send_telegram_message(chat_id: int, text: str, parse_mode: str = "Markdown") -> bool:
    """Send a message to a specific chat."""
    token = get_bot_token()
    if not token:
        return False
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = httpx.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"[TelegramBot] Failed to send message: {e}")
        return False


def answer_callback_query(callback_query_id: str, text: str = "", show_alert: bool = False) -> bool:
    """Answer a callback query (acknowledge button click)."""
    token = get_bot_token()
    if not token:
        return False
    
    url = f"https://api.telegram.org/bot{token}/answerCallbackQuery"
    try:
        resp = httpx.post(url, json={
            "callback_query_id": callback_query_id,
            "text": text,
            "show_alert": show_alert,
        }, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"[TelegramBot] Failed to answer callback: {e}")
        return False


def edit_message_reply_markup(chat_id: int, message_id: int, reply_markup: dict = None) -> bool:
    """Edit the reply markup (buttons) of a message."""
    token = get_bot_token()
    if not token:
        return False
    
    url = f"https://api.telegram.org/bot{token}/editMessageReplyMarkup"
    try:
        data = {"chat_id": chat_id, "message_id": message_id}
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)
        resp = httpx.post(url, json=data, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"[TelegramBot] Failed to edit markup: {e}")
        return False


def handle_callback_query(callback_query: Dict[str, Any]) -> Optional[str]:
    """
    Handle callback query (button click) from Telegram.
    """
    callback_id = callback_query.get("id")
    data = callback_query.get("data", "")
    message = callback_query.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    message_id = message.get("message_id")
    user = callback_query.get("from", {})
    first_name = user.get("first_name", "User")
    
    if not data or not callback_id:
        return None
    
    # Parse callback data: "feedback:incident_id:action"
    parts = data.split(":")
    if len(parts) >= 3 and parts[0] == "feedback":
        incident_id = parts[1]
        action = parts[2]
        
        if action == "confirmed":
            # User confirmed the threat
            answer_callback_query(callback_id, "✅ Thank you! Threat confirmed.", show_alert=False)
            
            # Update the message to show feedback received
            new_markup = {
                "inline_keyboard": [[
                    {"text": "✅ Confirmed by " + first_name, "callback_data": "noop"}
                ]]
            }
            edit_message_reply_markup(chat_id, message_id, new_markup)
            
            # Log the feedback
            print(f"[TelegramBot] Feedback: incident {incident_id} CONFIRMED by {first_name}")
            
            # Save feedback to file
            save_feedback(incident_id, "confirmed", chat_id, first_name)
            
            return "feedback_confirmed"
        
        elif action == "false_positive":
            # User marked as false alarm
            answer_callback_query(callback_id, "❌ Marked as false alarm. Thanks for the feedback!", show_alert=False)
            
            # Update the message to show feedback received
            new_markup = {
                "inline_keyboard": [[
                    {"text": "❌ False alarm - " + first_name, "callback_data": "noop"}
                ]]
            }
            edit_message_reply_markup(chat_id, message_id, new_markup)
            
            # Log the feedback
            print(f"[TelegramBot] Feedback: incident {incident_id} FALSE POSITIVE by {first_name}")
            
            # Save feedback to file
            save_feedback(incident_id, "false_positive", chat_id, first_name)
            
            return "feedback_false_positive"
    
    # Handle "noop" - already answered
    if data == "noop":
        answer_callback_query(callback_id, "Feedback already recorded.", show_alert=False)
        return "noop"
    
    # Unknown callback
    answer_callback_query(callback_id, "Unknown action", show_alert=False)
    return None


def save_feedback(incident_id: str, feedback_type: str, chat_id: int, user_name: str, detection_image: str = ""):
    """Save user feedback to a JSON file and handle false positives."""
    feedback_file = BASE_DIR / "alert_feedback.json"
    
    try:
        if feedback_file.exists():
            with open(feedback_file, "r") as f:
                feedback_data = json.load(f)
        else:
            feedback_data = {"feedbacks": [], "false_alarm_suppression": {}}
        
        feedback_data["feedbacks"].append({
            "incident_id": incident_id,
            "feedback": feedback_type,
            "chat_id": chat_id,
            "user_name": user_name,
            "timestamp": time.time(),
        })
        
        # If false positive, suppress alerts for 1 hour and delete image
        if feedback_type == "false_positive":
            if "false_alarm_suppression" not in feedback_data:
                feedback_data["false_alarm_suppression"] = {}
            
            suppression_until = time.time() + 3600
            feedback_data["false_alarm_suppression"][incident_id] = {
                "until": suppression_until,
                "user": user_name,
            }
            print(f"[TelegramBot] Alerts suppressed for incident {incident_id} until {time.strftime('%H:%M:%S', time.localtime(suppression_until))}")
            
            # Load incident mapping to find the image filename
            incident_mapping_file = BASE_DIR / "incident_mappings.json"
            image_filename = ""
            
            if incident_mapping_file.exists():
                try:
                    with open(incident_mapping_file, "r") as f:
                        incident_mappings = json.load(f)
                    
                    incident_data = incident_mappings.get(incident_id, {})
                    if incident_data and "image_filename" in incident_data:
                        image_filename = incident_data["image_filename"]
                        print(f"[TelegramBot] Found image mapping: {incident_id} -> {image_filename}")
                except Exception as e:
                    print(f"[TelegramBot] Failed to load incident mapping: {e}")
            
            if not image_filename:
                print(f"[TelegramBot] No image mapping found for {incident_id}")
            
            # Delete the detection image - try multiple possible locations
            if image_filename:
                from pathlib import Path
                deleted = False
                
                # List of possible directories where detection images live
                search_dirs = [
                    Path("/data/detections"),       # Modal volume
                    BASE_DIR / "detections",        # Local backend_fastapi/detections
                    BASE_DIR.parent / "detections",  # Project root/detections
                ]
                
                for search_dir in search_dirs:
                    img_path = search_dir / image_filename
                    if img_path.exists():
                        try:
                            img_path.unlink()
                            print(f"[TelegramBot] Deleted false positive image: {img_path}")
                            deleted = True
                            
                            # Commit Modal volume after deletion
                            if str(search_dir).startswith("/data/"):
                                try:
                                    from modal import Volume
                                    vol = Volume.from_name("farmguard-detections-volume")
                                    vol.commit()
                                    print(f"[TelegramBot] Modal volume committed after image deletion")
                                except Exception:
                                    pass
                            break
                        except Exception as e:
                            print(f"[TelegramBot] Failed to delete {img_path}: {e}")
                
                if not deleted:
                    print(f"[TelegramBot] Image not found in any location: {image_filename}")
                    print(f"[TelegramBot] Searched: {[str(d / image_filename) for d in search_dirs]}")
            
            # Also remove from detection history JSON
            try:
                history_file = BASE_DIR / "detection_history.json"
                if history_file.exists():
                    with open(history_file, "r") as f:
                        history = json.load(f)
                    
                    original_len = len(history)
                    history = [r for r in history if r.get("image") != image_filename]
                    
                    if len(history) < original_len:
                        with open(history_file, "w") as f:
                            json.dump(history, f, indent=2)
                        print(f"[TelegramBot] Removed {original_len - len(history)} record(s) from detection history")
            except Exception as e:
                print(f"[TelegramBot] Failed to clean detection history: {e}")
            
            # Also remove from Django DB
            try:
                import django
                os.environ.setdefault("DJANGO_SETTINGS_MODULE", "detection.settings")
                django.setup()
                from core.models import DetectionEvent
                deleted_count, _ = DetectionEvent.objects.filter(image_path=image_filename).delete()
                if deleted_count:
                    print(f"[TelegramBot] Deleted {deleted_count} DB record(s) for false positive image: {image_filename}")
            except Exception as e:
                print(f"[TelegramBot] Failed to delete DB record: {e}")
        
        with open(feedback_file, "w") as f:
            json.dump(feedback_data, f, indent=2)
    except Exception as e:
        print(f"[TelegramBot] Failed to save feedback: {e}")


def is_alert_suppressed(incident_id: str = "") -> bool:
    """Check if alerts should be suppressed based on false positive feedback."""
    feedback_file = BASE_DIR / "alert_feedback.json"
    
    if not feedback_file.exists():
        return False
    
    try:
        with open(feedback_file, "r") as f:
            feedback_data = json.load(f)
        
        suppressions = feedback_data.get("false_alarm_suppression", {})
        
        # Clean up expired suppressions
        current_time = time.time()
        expired = [k for k, v in suppressions.items() if v.get("until", 0) < current_time]
        for k in expired:
            del suppressions[k]
        
        # Save cleaned data
        if expired:
            feedback_data["false_alarm_suppression"] = suppressions
            with open(feedback_file, "w") as f:
                json.dump(feedback_data, f, indent=2)
        
        # Check if specific incident is suppressed
        if incident_id and incident_id in suppressions:
            return True
        
        return False
    except Exception as e:
        print(f"[TelegramBot] Error checking suppression: {e}")
        return False


def handle_telegram_update(update: Dict[str, Any]) -> Optional[str]:
    """
    Handle incoming Telegram update (message or callback query).
    Returns response message or None.
    """
    # Handle callback queries (button clicks) first
    callback_query = update.get("callback_query")
    if callback_query:
        return handle_callback_query(callback_query)
    
    # Handle regular messages
    message = update.get("message", {})
    if not message:
        return None
    
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "").strip()
    
    if not chat_id:
        return None
    
    pending = _load_pending()
    chat_id_str = str(chat_id)
    
    # Handle /start command
    if text.lower() == "/start":
        # Ask for username
        pending[chat_id_str] = {"state": "awaiting_username", "ts": time.time()}
        _save_pending(pending)
        
        response = (
            "🐘 *Welcome to FarmGuard Alerts!*\n\n"
            "To receive alerts, please enter your *website username* "
            "(the one you used to register on the FarmGuard website).\n\n"
            "Just type your username and send it."
        )
        send_telegram_message(chat_id, response)
        return "awaiting_username"
    
    # Handle /stop command
    elif text.lower() == "/stop":
        # Check if user is linked
        linked_ids = get_all_linked_chat_ids()
        if chat_id_str in linked_ids:
            # Find and unlink the user
            try:
                conn = sqlite3.connect(str(DJANGO_DB_PATH))
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE core_userprofile SET telegram_chat_id = '' WHERE telegram_chat_id = ?",
                    (chat_id_str,)
                )
                conn.commit()
                conn.close()
            except Exception:
                pass
        
        # Clear pending state
        if chat_id_str in pending:
            del pending[chat_id_str]
            _save_pending(pending)
        
        response = (
            "🔕 *Unsubscribed*\n\n"
            "You will no longer receive FarmGuard alerts.\n\n"
            "Send /start anytime to subscribe again."
        )
        send_telegram_message(chat_id, response)
        return "unsubscribed"
    
    # Handle /status command
    elif text.lower() == "/status":
        linked_ids = get_all_linked_chat_ids()
        if chat_id_str in linked_ids:
            # Find the user
            subscribers = get_all_subscribers()
            user_info = next((s for s in subscribers if s.get("telegram_chat_id") == chat_id_str), None)
            if user_info:
                response = (
                    "✅ *You are subscribed to FarmGuard Alerts*\n\n"
                    f"Linked to: *{user_info.get('username')}*\n"
                    f"Phone: {user_info.get('phone', 'N/A')}\n\n"
                    "You will receive alerts when dangerous animals are detected."
                )
            else:
                response = "✅ You are subscribed but user info not found."
        else:
            response = (
                "❌ *You are not subscribed*\n\n"
                "Send /start to link your website account and subscribe to alerts."
            )
        send_telegram_message(chat_id, response)
        return "status_checked"
    
    # Check if user is in pending state (awaiting username)
    elif chat_id_str in pending and pending[chat_id_str].get("state") == "awaiting_username":
        username = text.strip()
        
        # Look up user in database
        user = get_user_by_username(username)
        
        if user:
            # User found! Link the Telegram account
            if link_telegram_to_user(username, chat_id):
                # Clear pending state
                del pending[chat_id_str]
                _save_pending(pending)
                
                first_name = user.get("first_name") or username
                response = (
                    f"✅ *Success!*\n\n"
                    f"Hello *{first_name}*! Your Telegram is now linked to your FarmGuard account.\n\n"
                    f"*You will receive alerts when dangerous animals are detected:*\n"
                    f"• 🐘 Elephants\n"
                    f"• 🐯 Tigers\n"
                    f"• 🐻 Bears\n\n"
                    f"📸 Each alert includes a photo of the detection.\n\n"
                    f"Commands:\n"
                    f"/status - Check your subscription\n"
                    f"/stop - Unsubscribe from alerts"
                )
                send_telegram_message(chat_id, response)
                return "registered"
            else:
                response = (
                    "❌ *Error linking account*\n\n"
                    "There was a problem linking your Telegram. Please try again or contact support."
                )
                send_telegram_message(chat_id, response)
                return "link_error"
        else:
            # User not found
            response = (
                f"❌ *Username not found*\n\n"
                f"No account found with username: *{username}*\n\n"
                f"Please make sure you:\n"
                f"1. Have registered on the FarmGuard website first\n"
                f"2. Entered the correct username (case doesn't matter)\n\n"
                f"Try again or send /start to restart."
            )
            send_telegram_message(chat_id, response)
            return "user_not_found"
    
    # Handle any other message
    else:
        response = (
            "🤖 *FarmGuard Alert Bot*\n\n"
            "Available commands:\n"
            "/start - Link your website account\n"
            "/stop - Unsubscribe from alerts\n"
            "/status - Check subscription status"
        )
        send_telegram_message(chat_id, response)
        return "help"
    
    return None


def poll_telegram_updates():
    """
    Poll for new Telegram updates (for testing without webhook).
    Call this periodically to check for new messages.
    """
    token = get_bot_token()
    if not token:
        return []
    
    # Load last update ID from a simple file
    state_file = BASE_DIR / "telegram_state.json"
    last_update_id = 0
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
                last_update_id = state.get("last_update_id", 0)
        except Exception:
            pass
    
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        resp = httpx.get(url, params={
            "offset": last_update_id + 1,
            "timeout": 1,
        }, timeout=10)
        
        if resp.status_code != 200:
            return []
        
        result = resp.json()
        updates = result.get("result", [])
        
        processed = []
        for update in updates:
            update_id = update.get("update_id", 0)
            if update_id > last_update_id:
                last_update_id = update_id
                action = handle_telegram_update(update)
                if action:
                    processed.append({"update_id": update_id, "action": action})
        
        # Save last update ID
        with open(state_file, "w") as f:
            json.dump({"last_update_id": last_update_id}, f)
        
        return processed
    
    except Exception as e:
        print(f"[TelegramBot] Poll error: {e}")
        return []


def send_alert_to_all_subscribers(
    animal: str,
    confidence: float,
    message: str,
    image_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """
    Send an alert to all registered subscribers.
    Returns dict with results per chat_id.
    """
    token = get_bot_token()
    if not token:
        return {"error": "Bot token not configured"}
    
    chat_ids = get_subscriber_chat_ids()
    if not chat_ids:
        return {"error": "No subscribers", "sent": 0}
    
    base_url = f"https://api.telegram.org/bot{token}"
    results = {}
    
    caption = (
        f"🚨 *FarmGuard Alert*\n\n"
        f"*{animal.replace('_', ' ').title()}* detected!\n"
        f"Confidence: {confidence:.1%}\n\n"
        f"{message}"
    )
    
    for chat_id in chat_ids:
        try:
            if image_bytes:
                resp = httpx.post(
                    f"{base_url}/sendPhoto",
                    data={
                        "chat_id": chat_id,
                        "caption": caption,
                        "parse_mode": "Markdown",
                    },
                    files={"photo": ("detection.jpg", image_bytes, "image/jpeg")},
                    timeout=15,
                )
            else:
                resp = httpx.post(
                    f"{base_url}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": caption,
                        "parse_mode": "Markdown",
                    },
                    timeout=10,
                )
            
            results[str(chat_id)] = "sent" if resp.status_code == 200 else f"error:{resp.status_code}"
        except Exception as e:
            results[str(chat_id)] = f"failed:{str(e)[:50]}"
    
    return {"results": results, "sent": sum(1 for v in results.values() if v == "sent")}
