"""
sms.py - Phone normalization utilities.
SNS has been removed; alerts now go through ntfy.sh and Telegram (see backend_fastapi/alert_agent.py).
"""
import time

_last_alert = {}


def can_send_alert(key: str, cooldown: int = 900) -> bool:
    now = time.time()
    if key not in _last_alert or now - _last_alert[key] > cooldown:
        _last_alert[key] = now
        return True
    return False


def normalize_phone(phone_10: str, country: str = "NP") -> str:
    phone = (phone_10 or "").strip().replace(" ", "").replace("-", "")

    if not phone.isdigit():
        raise ValueError("Phone must contain digits only")

    if len(phone) != 10 or not phone.startswith("9"):
        raise ValueError("Phone must be 10 digits and start with 9")

    if country.upper() == "NP":
        return "+977" + phone
    if country.upper() == "IN":
        return "+91" + phone

    raise ValueError("Unsupported country (use 'NP' or 'IN')")
