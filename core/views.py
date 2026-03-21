import os
import httpx
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from .forms import RegisterForm

NTFY_BASE_URL = os.getenv("NTFY_BASE_URL", "https://ntfy.sh").strip()
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "farmguard-alerts").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


def home(request):
    return render(request, "home.html")

def about(request):
    return render(request, "about.html")

def contact(request):
    return render(request, "contact.html")


@login_required(login_url="login")
def deployment(request):
    phone = ""
    try:
        phone = request.user.userprofile.phone
    except Exception:
        pass

    return render(request, "deployment.html", {
        "user_phone": phone,
        "user_id": request.user.id,
    })


@login_required(login_url="login")
def repellent_control(request):
    """Repellent sound control page for farmers."""
    # Extend session to 30 minutes of inactivity
    request.session.set_expiry(1800)  # 30 minutes
    return render(request, "repellent_control.html")


def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("deployment")
        else:
            messages.error(request, "Invalid username or password")

    return render(request, "login.html")


def logout_view(request):
    logout(request)
    return redirect("login")


def _subscribe_ntfy(phone: str):
    """Send a welcome notification to the user's personal ntfy topic."""
    if not NTFY_TOPIC:
        return
    try:
        httpx.post(
            f"{NTFY_BASE_URL}/{NTFY_TOPIC}",
            data=f"Welcome to FarmGuard! You will receive animal detection alerts here. Phone: {phone}".encode("utf-8"),
            headers={
                "Title": "FarmGuard - Alert Subscription Active",
                "Priority": "default",
                "Tags": "white_check_mark,bell",
            },
            timeout=10,
        )
    except Exception as e:
        print(f"ntfy subscribe notification failed: {e}")


def _subscribe_telegram(phone: str, chat_id: str = ""):
    """Send a welcome message via Telegram if bot token and chat IDs are configured."""
    if not TELEGRAM_BOT_TOKEN:
        return
    chat_ids_raw = os.getenv("TELEGRAM_CHAT_IDS", "")
    if not chat_ids_raw:
        return
    chat_ids = [c.strip() for c in chat_ids_raw.split(",") if c.strip()]
    base = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    text = (
        f"*New FarmGuard User Registered*\n"
        f"Phone: `{phone}`\n"
        f"Alert notifications are now active for this user."
    )
    for cid in chat_ids:
        try:
            httpx.post(
                f"{base}/sendMessage",
                json={"chat_id": cid, "text": text, "parse_mode": "Markdown"},
                timeout=10,
            )
        except Exception:
            pass


def register(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            phone = user.userprofile.phone

            try:
                _subscribe_ntfy(phone)
                user.userprofile.ntfy_subscribed = True
                user.userprofile.save(update_fields=["ntfy_subscribed"])
            except Exception as e:
                messages.warning(request, f"Account created, but ntfy subscription had an issue: {e}")

            try:
                _subscribe_telegram(phone)
            except Exception:
                pass

            # Commit Modal database volume after registration
            try:
                import os
                if os.path.exists("/data/db/farmguard.db"):
                    from modal import Volume
                    db_vol = Volume.from_name("farmguard-db-volume")
                    db_vol.commit()
                    print("[DB] Volume committed after user registration")
            except Exception as e:
                print(f"[DB] Could not commit volume: {e}")

            messages.success(request, "Account created! You are now subscribed to alerts. Please login.")
            return redirect("login")
    else:
        form = RegisterForm()

    return render(request, "register.html", {"form": form})
