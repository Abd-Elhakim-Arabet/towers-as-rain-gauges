"""Alert levels, and the email that goes out when one is reached.

Levels (the same four as the replay):
  normal
  watch      rain so far is at least half of the rain that would flood the town
  warning    the flood model, run ahead on the expected rain, shows the town flooding within the next 2 hours
  flooding   the flood model shows the town flooding now

"The town floods" = built-up land under at least 30 cm of water reaches the area's alarm size (hectares).

Email settings (never in the repository), first match wins:
  1. the environment
  2. data/live/mail.json, written by the page's "Alarm email" set-up (save_mail)
  3. .env
  ALERT_TO                      who is warned (comma-separated)
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, ALERT_FROM     the sending account
Without SMTP settings the email is written to data/live/outbox/ instead, so nothing is lost and it can be shown.
"""

from __future__ import annotations

import json
import os
import re
import smtplib
import time
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTBOX = ROOT / "data" / "live" / "outbox"
SETTINGS = ROOT / "data" / "live" / "mail.json"
KEYS = ("ALERT_TO", "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD")
EMAIL = re.compile(r"^[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+$")
ORDER = ("normal", "watch", "warning", "flooding")
TITLE = {"normal": "Normal", "watch": "Watch", "warning": "Flood warning", "flooding": "Flooding"}
CALM_STEPS = 3            # a level only drops after this many calmer readings in a row


def level(ha_now: float, ha_ahead: float, alarm_ha: float, rain_ratio: float) -> str:
    if ha_now >= alarm_ha:
        return "flooding"
    if ha_ahead >= alarm_ha:
        return "warning"
    if rain_ratio >= 0.5 or ha_ahead >= 0.5 * alarm_ha:
        return "watch"
    return "normal"


def settle(current: str, reading: str, calm: int) -> tuple[str, int]:
    """Rise at once, fall only after CALM_STEPS calmer readings. Returns (level, calm counter)."""
    if ORDER.index(reading) >= ORDER.index(current):
        return reading, 0
    calm += 1
    return (reading, 0) if calm >= CALM_STEPS else (current, calm)


def _saved() -> dict:
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def env(name: str, default: str = "") -> str:
    if os.environ.get(name):
        return os.environ[name]
    if _saved().get(name):
        return str(_saved()[name])
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return default


def compose(city: str, new: str, old: str, facts: dict, drill: bool = False, simulated: bool = True) -> EmailMessage:
    tag = "[DRILL] " if drill else ""
    msg = EmailMessage()
    if new == "normal":
        msg["Subject"] = f"{tag}All clear: {city}"
        lead = f"The flood alert for {city} is over. The level is back to Normal."
    else:
        msg["Subject"] = f"{tag}{TITLE[new]}: {city}"
        lead = {"flooding": f"The flood model shows {city} flooding now.",
                "warning": f"The flood model expects {city} to start flooding within about 2 hours.",
                "watch": f"Heavy rain over {city}: the rain so far is at least half of what would flood the town."}[new]
    lines = [lead, "",
             f"Time (UTC):            {facts.get('time', '')}",
             f"Level:                 {TITLE[new]} (was {TITLE[old]})",
             f"Built-up land flooded: {facts.get('ha_now', 0):.1f} ha now, {facts.get('ha_ahead', 0):.1f} ha expected within 2 h "
             f"(alarm at {facts.get('alarm_ha', 0):.1f} ha)",
             f"Rain over the town:    {facts.get('rain_mmh', 0):.1f} mm/h now, {facts.get('rain_3h_mm', 0):.0f} mm in the last 3 h",
             f"Links seeing rain:     {facts.get('links_wet', 0)} of {facts.get('links_n', 0)}"]
    if facts.get("url"):
        lines += ["", f"Live map: {facts['url']}"]
    lines += [""]
    if drill:
        lines.append("THIS IS A DRILL. The storm is made up to test the alarm. Nothing is happening.")
    if simulated:
        lines.append("The tower links in this watch are simulated from weather-service rain; this is a demonstration "
                     "system, not an official warning.")
    lines.append("Sent by Towers as Rain Gauges (live watch).")
    msg.set_content("\n".join(lines))
    return msg


def send(msg: EmailMessage) -> str:
    """Send, or save to the outbox when no sending account is configured. Returns what happened, in a few words."""
    to = env("ALERT_TO")
    host = env("SMTP_HOST")
    msg["To"] = to or "nobody@localhost"
    msg["From"] = env("ALERT_FROM") or env("SMTP_USER") or "flood-watch@localhost"
    if to and host:
        try:
            port = int(env("SMTP_PORT", "587"))
            cls = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP
            with cls(host, port, timeout=30) as s:
                if port != 465:
                    s.starttls()
                if env("SMTP_USER"):
                    s.login(env("SMTP_USER"), env("SMTP_PASSWORD"))
                s.send_message(msg)
            return "sent"
        except smtplib.SMTPAuthenticationError:
            note = "not sent: the mail server refused the login (for Gmail, use an app password); saved to the outbox"
        except Exception as e:                      # a mail problem must not stop the watch: keep a copy
            note = f"not sent ({type(e).__name__}); saved to the outbox"
    else:
        note = "saved to the outbox (no sending account set up)" if to else "saved to the outbox (ALERT_TO is not set)"
    OUTBOX.mkdir(parents=True, exist_ok=True)
    (OUTBOX / f"{time.strftime('%Y%m%dT%H%M%S')}_{abs(hash(msg['Subject'])) % 10000:04d}.eml").write_bytes(bytes(msg))
    return note


def _mask(addr: str) -> str:
    name, _, dom = addr.strip().partition("@")
    return (name[:2] + "…@" + dom) if dom else ""


def mail_ready() -> dict:
    """What the page may know: whether each part is set, and the addresses partly hidden. Never the password."""
    to = [a for a in env("ALERT_TO").split(",") if a.strip()]
    return {"to": bool(to), "smtp": bool(env("SMTP_HOST")), "password": bool(env("SMTP_PASSWORD")),
            "to_shown": ", ".join(_mask(a) for a in to), "from_shown": _mask(env("SMTP_USER")),
            "host": env("SMTP_HOST"), "port": env("SMTP_PORT", "587"),
            "complete": bool(to) and bool(env("SMTP_HOST")) and bool(env("SMTP_USER")) and bool(env("SMTP_PASSWORD"))}


def save_mail(to: str, user: str = "", password: str = "", host: str = "", port: str = "") -> list[str]:
    """Store the page's email set-up in data/live/mail.json (owner-only). An empty field keeps the current value
    (from mail.json or .env). Returns problems in plain words; nothing is saved when there are any."""
    to, user, host = to.strip() or env("ALERT_TO"), user.strip() or env("SMTP_USER"), host.strip() or env("SMTP_HOST")
    port = port.strip() or env("SMTP_PORT", "587")
    to_list = [a.strip() for a in to.replace(";", ",").split(",") if a.strip()]
    problems = []
    if not to_list:
        problems.append("Give at least one address to send alarms to.")
    problems += [f"“{a}” does not look like an email address." for a in to_list if not EMAIL.match(a)]
    user = user.strip()
    if user and not EMAIL.match(user):
        problems.append(f"“{user}” does not look like an email address.")
    host = host.strip() or ("smtp.gmail.com" if user.lower().endswith(("@gmail.com", "@googlemail.com")) else "")
    if user and not host:
        problems.append("Give the mail server of the sending account (for Gmail it is filled in for you).")
    if not port.isdigit():
        problems.append("The port is a number, usually 587.")
    if problems:
        return problems
    keep_pw = env("SMTP_PASSWORD") if user == env("SMTP_USER") else ""
    new = {"ALERT_TO": ", ".join(to_list), "SMTP_HOST": host if user else "", "SMTP_PORT": port, "SMTP_USER": user,
           "SMTP_PASSWORD": password.replace(" ", "") if password.strip() else keep_pw}
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as f:   # owner-only from the start
        json.dump(new, f)
    tmp.replace(SETTINGS)
    return []


def test_message() -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = "[TEST] Flood watch alarm email is set up"
    msg.set_content("This is a test from the Towers as Rain Gauges live watch.\n\n"
                    "Alarm emails will arrive like this one when a watched town reaches Flood warning or Flooding, "
                    "and again when it is over. Drills are marked DRILL.\n")
    return msg
