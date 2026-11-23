"""
Small shared boundary helpers; no feature engineering or network I/O.
"""

from datetime import datetime, timezone
import re


def validate_api_url(url: str) -> None:
    if not re.fullmatch(r"https://api\.chess\.com/pub/[A-Za-z0-9_/-]+", url):
        raise ValueError(f"Not a permitted Published Data API URL: {url!r}")


def utc_timestamp(date_text: str, time_text: str) -> int:
    """Parse Chess.com's UTC PGN date/time without using local timezone settings."""
    value = datetime.strptime(date_text + " " + time_text, "%Y.%m.%d %H:%M:%S")
    return int(value.replace(tzinfo=timezone.utc).timestamp())
