"""
Chess.com Published Data API transport and exact endpoint access.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests

from .config import API, DEFAULT_CACHE
from .helpers import validate_api_url


@dataclass(frozen=True)
class ChessComClient:
    cache_dir: Path = DEFAULT_CACHE
    fetch: Callable[[str], dict] | None = field(default=None, repr=False)

    def get_json(self, url: str) -> dict:
        """Serial cached requests with bounded retries; never cache error pages."""
        validate_api_url(url)
        if self.fetch is not None:
            payload = self.fetch(url)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object: {url}")
            return payload
        path = Path(self.cache_dir) / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("url") != url or not isinstance(cached.get("data"), dict):
                raise ValueError(f"Invalid cache entry: {path}; remove it and retry")
            return cached["data"]
        for attempt in range(4):
            time.sleep(0.2 if attempt == 0 else 2 ** attempt)
            try:
                response = requests.get(
                    url, timeout=(10, 45), allow_redirects=False,
                    headers={"User-Agent": "chess-take-home/0.1 (educational research)"},
                )
                if response.status_code != 200:
                    response.raise_for_status()
                    raise ValueError(f"Unexpected HTTP {response.status_code}: {url}")
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected a JSON object: {url}")
            except requests.RequestException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status not in {403, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"API request failed: {url} ({status})") from exc
                if attempt == 3:
                    raise RuntimeError(f"API unavailable after 4 attempts: {url}") from exc
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            entry = {"url": url, "fetched_at": datetime.now(timezone.utc).isoformat(), "data": payload}
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(entry), encoding="utf-8")
            temporary.replace(path)
            return payload
        raise RuntimeError(f"Unreachable request failure: {url}")

    def player_games(self, username: str, year: int, month: int) -> list[dict]:
        """Exact monthly archive records, including POST-game ratings."""
        return self.get_json(f"{API}/player/{username}/games/{year}/{month:02d}")["games"]

    def tournament(self, url: str) -> dict:
        return self.get_json(url)

    def round_groups(self, round_url: str) -> list[str]:
        return self.get_json(round_url)["groups"]

    def group_games(self, group_url: str) -> list[dict]:
        return self.get_json(group_url)["games"]
