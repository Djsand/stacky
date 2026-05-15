from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import HomeAssistantConfig


class HomeAssistantError(RuntimeError):
    pass


@dataclass(frozen=True)
class HomeAssistantActionProposal:
    domain: str
    service: str
    payload: dict[str, Any]
    reason: str


class HomeAssistantClient:
    def __init__(self, config: HomeAssistantConfig) -> None:
        self.config = config

    async def read_state(self, entity_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._request_json, "GET", f"/api/states/{entity_id}", None)

    async def call_service(
        self,
        domain: str,
        service: str,
        payload: dict[str, Any],
        *,
        explicit_user_command: bool = False,
    ) -> dict[str, Any] | HomeAssistantActionProposal:
        if self.config.suggest_first and not explicit_user_command:
            return HomeAssistantActionProposal(
                domain=domain,
                service=service,
                payload=payload,
                reason="Stacky foreslår først Home Assistant-handlinger, medmindre Nicol gav en tydelig kommando.",
            )
        return await asyncio.to_thread(self._request_json, "POST", f"/api/services/{domain}/{service}", payload)

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        if not self.config.token:
            raise HomeAssistantError("HOME_ASSISTANT_TOKEN mangler.")
        url = self.config.base_url.rstrip("/") + path
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise HomeAssistantError(f"Home Assistant HTTP {exc.code}: {detail}") from exc
        except OSError as exc:
            raise HomeAssistantError(f"Home Assistant connection failed: {exc}") from exc
