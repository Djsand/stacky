from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .config import LMStudioConfig


class LLMError(RuntimeError):
    pass


class LMStudioError(LLMError):
    pass


class GeminiError(LLMError):
    pass


class GeminiPromptBlockedError(GeminiError):
    def __init__(self, block_reason: str, response: dict[str, Any] | None = None) -> None:
        self.block_reason = block_reason
        self.response = response or {}
        super().__init__(f"Gemini blocked prompt: {block_reason}")


@dataclass(frozen=True)
class ChatImageAttachment:
    mime_type: str
    data_base64: str


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    images: tuple[ChatImageAttachment, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        if not self.images:
            return {"role": self.role, "content": self.content}
        parts: list[dict[str, Any]] = []
        if self.content:
            parts.append({"type": "text", "text": self.content})
        for image in self.images:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image.mime_type};base64,{image.data_base64}",
                    },
                }
            )
        return {"role": self.role, "content": parts}


class ChatClient(Protocol):
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.4,
        max_tokens: int | None = None,
    ) -> str:
        ...


def create_chat_client(config: LMStudioConfig) -> ChatClient:
    provider = config.provider.strip().lower()
    if provider == "gemini":
        return GeminiClient(config)
    if provider in {"lmstudio", "openai-compatible", "openai_compatible"}:
        return LMStudioClient(config)
    raise LLMError(f"Unknown brain provider: {config.provider}")


class LMStudioClient:
    def __init__(self, config: LMStudioConfig) -> None:
        self.config = config

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.4,
        max_tokens: int | None = None,
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": [message.to_dict() for message in messages],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return await asyncio.to_thread(self._post_chat, payload)

    def _post_chat(self, payload: dict[str, Any]) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LMStudioError(f"LM Studio HTTP {exc.code}: {detail}") from exc
        except OSError as exc:
            raise LMStudioError(f"LM Studio connection failed: {exc}") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LMStudioError(f"Unexpected LM Studio response: {data!r}") from exc
        return str(content).strip()


class GeminiClient:
    def __init__(self, config: LMStudioConfig) -> None:
        self.config = config

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.4,
        max_tokens: int | None = None,
    ) -> str:
        payload = self._payload(messages, temperature=temperature, max_tokens=max_tokens)
        return await asyncio.to_thread(self._post_generate_content, payload)

    def _payload(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        system_parts: list[dict[str, str]] = []
        contents: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                system_parts.append({"text": message.content})
                continue
            role = "model" if message.role == "assistant" else "user"
            parts: list[dict[str, Any]] = []
            if message.content:
                parts.append({"text": message.content})
            for image in message.images:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": image.mime_type,
                            "data": image.data_base64,
                        }
                    }
                )
            if not parts:
                parts.append({"text": ""})
            contents.append({"role": role, "parts": parts})
        if not contents:
            contents.append({"role": "user", "parts": [{"text": ""}]})

        generation_config: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}
        return payload

    def _post_generate_content(self, payload: dict[str, Any]) -> str:
        if not self.config.api_key:
            raise GeminiError("GEMINI_API_KEY is not set")
        base_url = self.config.base_url.rstrip("/") or "https://generativelanguage.googleapis.com/v1beta"
        model = self.config.model.removeprefix("models/")
        url = f"{base_url}/models/{model}:generateContent?key={self.config.api_key}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise GeminiError(f"Gemini HTTP {exc.code}: {detail}") from exc
        except OSError as exc:
            raise GeminiError(f"Gemini connection failed: {exc}") from exc

        return self._extract_content(data)

    def _extract_content(self, data: dict[str, Any]) -> str:
        prompt_feedback = data.get("promptFeedback")
        if isinstance(prompt_feedback, dict):
            block_reason = prompt_feedback.get("blockReason")
            if block_reason:
                raise GeminiPromptBlockedError(str(block_reason), data)

        try:
            candidate = data["candidates"][0]
            parts = candidate["content"]["parts"]
            content = "".join(str(part.get("text", "")) for part in parts)
        except (KeyError, IndexError, TypeError) as exc:
            candidates = data.get("candidates")
            if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
                finish_reason = candidates[0].get("finishReason")
                if finish_reason in {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}:
                    raise GeminiPromptBlockedError(str(finish_reason), data) from exc
            raise GeminiError(f"Unexpected Gemini response: {data!r}") from exc
        return content.strip()
