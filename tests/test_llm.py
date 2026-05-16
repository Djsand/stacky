from __future__ import annotations

import unittest

from stacky.config import LMStudioConfig
from stacky.llm import ChatMessage, GeminiClient, LMStudioClient, create_chat_client


class LLMTest(unittest.TestCase):
    def test_creates_gemini_client_for_gemini_provider(self) -> None:
        client = create_chat_client(LMStudioConfig(provider="gemini", api_key="key"))

        self.assertIsInstance(client, GeminiClient)

    def test_creates_lmstudio_client_by_default(self) -> None:
        client = create_chat_client(LMStudioConfig())

        self.assertIsInstance(client, LMStudioClient)

    def test_gemini_payload_maps_system_and_user_messages(self) -> None:
        client = GeminiClient(LMStudioConfig(provider="gemini", api_key="key"))

        payload = client._payload(
            [
                ChatMessage("system", "Svar kort på dansk."),
                ChatMessage("user", "Hej"),
            ],
            temperature=0.3,
            max_tokens=40,
        )

        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], "Svar kort på dansk.")
        self.assertEqual(payload["contents"][0]["role"], "user")
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "Hej")
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 40)


if __name__ == "__main__":
    unittest.main()
