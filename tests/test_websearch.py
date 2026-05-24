from __future__ import annotations

import unittest
import urllib.request

from stacky.llm import ChatMessage
from stacky.websearch import (
    DuckDuckGoLiteSearch,
    WebSearchResult,
    classify_web_search_intent,
    extract_web_search_query,
    format_web_search_context,
    wants_web_search,
)


class FakeIntentBrain:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[list[ChatMessage]] = []

    async def chat(self, messages: list[ChatMessage], *, temperature: float = 0.4, max_tokens: int | None = None) -> str:
        self.messages.append(messages)
        return self.response


class FakeResponse:
    def __init__(self, body: str) -> None:
        self.body = body.encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class WebSearchTest(unittest.IsolatedAsyncioTestCase):
    def test_wants_web_search_only_for_explicit_requests(self) -> None:
        self.assertTrue(wants_web_search("søg på nettet efter stackchan"))
        self.assertTrue(wants_web_search("kan du søge på nettet efter lokale stemme modeller"))
        self.assertTrue(wants_web_search("kan du slå op hvad den nyeste firmware er"))
        self.assertTrue(wants_web_search("kan du google citrus consult"))
        self.assertFalse(wants_web_search("jeg synes websearch skal implementeres tidligt"))
        self.assertFalse(wants_web_search("vi skal forbedre websearch routing"))
        self.assertFalse(wants_web_search("Google Gemini provideren er hurtig"))
        self.assertFalse(wants_web_search("hvad tænker du om kameraet"))

    async def test_agentic_intent_routes_natural_web_request(self) -> None:
        brain = FakeIntentBrain('{"web_search": true, "query": "lokale stemme modeller nyheder"}')

        intent = await classify_web_search_intent(
            "kan du finde noget nyt om lokale stemme modeller eller andet interessant",
            brain,
        )

        self.assertTrue(intent.wants_search)
        self.assertEqual(intent.query, "lokale stemme modeller nyheder")
        self.assertTrue(brain.messages)

    async def test_agentic_intent_rejects_plain_conversation(self) -> None:
        brain = FakeIntentBrain('{"web_search": true, "query": "forkert"}')

        intent = await classify_web_search_intent("hvad tænker du om kameraet", brain)

        self.assertFalse(intent.wants_search)
        self.assertFalse(brain.messages)

    def test_extract_web_search_query_removes_command_words(self) -> None:
        self.assertEqual(
            extract_web_search_query("kan du lige søg på nettet efter StackChan CoreS3 firmware"),
            "StackChan CoreS3 firmware",
        )
        self.assertEqual(extract_web_search_query("slå op Supertonic dansk tts"), "Supertonic dansk tts")

    def test_duckduckgo_lite_parser_returns_results(self) -> None:
        html = """
        <html><body>
          <a class="result-link" href="/l/?kh=-1&amp;uddg=https%3A%2F%2Fexample.com%2Fone">First &amp; result</a>
          <td class="result-snippet">Short snippet here.</td>
          <a class="result-link" href="https://example.org/two">Second result</a>
          <td class="result-snippet">Another snippet.</td>
        </body></html>
        """

        def opener(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
            self.assertIn("q=stacky", request.full_url)
            self.assertEqual(timeout, 2.0)
            return FakeResponse(html)

        client = DuckDuckGoLiteSearch(timeout_seconds=2.0, opener=opener)
        results = client.search("stacky", max_results=2)

        self.assertEqual(
            results,
            (
                WebSearchResult("First & result", "https://example.com/one", "Short snippet here."),
                WebSearchResult("Second result", "https://example.org/two", "Another snippet."),
            ),
        )

    def test_format_web_search_context_keeps_sources_compact(self) -> None:
        context = format_web_search_context(
            "stacky",
            (WebSearchResult("Stacky docs", "https://example.com/docs", "A compact snippet."),),
        )

        self.assertIn("Runtime sogte paa nettet", context)
        self.assertIn("Stacky docs", context)
        self.assertIn("example.com", context)
        self.assertIn("A compact snippet.", context)


if __name__ == "__main__":
    unittest.main()
