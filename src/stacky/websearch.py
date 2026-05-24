from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol

from .llm import ChatClient, ChatMessage


class WebSearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str = ""


class WebSearchClient(Protocol):
    def search(self, query: str, *, max_results: int = 3) -> tuple[WebSearchResult, ...]:
        ...


@dataclass(frozen=True)
class WebSearchIntent:
    wants_search: bool
    query: str = ""


UrlOpen = Callable[..., object]


class DuckDuckGoLiteSearch:
    """Small stdlib-only search client for explicit Stacky web-search requests."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 8.0,
        opener: UrlOpen | None = None,
        user_agent: str = "Stacky/0.1 local websearch",
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.opener = opener or urllib.request.urlopen
        self.user_agent = user_agent

    def search(self, query: str, *, max_results: int = 3) -> tuple[WebSearchResult, ...]:
        clean_query = " ".join(query.split())
        if not clean_query:
            return ()
        params = urllib.parse.urlencode({"q": clean_query})
        request = urllib.request.Request(
            f"https://lite.duckduckgo.com/lite/?{params}",
            headers={"User-Agent": self.user_agent},
            method="GET",
        )
        try:
            response = self.opener(request, timeout=self.timeout_seconds)
            with response:  # type: ignore[attr-defined]
                body = response.read().decode("utf-8", errors="replace")  # type: ignore[attr-defined]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:240]
            raise WebSearchError(f"DuckDuckGo HTTP {exc.code}: {detail}") from exc
        except OSError as exc:
            raise WebSearchError(f"DuckDuckGo connection failed: {exc}") from exc

        parser = DuckDuckGoLiteParser()
        parser.feed(body)
        return tuple(parser.results[: max(0, max_results)])


class DuckDuckGoLiteParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[WebSearchResult] = []
        self._in_result_link = False
        self._current_href = ""
        self._current_title: list[str] = []
        self._in_snippet = False
        self._snippet_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        class_name = attr.get("class", "")
        if tag == "a" and "result-link" in class_name:
            self._in_result_link = True
            self._current_href = attr.get("href", "")
            self._current_title = []
            return
        if "result-snippet" in class_name:
            self._in_snippet = True
            self._snippet_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_result_link:
            self._current_title.append(data)
        if self._in_snippet:
            self._snippet_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_result_link:
            title = _clean_html_text(" ".join(self._current_title))
            url = _normalize_duckduckgo_url(self._current_href)
            if title and url:
                self.results.append(WebSearchResult(title=title, url=url))
            self._in_result_link = False
            self._current_href = ""
            self._current_title = []
            return
        if self._in_snippet and tag in {"td", "div"}:
            snippet = _clean_html_text(" ".join(self._snippet_parts))
            if snippet and self.results:
                last = self.results[-1]
                if not last.snippet:
                    self.results[-1] = WebSearchResult(last.title, last.url, snippet)
            self._in_snippet = False
            self._snippet_parts = []


async def classify_web_search_intent(text: str, brain: ChatClient | None) -> WebSearchIntent:
    """Decide whether Stacky's web ability should run before the reply.

    The deterministic rules are a fallback. The important path is the small
    agentic classifier: if the user naturally asks for fresh/current internet
    information, route the ability before the main brain can merely refuse it.
    """

    if wants_web_search(text):
        return WebSearchIntent(True, extract_web_search_query(text))
    if brain is None or not _looks_like_possible_web_need(text):
        return WebSearchIntent(False)

    prompt = (
        "Du er Stackys ability-router, ikke samtalehjernen. "
        "Afgør om brugerens danske besked beder Stacky om at hente frisk/aktuelt internetindhold før svaret. "
        "Svar KUN JSON: {\"web_search\": true/false, \"query\": \"kort søgeforespørgsel\"}. "
        "Sæt false for almindelig snak, kodeimplementering, holdninger, eller hvis web kun nævnes som et projektord.\n\n"
        f"Besked: {text[:500]}"
    )
    try:
        result = await brain.chat(
            [ChatMessage("system", prompt), ChatMessage("user", text)],
            temperature=0.0,
            max_tokens=120,
        )
    except Exception as exc:
        print(f"[web] ability-router failed: {exc}", flush=True)
        return WebSearchIntent(False)
    return _parse_web_search_intent(result, fallback_text=text)


def wants_web_search(text: str) -> bool:
    words = _fold_words(text)
    compact = "".join(words)
    if not compact:
        return False
    if any(word.startswith(("implement", "funktion", "feature")) for word in words) and not any(
        word in {"sog", "soeg", "soge", "soege", "google"} for word in words
    ):
        return False
    explicit_compact = (
        "websog",
        "sogpanettet",
        "soegpanettet",
        "sogepanettet",
        "soegepanettet",
        "sogonline",
        "soegonline",
        "sogeonline",
        "soegeonline",
        "sogefter",
        "soegefter",
        "slaaop",
        "slaop",
        "tjeknettet",
        "tjekpanettet",
        "findpanettet",
        "findonline",
    )
    if any(token in compact for token in explicit_compact):
        return True
    command_words = {"sog", "soeg", "soge", "soege", "soegning", "sogning", "google"}
    request_words = {"kan", "vil", "gider", "prov", "proev", "tjek", "find", "hent", "sla", "slaa"}
    return bool(command_words & set(words) and request_words & set(words))


def _looks_like_possible_web_need(text: str) -> bool:
    words = set(_fold_words(text))
    if not words:
        return False
    action_words = {
        "find",
        "tjek",
        "check",
        "hent",
        "laes",
        "las",
        "kig",
        "undersog",
        "undersoeg",
        "research",
        "nyeste",
        "nyt",
        "aktuel",
        "aktuelt",
        "frisk",
        "friske",
    }
    webish_words = {"nettet", "internet", "online", "web", "google", "kilder", "nyheder", "opdateringer"}
    current_info_words = {"nyeste", "aktuelt", "friske", "nyt"}
    return bool(words & action_words and words & webish_words) or bool(words & action_words and words & current_info_words)


def _parse_web_search_intent(raw: str, *, fallback_text: str) -> WebSearchIntent:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        return WebSearchIntent(False)
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return WebSearchIntent(False)
    if not bool(data.get("web_search")):
        return WebSearchIntent(False)
    query = str(data.get("query") or "").strip()
    if not query:
        query = extract_web_search_query(fallback_text)
    return WebSearchIntent(bool(query), query)


def extract_web_search_query(text: str) -> str:
    clean = " ".join(text.split()).strip()
    if not clean:
        return ""
    patterns = (
        r"^\s*(kan du|vil du|gider du|prøv at|proev at|stacky)\s+",
        r"\b(lige|gerne|for mig)\b",
        r"\b(web\s*search|websearch|websøg|websoeg)\b",
        r"\b(søg efter|søg på nettet efter|søg på nettet|søge efter|søge på nettet efter|søge på nettet|soeg efter|sog efter|soge efter|soege efter|sog panettet|soeg panettet|soge panettet|soege panettet|søg|soeg|sog|soge|soege)\b",
        r"\b(slå op|slaa op|slå det op|sla det op)\b",
        r"\b(tjek på nettet|tjek nettet|check nettet|find på nettet|find online|google)\b",
        r"\b(på nettet|paa nettet|online)\b",
    )
    result = clean
    for pattern in patterns:
        result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
    return " ".join(result.strip(" .?!,:;-").split())


def format_web_search_context(
    query: str,
    results: tuple[WebSearchResult, ...],
    *,
    error: str = "",
) -> str:
    clean_query = " ".join(query.split())
    if error:
        return (
            f'Web search-kontekst: Runtime forsogte at soge efter "{clean_query}", men sogningen fejlede: {error}. '
            "Svar kort og sig at websoegningen fejlede; opfind ikke friske fakta."
        )
    if not results:
        return (
            f'Web search-kontekst: Runtime sogte efter "{clean_query}", men fandt ingen brugbare resultater. '
            "Sig kort at du ikke fandt noget solidt; opfind ikke friske fakta."
        )

    lines = [
        f'Web search-kontekst: Runtime sogte paa nettet efter "{clean_query}".',
        "Brug kun disse resultater som frisk web-kontekst. Naevn kun kilder kort ved titel eller domaene; laes ikke lange URL'er op.",
    ]
    for index, result in enumerate(results, start=1):
        domain = urllib.parse.urlparse(result.url).netloc or result.url
        line = f"{index}. {result.title} ({domain}) - {result.url}"
        if result.snippet:
            line += f"\n   {result.snippet}"
        lines.append(line)
    return "\n".join(lines)


def _normalize_duckduckgo_url(raw_url: str) -> str:
    if not raw_url:
        return ""
    parsed = urllib.parse.urlparse(html.unescape(raw_url))
    query = urllib.parse.parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return urllib.parse.unquote(query["uddg"][0])
    if raw_url.startswith("//"):
        return "https:" + raw_url
    return raw_url


def _clean_html_text(text: str) -> str:
    return " ".join(html.unescape(text).split())


def _fold_words(text: str) -> list[str]:
    lowered = text.lower()
    replacements = {
        "æ": "ae",
        "ø": "o",
        "å": "a",
        "ä": "ae",
        "ö": "o",
        "ü": "u",
        "é": "e",
    }
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
    return re.sub(r"[^0-9a-z]+", " ", lowered).split()
