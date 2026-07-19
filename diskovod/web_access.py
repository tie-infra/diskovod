from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx2

from .http_client import PublicAsyncHTTPTransport, PublicNetworkError

MAX_FETCH_BYTES = 1_000_000
USER_AGENT = "Diskovod/1.0 (+local Discord assistant)"


class WebAccessError(RuntimeError):
    pass


async def search_web(query: str, limit: int = 5) -> list[dict[str, str]]:
    query = query.strip()
    if not query:
        raise WebAccessError("empty_query")
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query[:500])}"
    body, _ = await _request_text(url)
    parser = _SearchParser(max(1, min(limit, 8)))
    parser.feed(body)
    return parser.results


async def fetch_url(url: str) -> dict[str, Any]:
    body, metadata = await _request_text(url)
    content_type = metadata["content_type"]
    if "html" in content_type:
        parser = _TextParser()
        parser.feed(body)
        text = parser.text()
    else:
        text = body
    return {
        "url": metadata["url"],
        "content_type": content_type,
        "title": metadata.get("title") or "",
        "text": text[:40_000],
        "truncated": len(text) > 40_000,
    }


async def _request_text(url: str) -> tuple[str, dict[str, str]]:
    timeout = httpx2.Timeout(20, connect=8)
    try:
        async with httpx2.AsyncClient(
            transport=PublicAsyncHTTPTransport(),
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            max_redirects=5,
            trust_env=False,
        ) as session:
            async with session.stream("GET", url) as response:
                if response.status_code < 200 or response.status_code >= 300:
                    raise WebAccessError(f"http_status_{response.status_code}")
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].casefold()
                if not (
                    content_type.startswith("text/")
                    or content_type in {"application/json", "application/xml"}
                ):
                    raise WebAccessError("unsupported_content_type")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_FETCH_BYTES:
                        raise WebAccessError("response_too_large")
                charset = response.encoding or "utf-8"
                try:
                    text = body.decode(charset, errors="replace")
                except LookupError:
                    text = body.decode("utf-8", errors="replace")
                return text, {"url": str(response.url), "content_type": content_type}
    except WebAccessError:
        raise
    except PublicNetworkError as error:
        raise WebAccessError(str(error)) from error
    except httpx2.InvalidURL as error:
        raise WebAccessError("invalid_url") from error
    except httpx2.TooManyRedirects as error:
        raise WebAccessError("too_many_redirects") from error
    except httpx2.TimeoutException as error:
        raise WebAccessError("request_timeout") from error
    except httpx2.HTTPError as error:
        raise WebAccessError("request_failed") from error


class _SearchParser(HTMLParser):
    def __init__(self, limit: int):
        super().__init__(convert_charrefs=True)
        self.limit = limit
        self.results: list[dict[str, str]] = []
        self._url: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = (values.get("class") or "").split()
        if tag == "a" and "result__a" in classes and len(self.results) < self.limit:
            self._url = _decode_duckduckgo_url(values.get("href") or "")
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._url is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._url is not None:
            title = " ".join("".join(self._text).split())
            if title and self._url.startswith(("http://", "https://")):
                self.results.append({"title": html.unescape(title), "url": self._url})
            self._url = None
            self._text = []


def _decode_duckduckgo_url(url: str) -> str:
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if parsed.hostname and parsed.hostname.endswith("duckduckgo.com"):
        candidate = parse_qs(parsed.query).get("uddg", [""])[0]
        if candidate:
            return unquote(candidate)
    return url


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden += 1
        elif tag in {"p", "br", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)

    def text(self) -> str:
        value = html.unescape(" ".join(self.parts))
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n\s*\n+", "\n\n", value)
        return value.strip()
