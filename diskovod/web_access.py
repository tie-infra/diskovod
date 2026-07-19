from __future__ import annotations

import asyncio
import html
import ipaddress
import re
import socket
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import aiohttp

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
    await _validate_public_url(url)
    timeout = aiohttp.ClientTimeout(total=20, connect=8)
    async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": USER_AGENT}) as session:
        async with session.get(url, allow_redirects=True, max_redirects=5) as response:
            await _validate_public_url(str(response.url))
            if response.status < 200 or response.status >= 300:
                raise WebAccessError(f"http_status_{response.status}")
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].casefold()
            if not (
                content_type.startswith("text/") or content_type in {"application/json", "application/xml"}
            ):
                raise WebAccessError("unsupported_content_type")
            body = await response.content.read(MAX_FETCH_BYTES + 1)
            if len(body) > MAX_FETCH_BYTES:
                raise WebAccessError("response_too_large")
            charset = response.charset or "utf-8"
            try:
                text = body.decode(charset, errors="replace")
            except LookupError:
                text = body.decode("utf-8", errors="replace")
            return text, {"url": str(response.url), "content_type": content_type}


async def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise WebAccessError("invalid_url")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as error:
        raise WebAccessError("invalid_url") from error
    loop = asyncio.get_running_loop()
    try:
        addresses = await loop.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except OSError as error:
        raise WebAccessError("dns_failure") from error
    if not addresses:
        raise WebAccessError("dns_failure")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise WebAccessError("private_address_rejected")


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
