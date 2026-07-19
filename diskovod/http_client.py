from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import socket
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import httpcore2
import httpx2

USER_AGENT = "Diskovod/1.0 (+local Discord assistant)"


class PublicNetworkError(RuntimeError):
    """A stable error raised when an untrusted URL cannot be dialed safely."""


AddressInfo = tuple[int, int, int, str, tuple[Any, ...]]
Resolver = Callable[[str, int], Awaitable[list[AddressInfo]]]


@dataclass(frozen=True, slots=True)
class PublicHTTPResponse:
    url: str
    status_code: int
    headers: httpx2.Headers
    content: bytes
    encoding: str


class PublicHTTP(Protocol):
    async def get(
        self,
        url: str,
        *,
        max_bytes: int,
        timeout: httpx2.Timeout | float | None = None,
    ) -> PublicHTTPResponse: ...


async def _resolve(host: str, port: int) -> list[AddressInfo]:
    loop = asyncio.get_running_loop()
    return await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)


class PublicNetworkBackend(httpcore2.AsyncNetworkBackend):
    """Resolve, validate, and pin every TCP connection to a public IP address."""

    def __init__(
        self,
        *,
        resolver: Resolver = _resolve,
        backend: httpcore2.AsyncNetworkBackend | None = None,
    ) -> None:
        self._resolver = resolver
        self._backend = backend or httpcore2.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore2.SOCKET_OPTION] | None = None,
    ) -> httpcore2.AsyncNetworkStream:
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        try:
            async with asyncio.timeout(timeout):
                try:
                    addresses = await self._resolver(host, port)
                except OSError as error:
                    raise PublicNetworkError("dns_failure") from error
                if not addresses:
                    raise PublicNetworkError("dns_failure")

                public_addresses: list[str] = []
                for family, _socket_type, _protocol, _canonical_name, sockaddr in addresses:
                    if family not in {socket.AF_INET, socket.AF_INET6}:
                        continue
                    try:
                        candidate = ipaddress.ip_address(sockaddr[0])
                    except ValueError:
                        continue
                    normalized = str(candidate)
                    if candidate.is_global and normalized not in public_addresses:
                        public_addresses.append(normalized)
                if not public_addresses:
                    raise PublicNetworkError("private_address_rejected")

                last_error: Exception | None = None
                for address in public_addresses:
                    remaining = None if deadline is None else max(0.0, deadline - loop.time())
                    try:
                        # Dial the validated numeric address. HTTP core retains the original
                        # origin hostname and uses it later for TLS SNI and verification.
                        return await self._backend.connect_tcp(
                            address,
                            port,
                            timeout=remaining,
                            local_address=local_address,
                            socket_options=socket_options,
                        )
                    except (httpcore2.ConnectError, httpcore2.ConnectTimeout) as error:
                        last_error = error
                assert last_error is not None
                raise last_error
        except TimeoutError as error:
            raise httpcore2.ConnectTimeout("Timed out while resolving or connecting") from error

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore2.SOCKET_OPTION] | None = None,
    ) -> httpcore2.AsyncNetworkStream:
        del path, timeout, socket_options
        raise PublicNetworkError("private_address_rejected")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


_CORE_EXCEPTION_MAP: tuple[tuple[type[Exception], type[httpx2.HTTPError]], ...] = (
    (httpcore2.ConnectTimeout, httpx2.ConnectTimeout),
    (httpcore2.ReadTimeout, httpx2.ReadTimeout),
    (httpcore2.WriteTimeout, httpx2.WriteTimeout),
    (httpcore2.PoolTimeout, httpx2.PoolTimeout),
    (httpcore2.ConnectError, httpx2.ConnectError),
    (httpcore2.ReadError, httpx2.ReadError),
    (httpcore2.WriteError, httpx2.WriteError),
    (httpcore2.UnsupportedProtocol, httpx2.UnsupportedProtocol),
    (httpcore2.LocalProtocolError, httpx2.LocalProtocolError),
    (httpcore2.RemoteProtocolError, httpx2.RemoteProtocolError),
    (httpcore2.ProxyError, httpx2.ProxyError),
    (httpcore2.TimeoutException, httpx2.TimeoutException),
    (httpcore2.NetworkError, httpx2.NetworkError),
    (httpcore2.ProtocolError, httpx2.ProtocolError),
)


@contextlib.contextmanager
def _map_core_exceptions(request: httpx2.Request | None = None) -> Iterator[None]:
    try:
        yield
    except Exception as error:
        for source, target in _CORE_EXCEPTION_MAP:
            if isinstance(error, source):
                raise target(str(error), request=request) from error
        raise


class _ResponseStream(httpx2.AsyncByteStream):
    def __init__(self, stream: AsyncIterable[bytes], request: httpx2.Request) -> None:
        self._stream = stream
        self._request = request

    async def __aiter__(self) -> AsyncIterator[bytes]:
        with _map_core_exceptions(self._request):
            async for chunk in self._stream:
                yield chunk

    async def aclose(self) -> None:
        if hasattr(self._stream, "aclose"):
            await self._stream.aclose()


class PublicAsyncHTTPTransport(httpx2.AsyncBaseTransport):
    """HTTPX2 transport whose actual connector enforces the public-IP policy."""

    def __init__(
        self,
        *,
        network_backend: PublicNetworkBackend | None = None,
        limits: httpx2.Limits | None = None,
        proxy: httpx2.Proxy | None = None,
    ) -> None:
        limits = limits or httpx2.Limits(max_connections=100, max_keepalive_connections=20)
        ssl_context = httpx2.create_ssl_context(trust_env=True)
        backend = network_backend or PublicNetworkBackend()
        pool_options = {
            "ssl_context": ssl_context,
            "max_connections": limits.max_connections,
            "max_keepalive_connections": limits.max_keepalive_connections,
            "keepalive_expiry": limits.keepalive_expiry,
            "http1": True,
            "http2": True,
            "network_backend": backend,
        }
        if proxy is None:
            self._pool = httpcore2.AsyncConnectionPool(**pool_options)
        elif proxy.url.scheme in {"http", "https"}:
            proxy_options = (
                {"proxy_ssl_context": proxy.ssl_context or ssl_context} if proxy.url.scheme == "https" else {}
            )
            self._pool = httpcore2.AsyncHTTPProxy(
                proxy_url=httpcore2.URL(
                    scheme=proxy.url.raw_scheme,
                    host=proxy.url.raw_host,
                    port=proxy.url.port,
                    target=proxy.url.raw_path,
                ),
                proxy_auth=proxy.raw_auth,
                proxy_headers=proxy.headers.raw,
                **pool_options,
                **proxy_options,
            )
        elif proxy.url.scheme in {"socks5", "socks5h"}:
            self._pool = httpcore2.AsyncSOCKSProxy(
                proxy_url=httpcore2.URL(
                    scheme=proxy.url.raw_scheme,
                    host=proxy.url.raw_host,
                    port=proxy.url.port,
                    target=proxy.url.raw_path,
                ),
                proxy_auth=proxy.raw_auth,
                **pool_options,
            )
        else:
            raise ValueError(f"Unsupported untrusted-URL proxy scheme: {proxy.url.scheme}")

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        if (
            request.url.scheme not in {"http", "https"}
            or not request.url.host
            or request.url.username
            or request.url.password
        ):
            raise PublicNetworkError("invalid_url")
        assert isinstance(request.stream, httpx2.AsyncByteStream)
        core_request = httpcore2.Request(
            method=request.method,
            url=httpcore2.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        with _map_core_exceptions(request):
            response = await self._pool.handle_async_request(core_request)
        return httpx2.Response(
            status_code=response.status,
            headers=response.headers,
            stream=_ResponseStream(response.stream, request),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._pool.aclose()


class PublicHTTPClient:
    """Application-owned HTTP/2 client for bounded reads from untrusted URLs."""

    def __init__(self, *, network_backend: PublicNetworkBackend | None = None) -> None:
        backend = network_backend or PublicNetworkBackend()
        mounts = _environment_proxy_mounts(backend)
        self._client = httpx2.AsyncClient(
            transport=PublicAsyncHTTPTransport(network_backend=backend),
            mounts=mounts,
            timeout=httpx2.Timeout(20, connect=8),
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            max_redirects=5,
            trust_env=True,
        )

    async def get(
        self,
        url: str,
        *,
        max_bytes: int,
        timeout: httpx2.Timeout | float | None = None,
    ) -> PublicHTTPResponse:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        options = {"timeout": timeout} if timeout is not None else {}
        async with self._client.stream("GET", url, **options) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    if int(content_length) > max_bytes:
                        raise PublicNetworkError("response_too_large")
                except ValueError:
                    pass
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > max_bytes:
                    raise PublicNetworkError("response_too_large")
            return PublicHTTPResponse(
                url=str(response.url),
                status_code=response.status_code,
                headers=response.headers,
                content=bytes(body),
                encoding=response.encoding or "utf-8",
            )

    async def close(self) -> None:
        await self._client.aclose()


def _environment_proxy_mounts(
    network_backend: PublicNetworkBackend,
) -> dict[str, PublicAsyncHTTPTransport | None]:
    # HTTPX2 intentionally disables automatic environment proxies when an explicit
    # transport is supplied. Reuse its pinned environment/NO_PROXY interpretation,
    # but replace every generated proxy transport with our guarded connector.
    from httpx2._utils import get_environment_proxies

    mounts: dict[str, PublicAsyncHTTPTransport | None] = {}
    for pattern, proxy_url in get_environment_proxies().items():
        mounts[pattern] = (
            None
            if proxy_url is None
            else PublicAsyncHTTPTransport(
                network_backend=network_backend,
                proxy=httpx2.Proxy(proxy_url),
            )
        )
    return mounts
