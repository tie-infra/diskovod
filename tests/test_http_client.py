from __future__ import annotations

import socket
from collections.abc import Iterable

import httpcore2
import httpx2
import pytest

from diskovod.http_client import (
    AddressInfo,
    PublicAsyncHTTPTransport,
    PublicNetworkBackend,
    PublicNetworkError,
)


def _address(family: int, value: str, port: int = 80) -> AddressInfo:
    sockaddr = (value, port, 0, 0) if family == socket.AF_INET6 else (value, port)
    return family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr


class _Stream(httpcore2.AsyncNetworkStream):
    def __init__(self, response: bytes = b"") -> None:
        self.response = response
        self.writes: list[bytes] = []

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        del timeout
        chunk, self.response = self.response[:max_bytes], self.response[max_bytes:]
        return chunk

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        del timeout
        self.writes.append(buffer)

    async def aclose(self) -> None:
        pass

    async def start_tls(
        self,
        ssl_context,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore2.AsyncNetworkStream:
        del ssl_context, server_hostname, timeout
        return self

    def get_extra_info(self, info: str):
        if info == "is_readable":
            return False
        return None


class _Backend(httpcore2.AsyncNetworkBackend):
    def __init__(self, responses: list[bytes] | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.responses = list(responses or [])

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore2.SOCKET_OPTION] | None = None,
    ) -> httpcore2.AsyncNetworkStream:
        del timeout, local_address, socket_options
        self.calls.append((host, port))
        response = self.responses.pop(0) if self.responses else b""
        return _Stream(response)

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore2.SOCKET_OPTION] | None = None,
    ) -> httpcore2.AsyncNetworkStream:
        raise AssertionError("Unix sockets must not be used")

    async def sleep(self, seconds: float) -> None:
        del seconds


@pytest.mark.asyncio
async def test_connector_pins_ipv4_and_ipv6_addresses() -> None:
    dialer = _Backend()

    async def resolve(host: str, port: int) -> list[AddressInfo]:
        del host
        return [
            _address(socket.AF_INET, "93.184.216.34", port),
            _address(socket.AF_INET6, "2606:2800:220:1:248:1893:25c8:1946", port),
        ]

    backend = PublicNetworkBackend(resolver=resolve, backend=dialer)
    await backend.connect_tcp("example.com", 443)
    assert dialer.calls == [("93.184.216.34", 443)]

    async def resolve_ipv6(host: str, port: int) -> list[AddressInfo]:
        del host
        return [_address(socket.AF_INET6, "2606:2800:220:1:248:1893:25c8:1946", port)]

    backend = PublicNetworkBackend(resolver=resolve_ipv6, backend=dialer)
    await backend.connect_tcp("example.com", 443)
    assert dialer.calls[-1] == ("2606:2800:220:1:248:1893:25c8:1946", 443)


@pytest.mark.asyncio
async def test_connector_never_dials_non_global_addresses() -> None:
    dialer = _Backend()

    async def resolve(host: str, port: int) -> list[AddressInfo]:
        del host
        return [
            _address(socket.AF_INET, "127.0.0.1", port),
            _address(socket.AF_INET6, "::1", port),
            _address(socket.AF_INET, "10.0.0.1", port),
        ]

    backend = PublicNetworkBackend(resolver=resolve, backend=dialer)
    with pytest.raises(PublicNetworkError, match="private_address_rejected"):
        await backend.connect_tcp("internal.example", 80)
    assert dialer.calls == []


@pytest.mark.asyncio
async def test_connector_filters_private_answers_and_pins_public_answer() -> None:
    dialer = _Backend()

    async def resolve(host: str, port: int) -> list[AddressInfo]:
        del host
        return [
            _address(socket.AF_INET, "192.168.1.1", port),
            _address(socket.AF_INET, "93.184.216.34", port),
        ]

    backend = PublicNetworkBackend(resolver=resolve, backend=dialer)
    await backend.connect_tcp("mixed.example", 80)
    assert dialer.calls == [("93.184.216.34", 80)]


@pytest.mark.asyncio
async def test_redirect_to_private_origin_is_rejected_before_second_dial() -> None:
    redirect = (
        b"HTTP/1.1 302 Found\r\n"
        b"Location: http://internal.test/secret\r\n"
        b"Content-Length: 0\r\n"
        b"Connection: close\r\n\r\n"
    )
    dialer = _Backend([redirect])

    async def resolve(host: str, port: int) -> list[AddressInfo]:
        if host == "public.test":
            return [_address(socket.AF_INET, "93.184.216.34", port)]
        if host == "internal.test":
            return [_address(socket.AF_INET, "127.0.0.1", port)]
        raise AssertionError(host)

    backend = PublicNetworkBackend(resolver=resolve, backend=dialer)
    transport = PublicAsyncHTTPTransport(network_backend=backend)
    async with httpx2.AsyncClient(transport=transport, follow_redirects=True) as client:
        with pytest.raises(PublicNetworkError, match="private_address_rejected"):
            await client.get("http://public.test/start")
    assert dialer.calls == [("93.184.216.34", 80)]
