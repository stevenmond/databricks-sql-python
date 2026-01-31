"""
Async Thrift HTTP transport using aiohttp.

This module provides an async-compatible Thrift HTTP transport that uses aiohttp
instead of urllib3 for non-blocking HTTP operations.
"""

from __future__ import annotations

import logging
import ssl
import urllib.parse
from io import BytesIO
from typing import Dict, Optional, Any

try:
    import aiohttp
except ImportError:
    aiohttp = None

from thrift.transport.TTransport import TTransportBase

from databricks.sql.types import SSLOptions
from databricks.sql.common.http_utils import detect_and_parse_proxy

logger = logging.getLogger(__name__)


class AsyncTHttpClient(TTransportBase):
    """
    Async HTTP client transport for Thrift using aiohttp.

    This transport implements the Thrift TTransport interface but uses aiohttp
    for async HTTP operations. It maintains the same API as the sync THttpClient
    but provides async flush() method for non-blocking requests.
    """

    def __init__(
        self,
        auth_provider,
        uri_or_host: str,
        port: Optional[int] = None,
        path: Optional[str] = None,
        ssl_options: Optional[SSLOptions] = None,
        timeout: float = 900.0,
        **kwargs,
    ):
        """
        Initialize async Thrift HTTP transport.

        Args:
            auth_provider: Authentication provider for adding auth headers
            uri_or_host: Full URI (https://host:port/path) or just hostname
            port: Port number (deprecated, use full URI)
            path: HTTP path (deprecated, use full URI)
            ssl_options: SSL configuration options
            timeout: Request timeout in seconds
            **kwargs: Additional options (proxy_auth_method, etc.)
        """
        if aiohttp is None:
            raise ImportError(
                "aiohttp is required for async support. "
                "Install with: pip install databricks-sql-connector[async]"
            )

        self._ssl_options = ssl_options
        self._auth_provider = auth_provider
        self._timeout = timeout

        # Parse URI
        if port is not None:
            # Legacy format: host, port, path separately
            self.host = uri_or_host
            self.port = port
            self.path = path or "/"
            self.scheme = "https"
        else:
            # Modern format: full URI
            parsed = urllib.parse.urlsplit(uri_or_host)
            self.scheme = parsed.scheme or "https"
            self.host = parsed.hostname
            self.port = parsed.port or (443 if self.scheme == "https" else 80)
            self.path = parsed.path
            if parsed.query:
                self.path += f"?{parsed.query}"

        # Handle proxy settings
        proxy_auth_method = kwargs.get("_proxy_auth_method")
        proxy_uri, proxy_auth = detect_and_parse_proxy(
            self.scheme, self.host, proxy_auth_method=proxy_auth_method
        )
        self.proxy_uri = proxy_uri
        self.proxy_auth = proxy_auth

        # Build base URL
        self._base_url = f"{self.scheme}://{self.host}:{self.port}{self.path}"

        # Initialize buffers
        self._wbuf = BytesIO()
        self._rbuf = BytesIO()

        # Response state
        self.code: Optional[int] = None
        self.message: Optional[str] = None
        self.headers: Dict[str, str] = {}

        # Custom headers (set via setCustomHeaders)
        self._custom_headers: Dict[str, str] = {}

        # Session management
        self._session: Optional[aiohttp.ClientSession] = None

    def setCustomHeaders(self, headers: Dict[str, str]) -> None:
        """Set custom HTTP headers to be sent with requests."""
        self._custom_headers = dict(headers)

    def setTimeout(self, timeout_ms: Optional[float]) -> None:
        """Set request timeout in milliseconds."""
        if timeout_ms is not None:
            self._timeout = timeout_ms / 1000.0  # Convert to seconds

    def isOpen(self) -> bool:
        """Check if transport is open."""
        return self._session is not None

    async def open(self) -> None:
        """Open the transport (create aiohttp session)."""
        if self._session is not None:
            return

        # Create SSL context if using HTTPS
        ssl_context: Optional[ssl.SSLContext] = None
        if self.scheme == "https" and self._ssl_options:
            ssl_context = self._ssl_options.create_ssl_context()

        # Create connector with SSL settings
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=10,  # Connection pool size
        )

        # Create timeout
        timeout = aiohttp.ClientTimeout(total=self._timeout)

        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        )

    async def close(self) -> None:
        """Close the transport (close aiohttp session)."""
        if self._session is not None:
            await self._session.close()
            self._session = None

        # Clear response buffer
        self._rbuf = BytesIO()

    def read(self, sz: int) -> bytes:
        """Read sz bytes from response buffer (synchronous)."""
        return self._rbuf.read(sz)

    def write(self, buf: bytes) -> None:
        """Write bytes to request buffer (synchronous)."""
        self._wbuf.write(buf)

    async def flush(self) -> None:
        """
        Send the buffered request data asynchronously.

        This is the main async method that sends the Thrift request over HTTP
        using aiohttp and reads the response into the read buffer.
        """
        if self._session is None:
            await self.open()

        # Get data from write buffer
        data = self._wbuf.getvalue()
        self._wbuf = BytesIO()

        # Build headers
        headers = {
            "Content-Type": "application/x-thrift",
            "Content-Length": str(len(data)),
        }

        # Add auth headers
        self._auth_provider.add_headers(headers)

        # Add custom headers
        if self._custom_headers:
            headers.update(self._custom_headers)

        # Add proxy auth headers if needed
        if self.proxy_auth:
            headers.update(self.proxy_auth)

        try:
            async with self._session.post(
                self._base_url,
                data=data,
                headers=headers,
                proxy=self.proxy_uri,
            ) as response:
                self.code = response.status
                self.message = response.reason
                self.headers = dict(response.headers)

                # Read response body into buffer
                response_data = await response.read()
                self._rbuf = BytesIO(response_data)

                logger.debug(
                    "Async HTTP Response: status=%d, message=%s",
                    self.code,
                    self.message,
                )

        except aiohttp.ClientError as e:
            logger.error("Async HTTP request failed: %s", e)
            raise

    def flush_sync(self) -> None:
        """
        Synchronous flush - raises error since this is async-only transport.

        This method exists for interface compatibility but should not be used.
        Use flush() with await instead.
        """
        raise RuntimeError(
            "AsyncTHttpClient.flush_sync() called. "
            "This transport is async-only. Use 'await transport.flush()' instead."
        )

    def using_proxy(self) -> bool:
        """Check if proxy is being used."""
        return self.proxy_uri is not None


class AsyncThriftProtocol:
    """
    Wrapper around TBinaryProtocol that provides async request/response handling.

    This wraps the standard TBinaryProtocol but provides async methods for
    making Thrift RPC calls using the AsyncTHttpClient transport.
    """

    def __init__(self, transport: AsyncTHttpClient):
        """
        Initialize async protocol wrapper.

        Args:
            transport: AsyncTHttpClient transport instance
        """
        import thrift.protocol.TBinaryProtocol as TBinaryProtocol

        self._transport = transport
        self._protocol = TBinaryProtocol.TBinaryProtocol(transport)

    @property
    def trans(self):
        """Get the underlying transport."""
        return self._transport

    async def make_request(self, write_func, read_func):
        """
        Make an async Thrift request.

        Args:
            write_func: Function that writes the request to the protocol
            read_func: Function that reads the response from the protocol

        Returns:
            The result from read_func
        """
        # Write the request using the sync protocol methods
        write_func(self._protocol)

        # Flush asynchronously
        await self._transport.flush()

        # Read the response using the sync protocol methods
        return read_func(self._protocol)
