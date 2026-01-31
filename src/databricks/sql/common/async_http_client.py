"""
Async HTTP client for Databricks SQL connector using aiohttp.

This module provides an async HTTP client that mirrors the functionality of
UnifiedHttpClient but uses aiohttp for non-blocking I/O operations.
"""

from __future__ import annotations

import asyncio
import logging
import ssl
import urllib.parse
import urllib.request
from typing import Dict, Any, Optional, Union

try:
    import aiohttp
    from aiohttp import ClientSession, TCPConnector, ClientTimeout, ClientResponse

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None
    ClientSession = None
    TCPConnector = None
    ClientTimeout = None
    ClientResponse = None

from databricks.sql.exc import RequestError
from databricks.sql.common.http import HttpMethod

logger = logging.getLogger(__name__)


class AsyncUnifiedHttpClient:
    """
    Async HTTP client for Databricks SQL connector using aiohttp.

    This client provides non-blocking HTTP communication with retry policies,
    connection pooling, SSL support, and proxy support. It is designed to be
    used with Python's asyncio for true async/await support.

    The client supports per-request proxy decisions, automatically routing requests
    through proxy or direct connections based on system proxy bypass rules and
    the target hostname of each request.
    """

    # Default retry settings
    DEFAULT_RETRY_DELAY_MIN = 1.0
    DEFAULT_RETRY_DELAY_MAX = 60.0
    DEFAULT_RETRY_COUNT = 3

    def __init__(self, client_context):
        """
        Initialize the async HTTP client.

        Args:
            client_context: ClientContext instance containing HTTP configuration

        Raises:
            ImportError: If aiohttp is not installed
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp is required for async support. "
                "Install it with: pip install databricks-sql-connector[async]"
            )

        self.config = client_context
        self._session: Optional[ClientSession] = None
        self._proxy_uri: Optional[str] = None
        self._proxy_auth: Optional[aiohttp.BasicAuth] = None
        self._ssl_context: Optional[ssl.SSLContext] = None
        self._connector: Optional[TCPConnector] = None

        # Retry settings
        self._retry_delay_min = getattr(
            client_context, "retry_delay_min", self.DEFAULT_RETRY_DELAY_MIN
        )
        self._retry_delay_max = getattr(
            client_context, "retry_delay_max", self.DEFAULT_RETRY_DELAY_MAX
        )
        self._max_retries = getattr(
            client_context, "retry_stop_after_attempts_count", self.DEFAULT_RETRY_COUNT
        )

        # Parse hostname for proxy detection
        parsed_url = urllib.parse.urlparse(self.config.hostname)
        self.scheme = parsed_url.scheme or "https"
        self.host = parsed_url.hostname

        # Set up SSL context
        self._setup_ssl_context()

        # Detect proxy configuration
        self._setup_proxy()

    def _setup_ssl_context(self):
        """Set up SSL context based on configuration."""
        if self.config.ssl_options:
            self._ssl_context = ssl.create_default_context()

            # Configure SSL verification
            if not self.config.ssl_options.tls_verify:
                self._ssl_context.check_hostname = False
                self._ssl_context.verify_mode = ssl.CERT_NONE
            elif not self.config.ssl_options.tls_verify_hostname:
                self._ssl_context.check_hostname = False
                self._ssl_context.verify_mode = ssl.CERT_REQUIRED

            # Load custom CA file if specified
            if self.config.ssl_options.tls_trusted_ca_file:
                self._ssl_context.load_verify_locations(
                    self.config.ssl_options.tls_trusted_ca_file
                )

            # Load client certificate if specified
            if (
                self.config.ssl_options.tls_client_cert_file
                and self.config.ssl_options.tls_client_cert_key_file
            ):
                self._ssl_context.load_cert_chain(
                    self.config.ssl_options.tls_client_cert_file,
                    self.config.ssl_options.tls_client_cert_key_file,
                    self.config.ssl_options.tls_client_cert_key_password,
                )

    def _setup_proxy(self):
        """Detect and configure proxy settings."""
        try:
            from databricks.sql.common.http_utils import detect_and_parse_proxy

            proxy_url, proxy_auth = detect_and_parse_proxy(
                self.scheme,
                self.host,
                skip_bypass=True,
                proxy_auth_method=getattr(self.config, "proxy_auth_method", None),
            )

            if proxy_url:
                self._proxy_uri = proxy_url
                if proxy_auth:
                    # Parse proxy auth for aiohttp
                    # proxy_auth is typically a dict with 'Proxy-Authorization' header
                    auth_header = proxy_auth.get("Proxy-Authorization", "")
                    if auth_header.startswith("Basic "):
                        import base64

                        decoded = base64.b64decode(auth_header[6:]).decode()
                        if ":" in decoded:
                            username, password = decoded.split(":", 1)
                            self._proxy_auth = aiohttp.BasicAuth(username, password)
                logger.debug("Initialized with proxy support: %s", proxy_url)
            else:
                logger.debug(
                    "No system proxy detected, using direct connections only"
                )
        except Exception as e:
            logger.debug("Error detecting system proxy configuration: %s", e)

    async def _ensure_session(self):
        """Create aiohttp session if not exists or closed."""
        if self._session is None or self._session.closed:
            # Create connector with connection pooling
            self._connector = TCPConnector(
                limit=getattr(self.config, "pool_maxsize", 10),
                limit_per_host=getattr(self.config, "pool_maxsize", 10),
                ssl=self._ssl_context,
            )

            # Create timeout configuration
            socket_timeout = getattr(self.config, "socket_timeout", None)
            timeout = (
                ClientTimeout(total=socket_timeout) if socket_timeout else ClientTimeout()
            )

            # Prepare default headers
            headers = self._prepare_headers()

            self._session = ClientSession(
                connector=self._connector,
                timeout=timeout,
                headers=headers,
            )

    def _prepare_headers(
        self, headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Prepare headers for the request, including User-Agent."""
        request_headers = {}

        if hasattr(self.config, "user_agent") and self.config.user_agent:
            request_headers["User-Agent"] = self.config.user_agent

        if headers:
            request_headers.update(headers)

        return request_headers

    def _should_use_proxy(self, target_host: str) -> bool:
        """
        Determine if a request to the target host should use proxy.

        Args:
            target_host: The hostname of the target URL

        Returns:
            True if proxy should be used, False for direct connection
        """
        if not self._proxy_uri:
            return False

        try:
            # proxy_bypass returns True if the host should BYPASS the proxy
            return not urllib.request.proxy_bypass(target_host)
        except Exception as e:
            logger.debug("Error checking proxy bypass for host %s: %s", target_host, e)
            return True

    def _get_proxy_for_url(self, url: str) -> Optional[str]:
        """
        Get the appropriate proxy for the given URL.

        Args:
            url: The target URL

        Returns:
            Proxy URL if proxy should be used, None otherwise
        """
        parsed_url = urllib.parse.urlparse(url)
        target_host = parsed_url.hostname

        if target_host and self._should_use_proxy(target_host):
            logger.debug("Using proxy for request to %s", target_host)
            return self._proxy_uri
        else:
            logger.debug("Using direct connection for request to %s", target_host)
            return None

    def _get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff."""
        delay = min(
            self._retry_delay_max, self._retry_delay_min * (2**attempt)
        )
        return delay

    def _should_retry(self, status_code: int, attempt: int) -> bool:
        """Determine if a request should be retried based on status code."""
        if attempt >= self._max_retries:
            return False

        # Retry on server errors and rate limiting
        return status_code in [429, 500, 502, 503, 504]

    async def request(
        self,
        method: HttpMethod,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Union[bytes, str]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> ClientResponse:
        """
        Make an async HTTP request with retry support.

        Args:
            method: HTTP method (HttpMethod.GET, HttpMethod.POST, etc.)
            url: URL to request
            headers: Optional headers dict
            body: Optional request body (bytes or string)
            json: Optional JSON body (will be serialized)

        Returns:
            aiohttp.ClientResponse: The HTTP response object

        Raises:
            RequestError: If the request fails after all retries
        """
        await self._ensure_session()

        request_headers = self._prepare_headers(headers)
        proxy = self._get_proxy_for_url(url)

        last_error: Optional[Exception] = None
        last_status: Optional[int] = None

        for attempt in range(self._max_retries + 1):
            try:
                logger.debug(
                    "Making async %s request to %s (attempt %d/%d)",
                    method.value,
                    urllib.parse.urlparse(url).netloc,
                    attempt + 1,
                    self._max_retries + 1,
                )

                kwargs: Dict[str, Any] = {
                    "headers": request_headers,
                }

                if proxy:
                    kwargs["proxy"] = proxy
                    if self._proxy_auth:
                        kwargs["proxy_auth"] = self._proxy_auth

                if body is not None:
                    kwargs["data"] = body
                elif json is not None:
                    kwargs["json"] = json

                async with self._session.request(
                    method.value, url, **kwargs
                ) as response:
                    # Read response data to ensure it's available
                    data = await response.read()
                    last_status = response.status

                    # Check if we should retry
                    if self._should_retry(response.status, attempt):
                        delay = self._get_retry_delay(attempt)
                        logger.debug(
                            "Request failed with status %d, retrying in %.2f seconds",
                            response.status,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    # Create a response-like object that contains the data
                    # since the response context will be closed
                    return _AsyncResponse(
                        status=response.status,
                        headers=dict(response.headers),
                        data=data,
                    )

            except aiohttp.ClientError as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = self._get_retry_delay(attempt)
                    logger.debug(
                        "Request failed with error %s, retrying in %.2f seconds",
                        str(e),
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise RequestError(f"HTTP request failed: {e}")

            except Exception as e:
                logger.error("Unexpected error during HTTP request: %s", e)
                raise RequestError(f"HTTP request error: {e}")

        # If we get here, all retries failed
        context = {}
        if last_status is not None:
            context["http-code"] = last_status

        error_msg = f"HTTP request failed after {self._max_retries + 1} attempts"
        if last_error:
            error_msg += f": {last_error}"
        elif last_status:
            error_msg += f" with status code {last_status}"

        raise RequestError(error_msg, context=context)

    async def close(self):
        """Close the aiohttp session and connector."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        if self._connector and not self._connector.closed:
            await self._connector.close()
            self._connector = None

    async def __aenter__(self) -> "AsyncUnifiedHttpClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def using_proxy(self) -> bool:
        """Check if proxy support is available."""
        return self._proxy_uri is not None

    @property
    def proxy_uri(self) -> Optional[str]:
        """Get the configured proxy URI, if any."""
        return self._proxy_uri


class _AsyncResponse:
    """
    A simple response wrapper that holds response data after the aiohttp context closes.

    This is needed because aiohttp's ClientResponse is a context manager and the
    response body becomes unavailable after the context exits.
    """

    def __init__(self, status: int, headers: Dict[str, str], data: bytes):
        self.status = status
        self.headers = headers
        self.data = data

    async def json(self) -> Any:
        """Parse response body as JSON."""
        import json as json_module

        return json_module.loads(self.data.decode("utf-8"))

    def text(self) -> str:
        """Get response body as text."""
        return self.data.decode("utf-8")
