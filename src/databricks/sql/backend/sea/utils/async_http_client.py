"""
Async HTTP client for Statement Execution API (SEA).

This module provides an async HTTP client that mirrors the functionality of
SeaHttpClient but uses aiohttp for non-blocking I/O operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING

try:
    import aiohttp
    from aiohttp import ClientSession, TCPConnector, ClientTimeout

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None
    ClientSession = None
    TCPConnector = None
    ClientTimeout = None

from databricks.sql.auth.authenticators import AuthProvider
from databricks.sql.auth.retry import CommandType
from databricks.sql.types import SSLOptions
from databricks.sql.exc import RequestError
from databricks.sql.common.url_utils import normalize_host_with_protocol

if TYPE_CHECKING:
    from databricks.sql.common.async_http_client import AsyncUnifiedHttpClient

logger = logging.getLogger(__name__)


class AsyncSeaHttpClient:
    """
    Async HTTP client for Statement Execution API (SEA).

    This client uses aiohttp for non-blocking HTTP communication with retry policies
    and connection pooling. It provides true async/await support for SEA operations.
    """

    # Default retry settings
    DEFAULT_RETRY_DELAY_MIN = 1.0
    DEFAULT_RETRY_DELAY_MAX = 60.0
    DEFAULT_RETRY_COUNT = 30
    DEFAULT_RETRY_DURATION = 900.0

    def __init__(
        self,
        server_hostname: str,
        port: int,
        http_path: str,
        http_headers: List[Tuple[str, str]],
        auth_provider: AuthProvider,
        ssl_options: SSLOptions,
        http_client: Optional["AsyncUnifiedHttpClient"] = None,
        **kwargs,
    ):
        """
        Initialize the async SEA HTTP client.

        Args:
            server_hostname: Hostname of the Databricks server
            port: Port number for the connection
            http_path: HTTP path for the connection
            http_headers: List of HTTP headers to include in requests
            auth_provider: Authentication provider
            ssl_options: SSL configuration options
            http_client: Optional shared AsyncUnifiedHttpClient instance
            **kwargs: Additional keyword arguments including retry policy settings

        Raises:
            ImportError: If aiohttp is not installed
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp is required for async support. "
                "Install it with: pip install databricks-sql-connector[async]"
            )

        self.server_hostname = server_hostname
        self.port = port or 443
        self.http_path = http_path
        self.auth_provider = auth_provider
        self.ssl_options = ssl_options

        # Build base URL using url_utils for consistent normalization
        normalized_host = normalize_host_with_protocol(server_hostname)
        self.base_url = f"{normalized_host}:{self.port}"

        # Setup headers
        self.headers: Dict[str, str] = dict(http_headers)
        self.headers.update({"Content-Type": "application/json"})

        # Extract retry policy settings
        self._retry_delay_min = kwargs.get("_retry_delay_min", self.DEFAULT_RETRY_DELAY_MIN)
        self._retry_delay_max = kwargs.get("_retry_delay_max", self.DEFAULT_RETRY_DELAY_MAX)
        self._max_retries = kwargs.get(
            "_retry_stop_after_attempts_count", self.DEFAULT_RETRY_COUNT
        )
        self._retry_duration = kwargs.get(
            "_retry_stop_after_attempts_duration", self.DEFAULT_RETRY_DURATION
        )
        self.force_dangerous_codes = kwargs.get("_retry_dangerous_codes", [])

        # Store shared http client if provided
        self._shared_http_client = http_client
        self._owns_http_client = http_client is None

        # Session will be created on first use
        self._session: Optional[ClientSession] = None

    async def _ensure_session(self):
        """Create aiohttp session if not exists or closed."""
        if self._session is None or self._session.closed:
            import ssl

            # Create SSL context
            ssl_context = None
            if self.ssl_options:
                ssl_context = ssl.create_default_context()
                if not self.ssl_options.tls_verify:
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                elif not self.ssl_options.tls_verify_hostname:
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_REQUIRED
                if self.ssl_options.tls_trusted_ca_file:
                    ssl_context.load_verify_locations(
                        self.ssl_options.tls_trusted_ca_file
                    )
                if (
                    self.ssl_options.tls_client_cert_file
                    and self.ssl_options.tls_client_cert_key_file
                ):
                    ssl_context.load_cert_chain(
                        self.ssl_options.tls_client_cert_file,
                        self.ssl_options.tls_client_cert_key_file,
                        self.ssl_options.tls_client_cert_key_password,
                    )

            connector = TCPConnector(
                limit=10,
                limit_per_host=10,
                ssl=ssl_context,
            )

            self._session = ClientSession(
                connector=connector,
                timeout=ClientTimeout(total=300),
            )

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers from the auth provider."""
        headers: Dict[str, str] = {}
        self.auth_provider.add_headers(headers)
        return headers

    def _get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff."""
        delay = min(
            self._retry_delay_max, self._retry_delay_min * (2**attempt)
        )
        return delay

    def _should_retry(
        self, status_code: int, attempt: int, command_type: CommandType
    ) -> bool:
        """Determine if a request should be retried based on status code and command type."""
        if attempt >= self._max_retries:
            return False

        # Never retry 400 (bad request) or auth errors
        if status_code in [400, 401, 403, 501]:
            return False

        # For ExecuteStatement, only retry specific codes
        if command_type == CommandType.EXECUTE_STATEMENT:
            return status_code in [429, 503, *self.force_dangerous_codes]

        # For other commands, retry on server errors and rate limiting
        return status_code in [429, 500, 502, 503, 504]

    def _get_command_type_from_path(self, path: str, method: str) -> CommandType:
        """
        Determine the command type based on the API path and method.

        This helps the retry policy make appropriate decisions for different
        types of SEA operations.
        """
        path = path.lower()
        method = method.upper()

        if "/statements" in path:
            if method == "POST" and path.endswith("/statements"):
                return CommandType.EXECUTE_STATEMENT
            elif "/cancel" in path:
                return CommandType.OTHER
            elif method == "DELETE":
                return CommandType.CLOSE_OPERATION
            elif method == "GET":
                return CommandType.GET_OPERATION_STATUS
        elif "/sessions" in path:
            if method == "DELETE":
                return CommandType.CLOSE_SESSION

        return CommandType.OTHER

    async def _make_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make an async HTTP request to the SEA endpoint.

        Args:
            method: HTTP method (GET, POST, DELETE)
            path: API endpoint path
            data: Request payload data

        Returns:
            Dict[str, Any]: Response data parsed from JSON

        Raises:
            RequestError: If the request fails after retries
        """
        await self._ensure_session()

        # Prepare headers
        headers = {**self.headers, **self._get_auth_headers()}

        # Prepare request body
        body = json.dumps(data) if data else None
        if body:
            headers["Content-Length"] = str(len(body.encode("utf-8")))

        # Determine command type for retry logic
        command_type = self._get_command_type_from_path(path, method)

        url = f"{self.base_url}{path}"
        logger.debug(f"Making async {method} request to {path}")

        last_error: Optional[Exception] = None
        last_status: Optional[int] = None

        for attempt in range(self._max_retries + 1):
            try:
                async with self._session.request(
                    method.upper(),
                    url,
                    data=body,
                    headers=headers,
                ) as response:
                    last_status = response.status

                    # Handle successful responses
                    if 200 <= response.status < 300:
                        response_data = await response.read()
                        if response_data:
                            return json.loads(response_data.decode())
                        else:
                            return {}

                    # Check if we should retry
                    if self._should_retry(response.status, attempt, command_type):
                        delay = self._get_retry_delay(attempt)
                        logger.debug(
                            f"Request failed with status {response.status}, "
                            f"retrying in {delay:.2f} seconds (attempt {attempt + 1}/{self._max_retries + 1})"
                        )
                        await asyncio.sleep(delay)
                        continue

                    # Non-retryable error
                    error_data = await response.read()
                    error_message = (
                        f"SEA HTTP request failed with status {response.status}: "
                        f"{error_data.decode() if error_data else 'No error message'}"
                    )
                    raise RequestError(error_message, context={"http-code": response.status})

            except aiohttp.ClientError as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = self._get_retry_delay(attempt)
                    logger.debug(
                        f"Request failed with error {e}, "
                        f"retrying in {delay:.2f} seconds"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise RequestError(f"HTTP request failed: {e}")

            except RequestError:
                raise

            except Exception as e:
                logger.error(f"Unexpected error during HTTP request: {e}")
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
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
