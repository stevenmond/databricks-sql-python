"""
Async Thrift backend implementation.

This module provides an async Thrift backend client for Databricks SQL
with true async/await support using aiohttp.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import List, Optional, Union, Any, Dict, TYPE_CHECKING, Tuple

try:
    import pyarrow
except ImportError:
    pyarrow = None

import thrift.protocol.TBinaryProtocol

from databricks.sql.auth.authenticators import AuthProvider
from databricks.sql.auth.async_thrift_http_client import AsyncTHttpClient
from databricks.sql.thrift_api.TCLIService import TCLIService, ttypes
from databricks.sql.thrift_api.TCLIService.TCLIService import (
    Client as TCLIServiceClient,
)
from databricks.sql.backend.async_databricks_client import AsyncDatabricksClient
from databricks.sql.backend.types import (
    CommandState,
    SessionId,
    CommandId,
    ExecuteResponse,
    BackendType,
)
from databricks.sql.backend.utils import guid_to_hex_id
from databricks.sql.types import SSLOptions
from databricks.sql.exc import (
    DatabaseError,
    OperationalError,
    ServerOperationError,
    RequestError,
    Error,
    InvalidServerResponseError,
)
from databricks.sql.utils import (
    _bound,
    RequestErrorInfo,
    NoRetryReason,
    convert_arrow_based_set_to_arrow_table,
    convert_decimals_in_arrow_table,
    convert_column_based_set_to_arrow_table,
)

if TYPE_CHECKING:
    from databricks.sql.async_client import AsyncCursor
    from databricks.sql.backend.async_result_set import AsyncResultSet

logger = logging.getLogger(__name__)

THRIFT_ERROR_MESSAGE_HEADER = "x-thriftserver-error-message"
DATABRICKS_ERROR_OR_REDIRECT_HEADER = "x-databricks-error-or-redirect-message"
DATABRICKS_REASON_HEADER = "x-databricks-reason-phrase"
TIMESTAMP_AS_STRING_CONFIG = "spark.thriftserver.arrowBasedRowSet.timestampAsString"
DEFAULT_SOCKET_TIMEOUT = float(900)

# Retry policy defaults
_retry_policy = {
    "_retry_delay_min": (float, 1, 0.1, 60),
    "_retry_delay_max": (float, 60, 5, 3600),
    "_retry_stop_after_attempts_count": (int, 30, 1, 60),
    "_retry_stop_after_attempts_duration": (float, 900, 1, 86400),
    "_retry_delay_default": (float, 5, 1, 60),
}


class AsyncThriftDatabricksClient(AsyncDatabricksClient):
    """
    Async Thrift backend client for Databricks SQL.

    This client provides true async/await support for Thrift-based connections
    to Databricks clusters and SQL warehouses.
    """

    CLOSED_OP_STATE = CommandState.CLOSED
    ERROR_OP_STATE = CommandState.FAILED

    def __init__(
        self,
        server_hostname: str,
        port: int,
        http_path: str,
        http_headers: List[Tuple[str, str]],
        auth_provider: AuthProvider,
        ssl_options: SSLOptions,
        **kwargs,
    ):
        """
        Initialize async Thrift backend client.

        Args:
            server_hostname: Databricks server hostname
            port: Server port (usually 443)
            http_path: HTTP path to SQL endpoint
            http_headers: List of HTTP headers as (name, value) tuples
            auth_provider: Authentication provider
            ssl_options: SSL configuration options
            **kwargs: Additional options
        """
        logger.debug(
            "AsyncThriftDatabricksClient.__init__(server_hostname=%s, port=%s, http_path=%s)",
            server_hostname,
            port,
            http_path,
        )

        port = port or 443
        if kwargs.get("_connection_uri"):
            uri = kwargs.get("_connection_uri")
        elif server_hostname and http_path:
            uri = f"https://{server_hostname.rstrip('/')}:{port}/{http_path.lstrip('/')}"
        else:
            raise ValueError("No valid connection settings.")

        self._host = server_hostname
        self._port = port
        self._http_path = http_path
        self._initialize_retry_args(kwargs)

        self._use_arrow_native_complex_types = kwargs.get(
            "_use_arrow_native_complex_types", True
        )
        self._use_arrow_native_decimals = kwargs.get("_use_arrow_native_decimals", True)
        self._use_arrow_native_timestamps = kwargs.get(
            "_use_arrow_native_timestamps", True
        )

        # Cloud fetch settings
        self._max_download_threads = kwargs.get("max_download_threads", 10)

        self._ssl_options = ssl_options
        self._auth_provider = auth_provider

        # Create async transport
        additional_transport_args = {}
        proxy_auth_method = kwargs.get("_proxy_auth_method")
        if proxy_auth_method:
            additional_transport_args["_proxy_auth_method"] = proxy_auth_method

        timeout = kwargs.get("_socket_timeout", DEFAULT_SOCKET_TIMEOUT)

        self._transport = AsyncTHttpClient(
            auth_provider=self._auth_provider,
            uri_or_host=uri,
            ssl_options=self._ssl_options,
            timeout=timeout,
            **additional_transport_args,
        )

        # Set custom headers
        self._transport.setCustomHeaders(dict(http_headers))

        # Create Thrift protocol and client
        self._protocol = thrift.protocol.TBinaryProtocol.TBinaryProtocol(
            self._transport
        )
        self._client = TCLIService.Client(self._protocol)

        self._session_id_hex: Optional[str] = None

    @property
    def max_download_threads(self) -> int:
        """Get maximum number of download threads for cloud fetch."""
        return self._max_download_threads

    @property
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        return BackendType.THRIFT

    def _initialize_retry_args(self, kwargs: dict) -> None:
        """Initialize retry policy parameters."""
        for key, (type_, default, min_val, max_val) in _retry_policy.items():
            given_or_default = type_(kwargs.get(key, default))
            bound = _bound(min_val, max_val, given_or_default)
            setattr(self, key, bound)
            if bound != given_or_default:
                logger.warning(
                    "Override out of policy retry parameter: %s given %s, restricted to %s",
                    key,
                    given_or_default,
                    bound,
                )

        if (
            self._retry_stop_after_attempts_count > 1
            and self._retry_delay_min > self._retry_delay_max
        ):
            raise ValueError(
                f"Invalid configuration: retry delay min({self._retry_delay_min}) > max({self._retry_delay_max})"
            )

    @staticmethod
    def _check_response_for_error(response, host_url: Optional[str] = None) -> None:
        """Check Thrift response for errors."""
        if response.status and response.status.statusCode in [
            ttypes.TStatusCode.ERROR_STATUS,
            ttypes.TStatusCode.INVALID_HANDLE_STATUS,
        ]:
            raise DatabaseError(response.status.errorMessage, host_url=host_url)

    @staticmethod
    def _extract_error_message_from_headers(headers: Dict[str, str]) -> str:
        """Extract error message from HTTP response headers."""
        err_msg = ""
        if THRIFT_ERROR_MESSAGE_HEADER in headers:
            err_msg = headers[THRIFT_ERROR_MESSAGE_HEADER]
        if DATABRICKS_ERROR_OR_REDIRECT_HEADER in headers:
            if err_msg:
                err_msg = f"Thriftserver error: {err_msg}, Databricks error: {headers[DATABRICKS_ERROR_OR_REDIRECT_HEADER]}"
            else:
                err_msg = headers[DATABRICKS_ERROR_OR_REDIRECT_HEADER]
            if DATABRICKS_REASON_HEADER in headers:
                err_msg += ": " + headers[DATABRICKS_REASON_HEADER]
        if not err_msg and DATABRICKS_REASON_HEADER in headers:
            err_msg = ": " + headers[DATABRICKS_REASON_HEADER]
        return err_msg

    async def _make_request_async(
        self, method, request, retryable: bool = True
    ) -> Any:
        """
        Make an async Thrift request with retry support.

        Args:
            method: The Thrift client method to call
            request: The Thrift request object
            retryable: Whether to retry on transient errors

        Returns:
            The Thrift response object
        """
        t0 = time.time()

        def get_elapsed() -> float:
            return time.time() - t0

        def bound_retry_delay(attempt: int, proposed_delay: float) -> float:
            delay = max(proposed_delay, self._retry_delay_min * math.pow(1.5, attempt - 1))
            delay = min(delay, self._retry_delay_max)
            return delay

        max_attempts = self._retry_stop_after_attempts_count if retryable else 1

        for attempt in range(1, max_attempts + 1):
            try:
                method_name = getattr(method, "__name__", str(method))
                logger.debug("Async Thrift request: %s(<REDACTED>)", method_name)

                # Write request to transport buffer
                method.send(request)

                # Flush asynchronously
                await self._transport.flush()

                # Read response
                response = method.recv()

                logger.debug(
                    "Async Thrift response: %s(<REDACTED>)",
                    type(response).__name__,
                )

                self._check_response_for_error(response, self._host)
                return response

            except Exception as err:
                elapsed = get_elapsed()
                logger.error("Async Thrift request error: %s", err)

                # Extract retry delay if available
                retry_delay = None
                http_code = getattr(self._transport, "code", None)
                headers = getattr(self._transport, "headers", {})

                if http_code in [429, 503]:
                    retry_after = headers.get("Retry-After", "1")
                    retry_delay = bound_retry_delay(attempt, int(retry_after))

                # Check if we can retry
                error_message = self._extract_error_message_from_headers(headers)

                error_info = RequestErrorInfo(
                    error=err,
                    error_message=error_message,
                    retry_delay=retry_delay,
                    http_code=http_code,
                    method=getattr(method, "__name__", str(method)),
                    request=request,
                )

                # Determine if we should retry
                max_duration = self._retry_stop_after_attempts_duration
                no_retry_reason = None

                if retry_delay is not None and elapsed + retry_delay > max_duration:
                    no_retry_reason = NoRetryReason.OUT_OF_TIME
                elif retry_delay is not None and attempt >= max_attempts:
                    no_retry_reason = NoRetryReason.OUT_OF_ATTEMPTS
                elif retry_delay is None:
                    no_retry_reason = NoRetryReason.NOT_RETRYABLE

                if no_retry_reason is not None:
                    user_friendly_message = error_info.user_friendly_error_message(
                        no_retry_reason, attempt, elapsed
                    )
                    raise RequestError(
                        user_friendly_message,
                        error_info.full_info_logging_context(
                            no_retry_reason, attempt, max_attempts, elapsed, max_duration
                        ),
                        self._host,
                        err,
                    )

                # Sleep before retry using asyncio.sleep (non-blocking)
                logger.info(
                    "Retrying async request after error in %s seconds",
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)

        # Should not reach here, but raise if we do
        raise RuntimeError("Unexpected: exhausted retries without error")

    def _check_protocol_version(self, response) -> None:
        """Check that server protocol version is supported."""
        protocol_version = response.serverProtocolVersion
        if protocol_version < ttypes.TProtocolVersion.SPARK_CLI_SERVICE_PROTOCOL_V2:
            raise OperationalError(
                f"Error: expected server to use a protocol version >= "
                f"SPARK_CLI_SERVICE_PROTOCOL_V2, instead got: {protocol_version}",
                host_url=self._host,
            )

    def _check_initial_namespace(self, catalog: Optional[str], schema: Optional[str], response) -> None:
        """Check that initial namespace was set correctly."""
        if not (catalog or schema):
            return

        if response.serverProtocolVersion < ttypes.TProtocolVersion.SPARK_CLI_SERVICE_PROTOCOL_V4:
            raise InvalidServerResponseError(
                "Setting initial namespace not supported by the DBR version. "
                "Please use a Databricks SQL endpoint or a cluster with DBR >= 9.0.",
                host_url=self._host,
            )

        if catalog and not response.canUseMultipleCatalogs:
            raise InvalidServerResponseError(
                f"Unexpected response from server: Trying to set initial catalog to {catalog}, "
                f"but server does not support multiple catalogs.",
                host_url=self._host,
            )

    def _check_session_configuration(self, session_configuration: Optional[Dict[str, Any]]) -> None:
        """Validate session configuration."""
        if not session_configuration:
            return

        if session_configuration.get(TIMESTAMP_AS_STRING_CONFIG, "false").lower() != "false":
            raise Error(
                f"Invalid session configuration: {TIMESTAMP_AS_STRING_CONFIG} cannot be changed "
                f"while using the Databricks SQL connector, it must be false not "
                f"{session_configuration[TIMESTAMP_AS_STRING_CONFIG]}",
                host_url=self._host,
            )

    async def open_session(
        self,
        session_configuration: Optional[Dict[str, Any]] = None,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> SessionId:
        """
        Open a new session asynchronously.

        Args:
            session_configuration: Session configuration parameters
            catalog: Initial catalog name
            schema: Initial schema name

        Returns:
            SessionId for the opened session
        """
        try:
            await self._transport.open()

            session_config = {k: str(v) for k, v in (session_configuration or {}).items()}
            self._check_session_configuration(session_config)
            session_config[TIMESTAMP_AS_STRING_CONFIG] = "false"

            initial_namespace = None
            if catalog or schema:
                initial_namespace = ttypes.TNamespace(
                    catalogName=catalog, schemaName=schema
                )

            req = ttypes.TOpenSessionReq(
                client_protocol_i64=ttypes.TProtocolVersion.SPARK_CLI_SERVICE_PROTOCOL_V7,
                client_protocol=None,
                initialNamespace=initial_namespace,
                canUseMultipleCatalogs=True,
                configuration=session_config,
            )

            response = await self._make_request_async(self._client.OpenSession, req)

            self._check_initial_namespace(catalog, schema, response)
            self._check_protocol_version(response)

            properties = {}
            if response.serverProtocolVersion:
                properties["serverProtocolVersion"] = response.serverProtocolVersion

            session_id = SessionId.from_thrift_handle(response.sessionHandle, properties)
            self._session_id_hex = session_id.hex_guid

            return session_id

        except Exception:
            await self._transport.close()
            raise

    async def close_session(self, session_id: SessionId) -> None:
        """
        Close a session asynchronously.

        Args:
            session_id: Session ID to close
        """
        thrift_handle = session_id.to_thrift_handle()
        if not thrift_handle:
            raise ValueError("Not a valid Thrift session ID")

        try:
            req = ttypes.TCloseSessionReq(sessionHandle=thrift_handle)
            await self._make_request_async(self._client.CloseSession, req)
        finally:
            await self._transport.close()

    def _check_command_not_in_error_or_closed_state(
        self, op_handle, get_operations_resp
    ) -> None:
        """Check that command is not in error or closed state."""
        if get_operations_resp.operationState == ttypes.TOperationState.ERROR_STATE:
            if get_operations_resp.displayMessage:
                raise ServerOperationError(
                    get_operations_resp.displayMessage,
                    {
                        "operation-id": op_handle and guid_to_hex_id(op_handle.operationId.guid),
                        "diagnostic-info": get_operations_resp.diagnosticInfo,
                    },
                    host_url=self._host,
                )
            else:
                raise ServerOperationError(
                    get_operations_resp.errorMessage,
                    {
                        "operation-id": op_handle and guid_to_hex_id(op_handle.operationId.guid),
                        "diagnostic-info": None,
                    },
                    host_url=self._host,
                )
        elif get_operations_resp.operationState == ttypes.TOperationState.CLOSED_STATE:
            raise DatabaseError(
                f"Command {op_handle and guid_to_hex_id(op_handle.operationId.guid)} "
                f"unexpectedly closed server side",
                {"operation-id": op_handle and guid_to_hex_id(op_handle.operationId.guid)},
                host_url=self._host,
            )

    async def _poll_for_status_async(self, op_handle) -> Any:
        """Poll for operation status asynchronously."""
        req = ttypes.TGetOperationStatusReq(
            operationHandle=op_handle,
            getProgressUpdate=False,
        )
        return await self._make_request_async(self._client.GetOperationStatus, req)

    async def _wait_until_command_done_async(
        self, op_handle, initial_operation_status_resp
    ) -> int:
        """
        Wait for command to complete asynchronously.

        Uses asyncio.sleep() instead of time.sleep() for non-blocking polling.
        """
        if initial_operation_status_resp:
            self._check_command_not_in_error_or_closed_state(
                op_handle, initial_operation_status_resp
            )

        operation_state = (
            initial_operation_status_resp
            and initial_operation_status_resp.operationState
        )

        poll_interval = 0.1  # Start with 100ms
        max_poll_interval = 2.0  # Max 2 seconds

        while not operation_state or operation_state in [
            ttypes.TOperationState.RUNNING_STATE,
            ttypes.TOperationState.PENDING_STATE,
        ]:
            # Non-blocking sleep
            await asyncio.sleep(poll_interval)

            # Exponential backoff for polling
            poll_interval = min(poll_interval * 1.5, max_poll_interval)

            poll_resp = await self._poll_for_status_async(op_handle)
            operation_state = poll_resp.operationState
            self._check_command_not_in_error_or_closed_state(op_handle, poll_resp)

        return operation_state

    async def get_query_state(self, command_id: CommandId) -> CommandState:
        """
        Get the current state of a query asynchronously.

        Args:
            command_id: Command ID to check

        Returns:
            CommandState indicating current state
        """
        thrift_handle = command_id.to_thrift_handle()
        if not thrift_handle:
            raise ValueError("Not a valid Thrift command ID")

        poll_resp = await self._poll_for_status_async(thrift_handle)
        operation_state = poll_resp.operationState
        self._check_command_not_in_error_or_closed_state(thrift_handle, poll_resp)

        state = CommandState.from_thrift_state(operation_state)
        if state is None:
            raise ValueError(f"Unknown command state: {operation_state}")

        return state

    @staticmethod
    def _hive_schema_to_description(t_table_schema, schema_bytes=None, host_url=None):
        """Convert Hive schema to description tuples."""
        from databricks.sql.backend.thrift_backend import ThriftDatabricksClient

        return ThriftDatabricksClient._hive_schema_to_description(
            t_table_schema, schema_bytes, host_url
        )

    @staticmethod
    def _hive_schema_to_arrow_schema(t_table_schema, host_url=None):
        """Convert Hive schema to Arrow schema."""
        from databricks.sql.backend.thrift_backend import ThriftDatabricksClient

        return ThriftDatabricksClient._hive_schema_to_arrow_schema(
            t_table_schema, host_url
        )

    async def _get_metadata_resp_async(self, op_handle):
        """Get result set metadata asynchronously."""
        req = ttypes.TGetResultSetMetadataReq(operationHandle=op_handle)
        return await self._make_request_async(self._client.GetResultSetMetadata, req)

    @staticmethod
    def _check_direct_results_for_error(t_spark_direct_results, host_url=None):
        """Check direct results for errors."""
        from databricks.sql.backend.thrift_backend import ThriftDatabricksClient

        ThriftDatabricksClient._check_direct_results_for_error(
            t_spark_direct_results, host_url
        )

    async def _results_message_to_execute_response_async(
        self, resp, operation_state
    ) -> Tuple[ExecuteResponse, bool]:
        """Convert results message to ExecuteResponse."""
        if resp.directResults and resp.directResults.resultSetMetadata:
            t_result_set_metadata_resp = resp.directResults.resultSetMetadata
        else:
            t_result_set_metadata_resp = await self._get_metadata_resp_async(
                resp.operationHandle
            )

        if t_result_set_metadata_resp.resultFormat not in [
            ttypes.TSparkRowSetType.ARROW_BASED_SET,
            ttypes.TSparkRowSetType.COLUMN_BASED_SET,
            ttypes.TSparkRowSetType.URL_BASED_SET,
        ]:
            raise OperationalError(
                f"Expected results to be in Arrow or column based format, "
                f"instead they are: {ttypes.TSparkRowSetType._VALUES_TO_NAMES[t_result_set_metadata_resp.resultFormat]}",
                host_url=self._host,
            )

        direct_results = resp.directResults
        has_been_closed_server_side = direct_results and direct_results.closeOperation

        has_more_rows = (
            (not direct_results)
            or (not direct_results.resultSet)
            or direct_results.resultSet.hasMoreRows
        )

        if pyarrow:
            schema_bytes = (
                t_result_set_metadata_resp.arrowSchema
                or self._hive_schema_to_arrow_schema(
                    t_result_set_metadata_resp.schema, self._host
                )
                .serialize()
                .to_pybytes()
            )
        else:
            schema_bytes = None

        description = self._hive_schema_to_description(
            t_result_set_metadata_resp.schema,
            schema_bytes,
            self._host,
        )

        lz4_compressed = t_result_set_metadata_resp.lz4Compressed
        command_id = CommandId.from_thrift_handle(resp.operationHandle)

        status = CommandState.from_thrift_state(operation_state)
        if status is None:
            raise ValueError(f"Unknown command state: {operation_state}")

        execute_response = ExecuteResponse(
            command_id=command_id,
            status=status,
            description=description,
            has_been_closed_server_side=has_been_closed_server_side,
            lz4_compressed=lz4_compressed,
            is_staging_operation=t_result_set_metadata_resp.isStagingOperation,
            arrow_schema_bytes=schema_bytes,
            result_format=t_result_set_metadata_resp.resultFormat,
        )

        return execute_response, has_more_rows

    async def execute_command(
        self,
        operation: str,
        session_id: SessionId,
        max_rows: int,
        max_bytes: int,
        lz4_compression: bool,
        cursor: "AsyncCursor",
        use_cloud_fetch: bool = True,
        parameters: List = None,
        async_op: bool = False,
        enforce_embedded_schema_correctness: bool = False,
    ) -> Union["AsyncResultSet", None]:
        """
        Execute a SQL command asynchronously.

        Args:
            operation: SQL statement to execute
            session_id: Session ID
            max_rows: Maximum rows to fetch
            max_bytes: Maximum bytes to fetch
            lz4_compression: Whether to use LZ4 compression
            cursor: Parent cursor
            use_cloud_fetch: Whether to use cloud fetch
            parameters: Query parameters
            async_op: Whether to run asynchronously (non-blocking)
            enforce_embedded_schema_correctness: Schema enforcement

        Returns:
            AsyncResultSet if sync operation, None if async
        """
        thrift_handle = session_id.to_thrift_handle()
        if not thrift_handle:
            raise ValueError("Not a valid Thrift session ID")

        logger.debug(
            "AsyncThriftDatabricksClient.execute_command(operation=%s, session_handle=%s)",
            operation,
            thrift_handle,
        )

        if parameters is None:
            parameters = []

        spark_arrow_types = ttypes.TSparkArrowTypes(
            timestampAsArrow=self._use_arrow_native_timestamps,
            decimalAsArrow=self._use_arrow_native_decimals,
            complexTypesAsArrow=self._use_arrow_native_complex_types,
            intervalTypesAsArrow=False,
        )

        req = ttypes.TExecuteStatementReq(
            sessionHandle=thrift_handle,
            statement=operation,
            runAsync=True,
            getDirectResults=None
            if async_op
            else ttypes.TSparkGetDirectResults(
                maxRows=max_rows,
                maxBytes=max_bytes,
            ),
            canReadArrowResult=True if pyarrow else False,
            canDecompressLZ4Result=lz4_compression,
            canDownloadResult=use_cloud_fetch,
            confOverlay={
                "spark.thriftserver.arrowBasedRowSet.timestampAsString": "false"
            },
            useArrowNativeTypes=spark_arrow_types,
            parameters=parameters,
            enforceEmbeddedSchemaCorrectness=enforce_embedded_schema_correctness,
        )

        resp = await self._make_request_async(self._client.ExecuteStatement, req)

        command_id = CommandId.from_thrift_handle(resp.operationHandle)
        if command_id is None:
            raise ValueError(f"Invalid Thrift handle: {resp.operationHandle}")

        cursor.active_command_id = command_id
        self._check_direct_results_for_error(resp.directResults, self._host)

        if async_op:
            # For async operation, just return None
            return None

        # Wait for completion
        final_operation_state = await self._wait_until_command_done_async(
            resp.operationHandle,
            resp.directResults and resp.directResults.operationStatus,
        )

        execute_response, has_more_rows = await self._results_message_to_execute_response_async(
            resp, final_operation_state
        )

        t_row_set = None
        if resp.directResults and resp.directResults.resultSet:
            t_row_set = resp.directResults.resultSet.results

        # Import here to avoid circular imports
        from databricks.sql.backend.async_thrift_result_set import AsyncThriftResultSet

        return AsyncThriftResultSet(
            connection=cursor.connection,
            execute_response=execute_response,
            thrift_client=self,
            buffer_size_bytes=max_bytes,
            arraysize=max_rows,
            use_cloud_fetch=use_cloud_fetch,
            t_row_set=t_row_set,
            max_download_threads=self.max_download_threads,
            ssl_options=self._ssl_options,
            has_more_rows=has_more_rows,
        )

    async def cancel_command(self, command_id: CommandId) -> None:
        """
        Cancel a running command asynchronously.

        Args:
            command_id: Command ID to cancel
        """
        thrift_handle = command_id.to_thrift_handle()
        if not thrift_handle:
            raise ValueError("Not a valid Thrift command ID")

        logger.debug("Cancelling command %s", command_id.to_hex_guid())
        req = ttypes.TCancelOperationReq(thrift_handle)
        await self._make_request_async(self._client.CancelOperation, req)

    async def close_command(self, command_id: CommandId) -> None:
        """
        Close a command asynchronously.

        Args:
            command_id: Command ID to close
        """
        thrift_handle = command_id.to_thrift_handle()
        if not thrift_handle:
            raise ValueError("Not a valid Thrift command ID")

        logger.debug("AsyncThriftDatabricksClient.close_command(command_id=%s)", command_id)
        req = ttypes.TCloseOperationReq(operationHandle=thrift_handle)
        await self._make_request_async(self._client.CloseOperation, req)

    async def get_catalogs(
        self,
        session_id: SessionId,
        max_rows: int,
        max_bytes: int,
        cursor: "AsyncCursor",
    ) -> "AsyncResultSet":
        """Get available catalogs asynchronously."""
        thrift_handle = session_id.to_thrift_handle()
        if not thrift_handle:
            raise ValueError("Not a valid Thrift session ID")

        req = ttypes.TGetCatalogsReq(
            sessionHandle=thrift_handle,
            getDirectResults=ttypes.TSparkGetDirectResults(
                maxRows=max_rows, maxBytes=max_bytes
            ),
        )

        resp = await self._make_request_async(self._client.GetCatalogs, req)
        return await self._handle_execute_response_async(resp, cursor, max_rows, max_bytes)

    async def get_schemas(
        self,
        session_id: SessionId,
        max_rows: int,
        max_bytes: int,
        cursor: "AsyncCursor",
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
    ) -> "AsyncResultSet":
        """Get available schemas asynchronously."""
        thrift_handle = session_id.to_thrift_handle()
        if not thrift_handle:
            raise ValueError("Not a valid Thrift session ID")

        req = ttypes.TGetSchemasReq(
            sessionHandle=thrift_handle,
            getDirectResults=ttypes.TSparkGetDirectResults(
                maxRows=max_rows, maxBytes=max_bytes
            ),
            catalogName=catalog_name,
            schemaName=schema_name,
        )

        resp = await self._make_request_async(self._client.GetSchemas, req)
        return await self._handle_execute_response_async(resp, cursor, max_rows, max_bytes)

    async def get_tables(
        self,
        session_id: SessionId,
        max_rows: int,
        max_bytes: int,
        cursor: "AsyncCursor",
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        table_types: Optional[List[str]] = None,
    ) -> "AsyncResultSet":
        """Get available tables asynchronously."""
        thrift_handle = session_id.to_thrift_handle()
        if not thrift_handle:
            raise ValueError("Not a valid Thrift session ID")

        req = ttypes.TGetTablesReq(
            sessionHandle=thrift_handle,
            getDirectResults=ttypes.TSparkGetDirectResults(
                maxRows=max_rows, maxBytes=max_bytes
            ),
            catalogName=catalog_name,
            schemaName=schema_name,
            tableName=table_name,
            tableTypes=table_types,
        )

        resp = await self._make_request_async(self._client.GetTables, req)
        return await self._handle_execute_response_async(resp, cursor, max_rows, max_bytes)

    async def get_columns(
        self,
        session_id: SessionId,
        max_rows: int,
        max_bytes: int,
        cursor: "AsyncCursor",
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        column_name: Optional[str] = None,
    ) -> "AsyncResultSet":
        """Get column information asynchronously."""
        thrift_handle = session_id.to_thrift_handle()
        if not thrift_handle:
            raise ValueError("Not a valid Thrift session ID")

        req = ttypes.TGetColumnsReq(
            sessionHandle=thrift_handle,
            getDirectResults=ttypes.TSparkGetDirectResults(
                maxRows=max_rows, maxBytes=max_bytes
            ),
            catalogName=catalog_name,
            schemaName=schema_name,
            tableName=table_name,
            columnName=column_name,
        )

        resp = await self._make_request_async(self._client.GetColumns, req)
        return await self._handle_execute_response_async(resp, cursor, max_rows, max_bytes)

    async def _handle_execute_response_async(
        self, resp, cursor: "AsyncCursor", max_rows: int, max_bytes: int
    ) -> "AsyncResultSet":
        """Handle execute response and create result set."""
        command_id = CommandId.from_thrift_handle(resp.operationHandle)
        if command_id is None:
            raise ValueError(f"Invalid Thrift handle: {resp.operationHandle}")

        cursor.active_command_id = command_id
        self._check_direct_results_for_error(resp.directResults, self._host)

        final_operation_state = await self._wait_until_command_done_async(
            resp.operationHandle,
            resp.directResults and resp.directResults.operationStatus,
        )

        execute_response, has_more_rows = await self._results_message_to_execute_response_async(
            resp, final_operation_state
        )

        t_row_set = None
        if resp.directResults and resp.directResults.resultSet:
            t_row_set = resp.directResults.resultSet.results

        from databricks.sql.backend.async_thrift_result_set import AsyncThriftResultSet

        return AsyncThriftResultSet(
            connection=cursor.connection,
            execute_response=execute_response,
            thrift_client=self,
            buffer_size_bytes=max_bytes,
            arraysize=max_rows,
            use_cloud_fetch=cursor.connection.use_cloud_fetch,
            t_row_set=t_row_set,
            max_download_threads=self.max_download_threads,
            ssl_options=self._ssl_options,
            has_more_rows=has_more_rows,
        )
