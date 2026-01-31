"""
Async client for Databricks SQL connector.

This module provides AsyncConnection and AsyncCursor classes that offer
true async/await support for Databricks SQL operations.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Tuple, List, Optional, Any, Union, TYPE_CHECKING, Sequence

try:
    import pyarrow
except ImportError:
    pyarrow = None

from databricks.sql import __version__
from databricks.sql import USER_AGENT_NAME
from databricks.sql.exc import (
    InterfaceError,
    ProgrammingError,
    OperationalError,
    Error,
)
from databricks.sql.thrift_api.TCLIService.ttypes import TSparkParameter
from databricks.sql.backend.async_databricks_client import AsyncDatabricksClient
from databricks.sql.backend.async_result_set import AsyncResultSet
from databricks.sql.backend.types import CommandId, CommandState, SessionId
from databricks.sql.types import Row, SSLOptions
from databricks.sql.auth.auth import get_python_sql_connector_auth_provider
from databricks.sql.common.unified_http_client import UnifiedHttpClient
from databricks.sql.parameters.native import (
    DbsqlParameterBase,
    TDbsqlParameter,
    TParameterDict,
    TParameterSequence,
    TParameterCollection,
    ParameterStructure,
    dbsql_parameter_from_primitive,
    ParameterApproach,
)
from databricks.sql.utils import (
    ParamEscaper,
    inject_parameters,
    transform_paramstyle,
    build_client_context,
)

logger = logging.getLogger(__name__)

DEFAULT_RESULT_BUFFER_SIZE_BYTES = 104857600
DEFAULT_ARRAY_SIZE = 100000
NO_NATIVE_PARAMS: List = []


class AsyncSession:
    """
    Async session management for Databricks SQL.

    This class handles all session-related behavior and communication with the backend
    using async/await patterns.
    """

    def __init__(
        self,
        server_hostname: str,
        http_path: str,
        http_client: UnifiedHttpClient,
        http_headers: Optional[List[Tuple[str, str]]] = None,
        session_configuration: Optional[Dict[str, Any]] = None,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        _use_arrow_native_complex_types: Optional[bool] = True,
        **kwargs,
    ) -> None:
        self.is_open = False
        self.host = server_hostname
        self.port = kwargs.get("_port", 443)

        self.session_configuration = session_configuration
        self.catalog = catalog
        self.schema = schema
        self.http_path = http_path
        self.http_client = http_client
        self._autocommit = True
        self._session_id: Optional[SessionId] = None

        user_agent_entry = kwargs.get("user_agent_entry") or kwargs.get(
            "_user_agent_entry"
        )
        if user_agent_entry:
            self.useragent_header = "{}/{} ({})".format(
                USER_AGENT_NAME, __version__, user_agent_entry
            )
        else:
            self.useragent_header = "{}/{}".format(USER_AGENT_NAME, __version__)

        base_headers = [("User-Agent", self.useragent_header)]
        self.all_headers = (http_headers or []) + base_headers

        self.ssl_options = SSLOptions(
            tls_verify=not kwargs.get("_tls_no_verify", False),
            tls_verify_hostname=kwargs.get("_tls_verify_hostname", True),
            tls_trusted_ca_file=kwargs.get("_tls_trusted_ca_file"),
            tls_client_cert_file=kwargs.get("_tls_client_cert_file"),
            tls_client_cert_key_file=kwargs.get("_tls_client_cert_key_file"),
            tls_client_cert_key_password=kwargs.get("_tls_client_cert_key_password"),
        )

        self._server_hostname = server_hostname
        self._use_arrow_native_complex_types = _use_arrow_native_complex_types
        self._kwargs = kwargs
        self.backend: Optional[AsyncDatabricksClient] = None
        self.auth_provider = None

    async def open(self) -> None:
        """Open the async session."""
        # Create auth provider (sync operation - no network calls typically)
        from databricks.sql.auth.authenticators import AuthProvider

        # Create a simple auth provider that adds headers
        self.auth_provider = get_python_sql_connector_auth_provider(
            self._server_hostname, http_client=self.http_client, **self._kwargs
        )

        # Determine which backend to use
        # SEA is the default for async, but Thrift is also supported
        use_sea = self._kwargs.get("use_sea", True)  # Default to SEA for async
        use_thrift = self._kwargs.get("use_thrift", False)

        if use_thrift or not use_sea:
            # Use async Thrift backend
            logger.info("Using async Thrift backend")
            from databricks.sql.backend.async_thrift_backend import (
                AsyncThriftDatabricksClient,
            )

            self.backend = AsyncThriftDatabricksClient(
                server_hostname=self._server_hostname,
                port=self.port,
                http_path=self.http_path,
                http_headers=self.all_headers,
                auth_provider=self.auth_provider,
                ssl_options=self.ssl_options,
                _use_arrow_native_complex_types=self._use_arrow_native_complex_types,
                **self._kwargs,
            )
        else:
            # Use async SEA backend (default)
            logger.info("Using async SEA backend")
            from databricks.sql.backend.sea.async_backend import (
                AsyncSeaDatabricksClient,
            )

            self.backend = AsyncSeaDatabricksClient(
                server_hostname=self._server_hostname,
                port=self.port,
                http_path=self.http_path,
                http_headers=self.all_headers,
                auth_provider=self.auth_provider,
                ssl_options=self.ssl_options,
                _use_arrow_native_complex_types=self._use_arrow_native_complex_types,
                **self._kwargs,
            )

        self._session_id = await self.backend.open_session(
            session_configuration=self.session_configuration,
            catalog=self.catalog,
            schema=self.schema,
        )

        self.is_open = True
        logger.info("Successfully opened async session %s", str(self.guid_hex))

    @property
    def session_id(self) -> SessionId:
        """Get the normalized session ID."""
        if self._session_id is None:
            raise InterfaceError("Session is not open")
        return self._session_id

    @property
    def guid(self) -> Any:
        """Get the raw session ID (backend-specific)."""
        return self._session_id.guid if self._session_id else None

    @property
    def guid_hex(self) -> str:
        """Get the session ID in hex format."""
        return self._session_id.hex_guid if self._session_id else ""

    async def close(self) -> None:
        """Close the async session."""
        logger.info("Closing async session %s", self.guid_hex)
        if not self.is_open:
            logger.debug("Session appears to have been closed already")
            return

        try:
            if self.backend and self._session_id:
                await self.backend.close_session(self._session_id)
        except Exception as e:
            logger.error("Error closing async session: %s", e)

        self.is_open = False


class AsyncConnection:
    """
    Async connection to Databricks SQL endpoint.

    This class provides true async/await support for connecting to Databricks SQL
    warehouses and executing queries.

    Usage:
        async with async_connect(...) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT * FROM table")
                rows = await cursor.fetchall()
    """

    def __init__(
        self,
        server_hostname: str,
        http_path: str,
        access_token: Optional[str] = None,
        http_headers: Optional[List[Tuple[str, str]]] = None,
        session_configuration: Optional[Dict[str, Any]] = None,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        _use_arrow_native_complex_types: Optional[bool] = True,
        **kwargs,
    ) -> None:
        """
        Initialize an async connection to Databricks SQL endpoint.

        Args:
            server_hostname: Databricks instance host name
            http_path: HTTP path to the SQL endpoint or cluster
            access_token: HTTP Bearer access token (e.g., Databricks Personal Access Token)
            http_headers: Optional list of (key, value) pairs for HTTP headers
            session_configuration: Optional dictionary of Spark session parameters
            catalog: Optional initial catalog to use
            schema: Optional initial schema to use
            _use_arrow_native_complex_types: Whether to use native Arrow types for complex types
            **kwargs: Additional keyword arguments
        """
        if access_token:
            kwargs["access_token"] = access_token

        self._server_hostname = server_hostname
        self._http_path = http_path
        self._http_headers = http_headers
        self._session_configuration = session_configuration
        self._catalog = catalog
        self._schema = schema
        self._use_arrow_native_complex_types = _use_arrow_native_complex_types
        self._kwargs = kwargs

        self.session: Optional[AsyncSession] = None
        self._cursors: List["AsyncCursor"] = []
        self._is_open = False

        self.disable_pandas = kwargs.get("_disable_pandas", False)
        self.lz4_compression = kwargs.get("enable_query_result_lz4_compression", True)
        self.use_cloud_fetch = kwargs.get("use_cloud_fetch", True)
        self.use_inline_params = kwargs.get("use_inline_params", False)

        client_context = build_client_context(server_hostname, __version__, **kwargs)
        self.http_client = UnifiedHttpClient(client_context)

    async def __aenter__(self) -> "AsyncConnection":
        await self._open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _open(self) -> None:
        """Open the async connection (internal method)."""
        if self._is_open:
            return

        self.session = AsyncSession(
            server_hostname=self._server_hostname,
            http_path=self._http_path,
            http_client=self.http_client,
            http_headers=self._http_headers,
            session_configuration=self._session_configuration,
            catalog=self._catalog,
            schema=self._schema,
            _use_arrow_native_complex_types=self._use_arrow_native_complex_types,
            **self._kwargs,
        )

        await self.session.open()
        self._is_open = True

    async def close(self) -> None:
        """Close the async connection and all associated cursors."""
        if not self._is_open:
            return

        # Close all cursors
        for cursor in self._cursors:
            await cursor.close()
        self._cursors.clear()

        # Close session
        if self.session:
            await self.session.close()
            if self.session.backend:
                await self.session.backend.close()

        self._is_open = False

    @property
    def open(self) -> bool:
        """Return whether the connection is open."""
        return self._is_open and self.session is not None and self.session.is_open

    def get_session_id_hex(self) -> str:
        """Get the session ID in hex format."""
        if self.session:
            return self.session.guid_hex
        return ""

    def cursor(
        self,
        arraysize: int = DEFAULT_ARRAY_SIZE,
        buffer_size_bytes: int = DEFAULT_RESULT_BUFFER_SIZE_BYTES,
        row_limit: Optional[int] = None,
    ) -> "AsyncCursor":
        """
        Create an async cursor.

        Args:
            arraysize: The maximum number of rows in direct results
            buffer_size_bytes: The maximum number of bytes in direct results
            row_limit: The maximum number of rows in the result

        Returns:
            AsyncCursor: A new async cursor object

        Raises:
            InterfaceError: If the connection is not open
        """
        if not self.open:
            raise InterfaceError(
                "Cannot create cursor from closed connection",
                host_url=self._server_hostname,
                session_id_hex=self.get_session_id_hex(),
            )

        if self.session is None or self.session.backend is None:
            raise InterfaceError(
                "Session or backend not initialized",
                host_url=self._server_hostname,
                session_id_hex=self.get_session_id_hex(),
            )

        cursor = AsyncCursor(
            connection=self,
            backend=self.session.backend,
            arraysize=arraysize,
            result_buffer_size_bytes=buffer_size_bytes,
            row_limit=row_limit,
        )
        self._cursors.append(cursor)
        return cursor


class AsyncCursor:
    """
    Async cursor for executing queries.

    This class provides true async/await support for executing SQL queries
    and fetching results.
    """

    def __init__(
        self,
        connection: AsyncConnection,
        backend: AsyncDatabricksClient,
        result_buffer_size_bytes: int = DEFAULT_RESULT_BUFFER_SIZE_BYTES,
        arraysize: int = DEFAULT_ARRAY_SIZE,
        row_limit: Optional[int] = None,
    ) -> None:
        self.connection = connection
        self.backend = backend
        self.rowcount: int = -1
        self.buffer_size_bytes: int = result_buffer_size_bytes
        self.active_result_set: Optional[AsyncResultSet] = None
        self.arraysize: int = arraysize
        self.row_limit: Optional[int] = row_limit
        self.open: bool = True
        self.active_command_id: Optional[CommandId] = None
        self.escaper = ParamEscaper()
        self.lastrowid = None

        self.ASYNC_DEFAULT_POLLING_INTERVAL = 2

    async def __aenter__(self) -> "AsyncCursor":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def __aiter__(self):
        """Async iterator for rows."""
        if self.active_result_set:
            async for row in self.active_result_set:
                yield row
        else:
            raise ProgrammingError(
                "There is no active result set",
                host_url=self.connection._server_hostname,
                session_id_hex=self.connection.get_session_id_hex(),
            )

    def _check_not_closed(self):
        if not self.open:
            raise InterfaceError(
                "Attempting operation on closed cursor",
                host_url=self.connection._server_hostname,
                session_id_hex=self.connection.get_session_id_hex(),
            )

    def _get_session_id(self) -> SessionId:
        """Get the session ID, raising an error if not available."""
        if self.connection.session is None:
            raise InterfaceError(
                "Session not initialized",
                host_url=self.connection._server_hostname,
            )
        return self._get_session_id()

    async def _close_and_clear_active_result_set(self):
        try:
            if self.active_result_set:
                await self.active_result_set.close()
        finally:
            self.active_result_set = None

    def _determine_parameter_approach(
        self, params: Optional[TParameterCollection]
    ) -> ParameterApproach:
        """Determine the parameter approach to use."""
        if params is None:
            return ParameterApproach.NONE
        if self.connection.use_inline_params:
            return ParameterApproach.INLINE
        return ParameterApproach.NATIVE

    def _normalize_tparametersequence(
        self, params: TParameterSequence
    ) -> List[TDbsqlParameter]:
        """Normalize a sequence of parameters, retaining order."""
        output: List[TDbsqlParameter] = []
        for p in params:
            if isinstance(p, DbsqlParameterBase):
                output.append(p)  # type: ignore[arg-type]
            else:
                output.append(dbsql_parameter_from_primitive(value=p))
        return output

    def _normalize_tparameterdict(
        self, params: TParameterDict
    ) -> List[TDbsqlParameter]:
        """Normalize a dictionary of parameters."""
        return [
            dbsql_parameter_from_primitive(value=value, name=name)
            for name, value in params.items()
        ]

    def _normalize_tparametercollection(
        self, params: Optional[TParameterCollection]
    ) -> List[TDbsqlParameter]:
        """Normalize parameters to a list of TDbsqlParameter."""
        if params is None:
            return []
        if isinstance(params, dict):
            return self._normalize_tparameterdict(params)  # type: ignore[arg-type]
        if isinstance(params, Sequence):
            return self._normalize_tparametersequence(list(params))
        return []

    def _all_dbsql_parameters_are_named(self, params: List[TDbsqlParameter]) -> bool:
        """Return True if all parameters have a non-null name attribute."""
        return all([i.name is not None for i in params])

    def _determine_parameter_structure(
        self, parameters: List[TDbsqlParameter]
    ) -> ParameterStructure:
        """Determine if parameters are named or positional."""
        if self._all_dbsql_parameters_are_named(parameters):
            return ParameterStructure.NAMED
        return ParameterStructure.POSITIONAL

    def _prepare_inline_parameters(
        self, stmt: str, params: Optional[Union[Sequence, Dict[str, Any]]]
    ) -> Tuple[str, List]:
        """Prepare statement with inline parameters."""
        escaped_values = self.escaper.escape_args(params)
        rendered_statement = inject_parameters(stmt, escaped_values)
        return rendered_statement, NO_NATIVE_PARAMS

    def _prepare_native_parameters(
        self,
        stmt: str,
        params: List[TDbsqlParameter],
        param_structure: ParameterStructure,
    ) -> Tuple[str, List[TSparkParameter]]:
        """Prepare statement with native parameters."""
        output = [
            p.as_tspark_param(named=param_structure == ParameterStructure.NAMED)
            for p in params
        ]
        return stmt, output

    async def execute(
        self,
        operation: str,
        parameters: Optional[TParameterCollection] = None,
        enforce_embedded_schema_correctness: bool = False,
    ) -> "AsyncCursor":
        """
        Execute a query asynchronously and wait for execution to complete.

        Args:
            operation: SQL query to execute
            parameters: Optional parameters to bind to the query
            enforce_embedded_schema_correctness: Whether to enforce schema correctness

        Returns:
            self for method chaining
        """
        logger.debug(
            "AsyncCursor.execute(operation=%s, parameters=%s)", operation, parameters
        )

        param_approach = self._determine_parameter_approach(parameters)
        if param_approach == ParameterApproach.NONE:
            prepared_params = NO_NATIVE_PARAMS
            prepared_operation = operation
        elif param_approach == ParameterApproach.INLINE:
            prepared_operation, prepared_params = self._prepare_inline_parameters(
                operation, parameters
            )
        elif param_approach == ParameterApproach.NATIVE:
            normalized_parameters = self._normalize_tparametercollection(parameters)
            param_structure = self._determine_parameter_structure(normalized_parameters)
            transformed_operation = transform_paramstyle(
                operation, normalized_parameters, param_structure
            )
            prepared_operation, prepared_params = self._prepare_native_parameters(
                transformed_operation, normalized_parameters, param_structure
            )

        self._check_not_closed()
        await self._close_and_clear_active_result_set()

        self.active_result_set = await self.backend.execute_command(
            operation=prepared_operation,
            session_id=self._get_session_id(),
            max_rows=self.arraysize,
            max_bytes=self.buffer_size_bytes,
            lz4_compression=self.connection.lz4_compression,
            cursor=self,
            use_cloud_fetch=self.connection.use_cloud_fetch,
            parameters=prepared_params,
            async_op=False,
            enforce_embedded_schema_correctness=enforce_embedded_schema_correctness,
            row_limit=self.row_limit,
        )

        return self

    async def execute_async(
        self,
        operation: str,
        parameters: Optional[TParameterCollection] = None,
        enforce_embedded_schema_correctness: bool = False,
    ) -> "AsyncCursor":
        """
        Execute a query asynchronously without waiting for completion.

        The query will be submitted to the server and this method returns immediately.
        Use get_query_state() and get_async_execution_result() to poll for completion.

        Args:
            operation: SQL query to execute
            parameters: Optional parameters to bind to the query
            enforce_embedded_schema_correctness: Whether to enforce schema correctness

        Returns:
            self for method chaining
        """
        param_approach = self._determine_parameter_approach(parameters)
        if param_approach == ParameterApproach.NONE:
            prepared_params = NO_NATIVE_PARAMS
            prepared_operation = operation
        elif param_approach == ParameterApproach.INLINE:
            prepared_operation, prepared_params = self._prepare_inline_parameters(
                operation, parameters
            )
        elif param_approach == ParameterApproach.NATIVE:
            normalized_parameters = self._normalize_tparametercollection(parameters)
            param_structure = self._determine_parameter_structure(normalized_parameters)
            transformed_operation = transform_paramstyle(
                operation, normalized_parameters, param_structure
            )
            prepared_operation, prepared_params = self._prepare_native_parameters(
                transformed_operation, normalized_parameters, param_structure
            )

        self._check_not_closed()
        await self._close_and_clear_active_result_set()

        await self.backend.execute_command(
            operation=prepared_operation,
            session_id=self._get_session_id(),
            max_rows=self.arraysize,
            max_bytes=self.buffer_size_bytes,
            lz4_compression=self.connection.lz4_compression,
            cursor=self,
            use_cloud_fetch=self.connection.use_cloud_fetch,
            parameters=prepared_params,
            async_op=True,
            enforce_embedded_schema_correctness=enforce_embedded_schema_correctness,
            row_limit=self.row_limit,
        )

        return self

    async def get_query_state(self) -> CommandState:
        """
        Get the state of the async executing query.

        Returns:
            CommandState: The current state of the command
        """
        self._check_not_closed()
        if self.active_command_id is None:
            raise Error("No active command to get state for")
        return await self.backend.get_query_state(self.active_command_id)

    async def is_query_pending(self) -> bool:
        """
        Check if the async executing query is still pending.

        Returns:
            bool: True if query is still pending or running
        """
        operation_state = await self.get_query_state()
        return operation_state in [CommandState.PENDING, CommandState.RUNNING]

    async def get_async_execution_result(self) -> "AsyncCursor":
        """
        Wait for async query completion and fetch results.

        This method polls the query status using asyncio.sleep (non-blocking)
        until the query completes, then fetches the results.

        Returns:
            self for method chaining
        """
        self._check_not_closed()

        while await self.is_query_pending():
            # Use asyncio.sleep for non-blocking wait
            await asyncio.sleep(self.ASYNC_DEFAULT_POLLING_INTERVAL)

        operation_state = await self.get_query_state()
        if operation_state == CommandState.SUCCEEDED:
            if self.active_command_id is None:
                raise Error("No active command to get results for")
            self.active_result_set = await self.backend.get_execution_result(
                self.active_command_id, self
            )
            return self
        else:
            raise OperationalError(
                f"get_execution_result failed with Operation status {operation_state}",
                host_url=self.connection._server_hostname,
                session_id_hex=self.connection.get_session_id_hex(),
            )

    async def fetchone(self) -> Optional[Row]:
        """
        Fetch the next row of a query result set asynchronously.

        Returns:
            A single Row object, or None when no more data is available
        """
        self._check_not_closed()
        if self.active_result_set:
            return await self.active_result_set.fetchone()
        else:
            raise ProgrammingError(
                "There is no active result set",
                host_url=self.connection._server_hostname,
                session_id_hex=self.connection.get_session_id_hex(),
            )

    async def fetchmany(self, size: Optional[int] = None) -> List[Row]:
        """
        Fetch the next set of rows of a query result asynchronously.

        Args:
            size: Number of rows to fetch (defaults to arraysize)

        Returns:
            List of Row objects
        """
        self._check_not_closed()
        if self.active_result_set:
            return await self.active_result_set.fetchmany(size or self.arraysize)
        else:
            raise ProgrammingError(
                "There is no active result set",
                host_url=self.connection._server_hostname,
                session_id_hex=self.connection.get_session_id_hex(),
            )

    async def fetchall(self) -> List[Row]:
        """
        Fetch all remaining rows of a query result asynchronously.

        Returns:
            List of Row objects
        """
        self._check_not_closed()
        if self.active_result_set:
            return await self.active_result_set.fetchall()
        else:
            raise ProgrammingError(
                "There is no active result set",
                host_url=self.connection._server_hostname,
                session_id_hex=self.connection.get_session_id_hex(),
            )

    async def fetchall_arrow(self) -> "pyarrow.Table":
        """
        Fetch all remaining rows as a PyArrow table asynchronously.

        Returns:
            PyArrow Table
        """
        self._check_not_closed()
        if self.active_result_set:
            return await self.active_result_set.fetchall_arrow()
        else:
            raise ProgrammingError(
                "There is no active result set",
                host_url=self.connection._server_hostname,
                session_id_hex=self.connection.get_session_id_hex(),
            )

    async def fetchmany_arrow(self, size: int) -> "pyarrow.Table":
        """
        Fetch the next set of rows as a PyArrow table asynchronously.

        Args:
            size: Number of rows to fetch

        Returns:
            PyArrow Table
        """
        self._check_not_closed()
        if self.active_result_set:
            return await self.active_result_set.fetchmany_arrow(size)
        else:
            raise ProgrammingError(
                "There is no active result set",
                host_url=self.connection._server_hostname,
                session_id_hex=self.connection.get_session_id_hex(),
            )

    async def cancel(self) -> None:
        """
        Cancel a running command asynchronously.

        The command should be closed to free resources from the server.
        """
        if self.active_command_id is not None:
            await self.backend.cancel_command(self.active_command_id)
        else:
            logger.warning(
                "Attempting to cancel a command, but there is no "
                "currently executing command"
            )

    async def close(self) -> None:
        """Close the cursor asynchronously."""
        self.open = False
        self.active_command_id = None
        if self.active_result_set:
            await self._close_and_clear_active_result_set()

    # Metadata operations
    async def catalogs(self) -> "AsyncCursor":
        """Get all available catalogs asynchronously."""
        self._check_not_closed()
        await self._close_and_clear_active_result_set()
        self.active_result_set = await self.backend.get_catalogs(
            session_id=self._get_session_id(),
            max_rows=self.arraysize,
            max_bytes=self.buffer_size_bytes,
            cursor=self,
        )
        return self

    async def schemas(
        self, catalog_name: Optional[str] = None, schema_name: Optional[str] = None
    ) -> "AsyncCursor":
        """Get schemas asynchronously."""
        self._check_not_closed()
        await self._close_and_clear_active_result_set()
        self.active_result_set = await self.backend.get_schemas(
            session_id=self._get_session_id(),
            max_rows=self.arraysize,
            max_bytes=self.buffer_size_bytes,
            cursor=self,
            catalog_name=catalog_name,
            schema_name=schema_name,
        )
        return self

    async def tables(
        self,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        table_types: Optional[List[str]] = None,
    ) -> "AsyncCursor":
        """Get tables asynchronously."""
        self._check_not_closed()
        await self._close_and_clear_active_result_set()
        self.active_result_set = await self.backend.get_tables(
            session_id=self._get_session_id(),
            max_rows=self.arraysize,
            max_bytes=self.buffer_size_bytes,
            cursor=self,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            table_types=table_types,
        )
        return self

    async def columns(
        self,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        column_name: Optional[str] = None,
    ) -> "AsyncCursor":
        """Get columns asynchronously."""
        self._check_not_closed()
        await self._close_and_clear_active_result_set()
        self.active_result_set = await self.backend.get_columns(
            session_id=self._get_session_id(),
            max_rows=self.arraysize,
            max_bytes=self.buffer_size_bytes,
            cursor=self,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            column_name=column_name,
        )
        return self

    @property
    def description(self) -> Optional[List[Tuple]]:
        """Get the column descriptions of the result set."""
        if self.active_result_set:
            return self.active_result_set.description
        return None

    @property
    def rownumber(self) -> int:
        """Get the current row number."""
        return self.active_result_set.rownumber if self.active_result_set else 0

    @property
    def query_id(self) -> Optional[str]:
        """Get the query ID of the last executed query."""
        if self.active_command_id is not None:
            return self.active_command_id.to_hex_guid()
        return None


async def async_connect(
    server_hostname: str,
    http_path: str,
    access_token: Optional[str] = None,
    http_headers: Optional[List[Tuple[str, str]]] = None,
    session_configuration: Optional[Dict[str, Any]] = None,
    catalog: Optional[str] = None,
    schema: Optional[str] = None,
    **kwargs,
) -> AsyncConnection:
    """
    Create and open an async connection to Databricks SQL endpoint.

    This is the main entry point for async connections. It creates an AsyncConnection
    and opens it, returning a ready-to-use connection object.

    Usage:
        async with await async_connect(...) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT * FROM table")
                rows = await cursor.fetchall()

        # Or without context manager:
        conn = await async_connect(...)
        try:
            cursor = conn.cursor()
            await cursor.execute("SELECT 1")
            print(await cursor.fetchall())
        finally:
            await conn.close()

    Args:
        server_hostname: Databricks instance host name
        http_path: HTTP path to the SQL endpoint or warehouse
        access_token: HTTP Bearer access token (e.g., Databricks Personal Access Token)
        http_headers: Optional list of (key, value) pairs for HTTP headers
        session_configuration: Optional dictionary of Spark session parameters
        catalog: Optional initial catalog to use
        schema: Optional initial schema to use
        **kwargs: Additional keyword arguments

    Returns:
        AsyncConnection: An open async connection object
    """
    connection = AsyncConnection(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=access_token,
        http_headers=http_headers,
        session_configuration=session_configuration,
        catalog=catalog,
        schema=schema,
        **kwargs,
    )
    await connection._open()
    return connection
