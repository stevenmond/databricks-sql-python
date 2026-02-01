"""
Async SEA result set implementation.

This module provides an async result set implementation for the SEA backend
with true async/await support.
"""

from __future__ import annotations

from typing import Any, List, Optional, TYPE_CHECKING

import logging

from databricks.sql.backend.sea.models.base import ResultData, ResultManifest
from databricks.sql.backend.sea.utils.conversion import SqlTypeConverter

try:
    import pyarrow
except ImportError:
    pyarrow = None

if TYPE_CHECKING:
    from databricks.sql.async_client import AsyncConnection
    from databricks.sql.backend.sea.async_backend import AsyncSeaDatabricksClient

from databricks.sql.types import Row
from databricks.sql.backend.sea.queue import JsonQueue, SeaResultSetQueueFactory
from databricks.sql.backend.types import ExecuteResponse
from databricks.sql.backend.async_result_set import AsyncResultSet

logger = logging.getLogger(__name__)


class AsyncSeaResultSet(AsyncResultSet):
    """Async ResultSet implementation for SEA backend."""

    def __init__(
        self,
        connection: "AsyncConnection",
        execute_response: ExecuteResponse,
        sea_client: "AsyncSeaDatabricksClient",
        result_data: ResultData,
        manifest: ResultManifest,
        buffer_size_bytes: int = 104857600,
        arraysize: int = 10000,
    ):
        """
        Initialize an AsyncSeaResultSet with the response from a SEA query execution.

        Args:
            connection: The parent async connection
            execute_response: Response from the execute command
            sea_client: The AsyncSeaDatabricksClient instance for direct access
            result_data: Result data from SEA response
            manifest: Manifest from SEA response
            buffer_size_bytes: Buffer size for fetching results
            arraysize: Default number of rows to fetch
        """
        self.manifest = manifest
        self._sea_client = sea_client

        statement_id = execute_response.command_id.to_sea_statement_id()
        if statement_id is None:
            raise ValueError("Command ID is not a SEA statement ID")

        # Build results queue using the sync factory (queue operations are sync)
        # For cloud fetch, we use the sync http_client from the session.
        # Note: We pass sea_client=None because the LinkFetcher in SeaCloudFetchQueue
        # uses threading and sync methods, but AsyncSeaDatabricksClient has async methods.
        # This means only the initial batch of external links will be processed for cloud fetch.
        # For true async cloud fetch with pagination, we would need an async queue factory.
        http_client = None
        ssl_options = None
        if hasattr(connection, 'session') and connection.session is not None:
            ssl_options = connection.session.ssl_options
            http_client = connection.session.http_client

        results_queue = SeaResultSetQueueFactory.build_queue(
            result_data,
            self.manifest,
            statement_id,
            ssl_options=ssl_options,
            description=execute_response.description,
            max_download_threads=sea_client.max_download_threads,
            sea_client=None,  # Pass None - async client not compatible with sync LinkFetcher
            lz4_compressed=execute_response.lz4_compressed,
            http_client=http_client,
        )

        # Call parent constructor with common attributes
        super().__init__(
            connection=connection,
            backend=sea_client,
            arraysize=arraysize,
            buffer_size_bytes=buffer_size_bytes,
            command_id=execute_response.command_id,
            status=execute_response.status,
            has_been_closed_server_side=execute_response.has_been_closed_server_side,
            results_queue=results_queue,
            description=execute_response.description,
            is_staging_operation=execute_response.is_staging_operation,
            lz4_compressed=execute_response.lz4_compressed,
            arrow_schema_bytes=execute_response.arrow_schema_bytes,
        )

    def _convert_json_types(self, row: List[str]) -> List[Any]:
        """
        Convert string values in the row to appropriate Python types based on column metadata.
        """
        converted_row = []

        for i, value in enumerate(row):
            column_name = self.description[i][0]
            column_type = self.description[i][1]
            precision = self.description[i][4]
            scale = self.description[i][5]

            converted_value = SqlTypeConverter.convert_value(
                value,
                column_type,
                column_name=column_name,
                precision=precision,
                scale=scale,
            )
            converted_row.append(converted_value)

        return converted_row

    def _convert_json_to_arrow_table(self, rows: List[List[str]]) -> "pyarrow.Table":
        """Convert raw data rows to Arrow table."""
        if not rows:
            return pyarrow.Table.from_pydict({})

        converted_rows_iter = (self._convert_json_types(row) for row in rows)
        cols = list(map(list, zip(*converted_rows_iter)))

        names = [col[0] for col in self.description]
        return pyarrow.Table.from_arrays(cols, names=names)

    def _create_json_table(self, rows: List[List[str]]) -> List[Row]:
        """Convert raw data rows to Row objects with named columns."""
        ResultRow = Row(*[col[0] for col in self.description])
        return [ResultRow(*self._convert_json_types(row)) for row in rows]

    def _fetchmany_json_sync(self, size: int) -> List[List[str]]:
        """
        Fetch the next set of rows as JSON (synchronous internal method).
        The queue operations are synchronous.
        """
        if size < 0:
            raise ValueError(f"size argument for fetchmany is {size} but must be >= 0")

        results = self.results.next_n_rows(size)
        self._next_row_index += len(results)

        return results

    def _fetchall_json_sync(self) -> List[List[str]]:
        """Fetch all remaining rows as JSON (synchronous internal method)."""
        results = self.results.remaining_rows()
        self._next_row_index += len(results)

        return results

    async def fetchmany_arrow(self, size: int) -> "pyarrow.Table":
        """
        Fetch the next set of rows as an Arrow table asynchronously.

        Args:
            size: Number of rows to fetch

        Returns:
            PyArrow Table containing the fetched rows
        """
        if size < 0:
            raise ValueError(f"size argument for fetchmany is {size} but must be >= 0")

        # Queue operations are synchronous, but we make this method async
        # for API consistency and potential future async cloud fetch
        results = self.results.next_n_rows(size)
        if isinstance(self.results, JsonQueue):
            results = self._convert_json_to_arrow_table(results)

        self._next_row_index += results.num_rows

        return results

    async def fetchall_arrow(self) -> "pyarrow.Table":
        """Fetch all remaining rows as an Arrow table asynchronously."""
        results = self.results.remaining_rows()
        if isinstance(self.results, JsonQueue):
            results = self._convert_json_to_arrow_table(results)

        self._next_row_index += results.num_rows

        return results

    async def fetchone(self) -> Optional[Row]:
        """
        Fetch the next row of a query result set asynchronously.

        Returns:
            A single Row object or None if no more rows are available
        """
        if isinstance(self.results, JsonQueue):
            rows = self._fetchmany_json_sync(1)
            res = self._create_json_table(rows)
        else:
            table = await self.fetchmany_arrow(1)
            res = self._convert_arrow_table(table)

        return res[0] if res else None

    async def fetchmany(self, size: int) -> List[Row]:
        """
        Fetch the next set of rows of a query result asynchronously.

        Args:
            size: Number of rows to fetch

        Returns:
            List of Row objects
        """
        if isinstance(self.results, JsonQueue):
            rows = self._fetchmany_json_sync(size)
            return self._create_json_table(rows)
        else:
            table = await self.fetchmany_arrow(size)
            return self._convert_arrow_table(table)

    async def fetchall(self) -> List[Row]:
        """
        Fetch all remaining rows of a query result asynchronously.

        Returns:
            List of Row objects containing all remaining rows
        """
        if isinstance(self.results, JsonQueue):
            rows = self._fetchall_json_sync()
            return self._create_json_table(rows)
        else:
            table = await self.fetchall_arrow()
            return self._convert_arrow_table(table)
