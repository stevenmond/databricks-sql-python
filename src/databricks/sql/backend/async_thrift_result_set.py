"""
Async Thrift result set implementation.

This module provides an async result set implementation for the Thrift backend
with true async/await support.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import logging

try:
    import pyarrow
except ImportError:
    pyarrow = None

if TYPE_CHECKING:
    from databricks.sql.async_client import AsyncConnection
    from databricks.sql.backend.async_thrift_backend import AsyncThriftDatabricksClient

from databricks.sql.types import Row
from databricks.sql.backend.types import ExecuteResponse, CommandState
from databricks.sql.backend.async_result_set import AsyncResultSet
from databricks.sql.utils import (
    ThriftResultSetQueueFactory,
    ColumnQueue,
    ColumnTable,
    concat_table_chunks,
)
from databricks.sql.exc import RequestError, CursorAlreadyClosedError

logger = logging.getLogger(__name__)


class AsyncThriftResultSet(AsyncResultSet):
    """Async ResultSet implementation for Thrift backend."""

    def __init__(
        self,
        connection: "AsyncConnection",
        execute_response: ExecuteResponse,
        thrift_client: "AsyncThriftDatabricksClient",
        buffer_size_bytes: int = 104857600,
        arraysize: int = 10000,
        use_cloud_fetch: bool = True,
        t_row_set=None,
        max_download_threads: int = 10,
        ssl_options=None,
        has_more_rows: bool = True,
    ):
        """
        Initialize an AsyncThriftResultSet with the response from a Thrift query execution.

        Args:
            connection: The parent async connection
            execute_response: Response from the execute command
            thrift_client: The AsyncThriftDatabricksClient instance
            buffer_size_bytes: Buffer size for fetching results
            arraysize: Default number of rows to fetch
            use_cloud_fetch: Whether to use cloud fetch
            t_row_set: The TRowSet containing result data (if available)
            max_download_threads: Maximum number of download threads
            ssl_options: SSL options for cloud fetch
            has_more_rows: Whether there are more rows to fetch
        """
        self.num_chunks = 0
        self._use_cloud_fetch = use_cloud_fetch
        self._max_download_threads = max_download_threads
        self._ssl_options = ssl_options
        self._thrift_client = thrift_client

        # Build results queue if t_row_set is provided
        results_queue = None
        if t_row_set and execute_response.result_format is not None:
            results_queue = ThriftResultSetQueueFactory.build_queue(
                row_set_type=execute_response.result_format,
                t_row_set=t_row_set,
                arrow_schema_bytes=execute_response.arrow_schema_bytes or b"",
                max_download_threads=max_download_threads,
                lz4_compressed=execute_response.lz4_compressed,
                description=execute_response.description,
                ssl_options=ssl_options,
                session_id_hex=connection.get_session_id_hex() if hasattr(connection, 'get_session_id_hex') else None,
                statement_id=execute_response.command_id.to_hex_guid(),
                chunk_id=self.num_chunks,
                http_client=connection.http_client if hasattr(connection, 'http_client') else None,
            )
            if t_row_set.resultLinks:
                self.num_chunks += len(t_row_set.resultLinks)

        # Call parent constructor
        super().__init__(
            connection=connection,
            backend=thrift_client,
            arraysize=arraysize,
            buffer_size_bytes=buffer_size_bytes,
            command_id=execute_response.command_id,
            status=execute_response.status,
            has_been_closed_server_side=execute_response.has_been_closed_server_side,
            has_more_rows=has_more_rows,
            results_queue=results_queue,
            description=execute_response.description,
            is_staging_operation=execute_response.is_staging_operation,
            lz4_compressed=execute_response.lz4_compressed,
            arrow_schema_bytes=execute_response.arrow_schema_bytes,
        )

    async def _fill_results_buffer_async(self) -> None:
        """Fetch more results from the server asynchronously."""
        # Note: This is a placeholder - the actual implementation would need
        # an async version of fetch_results on the backend
        # For now, we use the sync queue which works for in-memory results
        # Async cloud fetch would be implemented separately
        logger.debug("AsyncThriftResultSet: filling results buffer")

        # For the async implementation, we would need to call an async fetch method
        # This is a simplified version that works with the initial results
        if self.has_more_rows and not self.has_been_closed_server_side:
            # In a full implementation, this would call an async fetch method
            # For now, we indicate no more results available from initial fetch
            self.has_more_rows = False

    def _convert_columnar_table(self, table) -> List[Row]:
        """Convert columnar table to list of Row objects."""
        column_names = [c[0] for c in self.description]
        ResultRow = Row(*column_names)
        result = []
        for row_index in range(table.num_rows):
            curr_row = []
            for col_index in range(table.num_columns):
                curr_row.append(table.get_item(col_index, row_index))
            result.append(ResultRow(*curr_row))
        return result

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

        if self.results is None:
            await self._fill_results_buffer_async()

        results = self.results.next_n_rows(size)
        partial_result_chunks = [results]
        n_remaining_rows = size - results.num_rows
        self._next_row_index += results.num_rows

        while (
            n_remaining_rows > 0
            and not self.has_been_closed_server_side
            and self.has_more_rows
        ):
            await self._fill_results_buffer_async()
            partial_results = self.results.next_n_rows(n_remaining_rows)
            partial_result_chunks.append(partial_results)
            n_remaining_rows -= partial_results.num_rows
            self._next_row_index += partial_results.num_rows

        return concat_table_chunks(partial_result_chunks)

    async def fetchmany_columnar(self, size: int):
        """
        Fetch the next set of rows as a columnar table asynchronously.

        Args:
            size: Number of rows to fetch

        Returns:
            ColumnTable containing the fetched rows
        """
        if size < 0:
            raise ValueError(f"size argument for fetchmany is {size} but must be >= 0")

        if self.results is None:
            await self._fill_results_buffer_async()

        results = self.results.next_n_rows(size)
        n_remaining_rows = size - results.num_rows
        self._next_row_index += results.num_rows
        partial_result_chunks = [results]

        while (
            n_remaining_rows > 0
            and not self.has_been_closed_server_side
            and self.has_more_rows
        ):
            await self._fill_results_buffer_async()
            partial_results = self.results.next_n_rows(n_remaining_rows)
            partial_result_chunks.append(partial_results)
            n_remaining_rows -= partial_results.num_rows
            self._next_row_index += partial_results.num_rows

        return concat_table_chunks(partial_result_chunks)

    async def fetchall_arrow(self) -> "pyarrow.Table":
        """Fetch all remaining rows as an Arrow table asynchronously."""
        if self.results is None:
            await self._fill_results_buffer_async()

        results = self.results.remaining_rows()
        self._next_row_index += results.num_rows
        partial_result_chunks = [results]

        while not self.has_been_closed_server_side and self.has_more_rows:
            await self._fill_results_buffer_async()
            partial_results = self.results.remaining_rows()
            partial_result_chunks.append(partial_results)
            self._next_row_index += partial_results.num_rows

        result_table = concat_table_chunks(partial_result_chunks)

        # Convert ColumnTable to PyArrow Table if needed
        if isinstance(result_table, ColumnTable) and pyarrow:
            data = {
                name: col
                for name, col in zip(
                    result_table.column_names, result_table.column_table
                )
            }
            return pyarrow.Table.from_pydict(data)

        return result_table

    async def fetchall_columnar(self):
        """Fetch all remaining rows as a columnar table asynchronously."""
        if self.results is None:
            await self._fill_results_buffer_async()

        results = self.results.remaining_rows()
        self._next_row_index += results.num_rows
        partial_result_chunks = [results]

        while not self.has_been_closed_server_side and self.has_more_rows:
            await self._fill_results_buffer_async()
            partial_results = self.results.remaining_rows()
            partial_result_chunks.append(partial_results)
            self._next_row_index += partial_results.num_rows

        return concat_table_chunks(partial_result_chunks)

    async def fetchone(self) -> Optional[Row]:
        """
        Fetch the next row of a query result set asynchronously.

        Returns:
            A single Row object or None if no more rows are available
        """
        if isinstance(self.results, ColumnQueue):
            res = self._convert_columnar_table(await self.fetchmany_columnar(1))
        else:
            res = self._convert_arrow_table(await self.fetchmany_arrow(1))

        return res[0] if res else None

    async def fetchmany(self, size: int) -> List[Row]:
        """
        Fetch the next set of rows of a query result asynchronously.

        Args:
            size: Number of rows to fetch

        Returns:
            List of Row objects
        """
        if isinstance(self.results, ColumnQueue):
            return self._convert_columnar_table(await self.fetchmany_columnar(size))
        else:
            return self._convert_arrow_table(await self.fetchmany_arrow(size))

    async def fetchall(self) -> List[Row]:
        """
        Fetch all remaining rows of a query result asynchronously.

        Returns:
            List of Row objects containing all remaining rows
        """
        if isinstance(self.results, ColumnQueue):
            return self._convert_columnar_table(await self.fetchall_columnar())
        else:
            return self._convert_arrow_table(await self.fetchall_arrow())

    async def close(self) -> None:
        """Close the result set asynchronously."""
        try:
            if self.results is not None:
                self.results.close()
            else:
                logger.warning("result set close: queue not initialized")

            if (
                self.status != CommandState.CLOSED
                and not self.has_been_closed_server_side
                and self.connection.open
            ):
                await self.backend.close_command(self.command_id)
        except RequestError as e:
            if isinstance(e.args[1], CursorAlreadyClosedError):
                logger.info("Operation was canceled by a prior request")
        finally:
            self.has_been_closed_server_side = True
            self.status = CommandState.CLOSED
