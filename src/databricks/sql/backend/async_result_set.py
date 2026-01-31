"""
Abstract async result set for Databricks SQL connector.

This module defines the abstract base class for async result sets that all
async backend implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING, Tuple, Any

import logging
import pandas

try:
    import pyarrow
except ImportError:
    pyarrow = None

if TYPE_CHECKING:
    from databricks.sql.async_client import AsyncConnection
from databricks.sql.backend.async_databricks_client import AsyncDatabricksClient
from databricks.sql.types import Row
from databricks.sql.exc import RequestError, CursorAlreadyClosedError
from databricks.sql.backend.types import CommandId, CommandState, ExecuteResponse

logger = logging.getLogger(__name__)


class AsyncResultSet(ABC):
    """
    Abstract base class for async result sets returned by different backend implementations.

    This class defines the interface that all concrete async result set implementations must follow.
    All fetch methods are async and should be awaited.
    """

    def __init__(
        self,
        connection: "AsyncConnection",
        backend: AsyncDatabricksClient,
        arraysize: int,
        buffer_size_bytes: int,
        command_id: CommandId,
        status: CommandState,
        has_been_closed_server_side: bool = False,
        has_more_rows: bool = False,
        results_queue=None,
        description: List[Tuple] = [],
        is_staging_operation: bool = False,
        lz4_compressed: bool = False,
        arrow_schema_bytes: Optional[bytes] = None,
    ):
        """
        An AsyncResultSet manages the results of a single command asynchronously.

        Parameters:
            :param connection: The parent async connection that was used to execute this command
            :param backend: The specialized async backend client to be invoked in the fetch phase
            :param arraysize: The max number of rows to fetch at a time (PEP-249)
            :param buffer_size_bytes: The size (in bytes) of the internal buffer + max fetch
            :param command_id: The command ID
            :param status: The command status
            :param has_been_closed_server_side: Whether the command has been closed on the server
            :param has_more_rows: Whether the command has more rows
            :param results_queue: The results queue
            :param description: column description of the results
            :param is_staging_operation: Whether the command is a staging operation
        """
        self.connection = connection
        self.backend = backend
        self.arraysize = arraysize
        self.buffer_size_bytes = buffer_size_bytes
        self._next_row_index = 0
        self.description = description
        self.command_id = command_id
        self.status = status
        self.has_been_closed_server_side = has_been_closed_server_side
        self.has_more_rows = has_more_rows
        self.results = results_queue
        self._is_staging_operation = is_staging_operation
        self.lz4_compressed = lz4_compressed
        self._arrow_schema_bytes = arrow_schema_bytes

    async def __aiter__(self):
        """Async iterator for rows."""
        while True:
            row = await self.fetchone()
            if row:
                yield row
            else:
                break

    def _convert_arrow_table(self, table) -> List[Row]:
        """Convert Arrow table to list of Row objects."""
        column_names = [c[0] for c in self.description]
        ResultRow = Row(*column_names)

        if getattr(self.connection, "disable_pandas", False) is True:
            return [
                ResultRow(*[v.as_py() for v in r]) for r in zip(*table.itercolumns())
            ]

        # Need to use nullable types
        dtype_mapping = {
            pyarrow.int8(): pandas.Int8Dtype(),
            pyarrow.int16(): pandas.Int16Dtype(),
            pyarrow.int32(): pandas.Int32Dtype(),
            pyarrow.int64(): pandas.Int64Dtype(),
            pyarrow.uint8(): pandas.UInt8Dtype(),
            pyarrow.uint16(): pandas.UInt16Dtype(),
            pyarrow.uint32(): pandas.UInt32Dtype(),
            pyarrow.uint64(): pandas.UInt64Dtype(),
            pyarrow.bool_(): pandas.BooleanDtype(),
            pyarrow.float32(): pandas.Float32Dtype(),
            pyarrow.float64(): pandas.Float64Dtype(),
            pyarrow.string(): pandas.StringDtype(),
        }

        # Rename columns for pandas compatibility
        table_renamed = table.rename_columns([str(c) for c in range(table.num_columns)])
        df = table_renamed.to_pandas(
            types_mapper=dtype_mapping.get,
            date_as_object=True,
            timestamp_as_object=True,
        )

        res = df.to_numpy(na_value=None, dtype="object")
        return [ResultRow(*v) for v in res]

    @property
    def rownumber(self):
        return self._next_row_index

    @property
    def is_staging_operation(self) -> bool:
        """Whether this result set represents a staging operation."""
        return self._is_staging_operation

    @abstractmethod
    async def fetchone(self) -> Optional[Row]:
        """Fetch the next row of a query result set asynchronously."""
        pass

    @abstractmethod
    async def fetchmany(self, size: int) -> List[Row]:
        """Fetch the next set of rows of a query result asynchronously."""
        pass

    @abstractmethod
    async def fetchall(self) -> List[Row]:
        """Fetch all remaining rows of a query result asynchronously."""
        pass

    @abstractmethod
    async def fetchmany_arrow(self, size: int) -> "pyarrow.Table":
        """Fetch the next set of rows as an Arrow table asynchronously."""
        pass

    @abstractmethod
    async def fetchall_arrow(self) -> "pyarrow.Table":
        """Fetch all remaining rows as an Arrow table asynchronously."""
        pass

    async def close(self) -> None:
        """
        Close the result set asynchronously.

        If the connection has not been closed, and the result set has not already
        been closed on the server for some other reason, issue a request to the server to close it.
        """
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
