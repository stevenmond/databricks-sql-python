"""
Async queue implementations for SEA cloud fetch.

This module provides async versions of the LinkFetcher and SeaCloudFetchQueue
for use with the async SEA backend.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

try:
    import pyarrow
except ImportError:
    pyarrow = None

if TYPE_CHECKING:
    from databricks.sql.backend.sea.async_backend import AsyncSeaDatabricksClient
    from databricks.sql.backend.sea.models.base import ExternalLink, ResultData

from databricks.sql.backend.sea.queue import LinkFetcher
from databricks.sql.cloudfetch.async_download_manager import (
    AsyncResultFileDownloadManager,
)
from databricks.sql.exc import ProgrammingError
from databricks.sql.types import SSLOptions
from databricks.sql.utils import create_arrow_table_from_arrow_file

logger = logging.getLogger(__name__)


class AsyncLinkFetcher:
    """
    Async version of LinkFetcher that uses asyncio instead of threading.

    This class incrementally retrieves external links for a result set produced
    by the SEA backend and feeds them to an AsyncResultFileDownloadManager.

    Key differences from sync LinkFetcher:
    - Uses asyncio.Task instead of threading.Thread
    - Uses asyncio.Event and asyncio.Condition instead of threading versions
    - Calls async get_chunk_links method with await
    """

    def __init__(
        self,
        download_manager: AsyncResultFileDownloadManager,
        backend: "AsyncSeaDatabricksClient",
        statement_id: str,
        initial_links: List["ExternalLink"],
        total_chunk_count: int,
    ):
        """
        Initialize the async link fetcher.

        Args:
            download_manager: Async download manager to add links to
            backend: Async SEA backend client
            statement_id: Statement ID for the query
            initial_links: Initial list of external links
            total_chunk_count: Total number of chunks expected
        """
        self.download_manager = download_manager
        self.backend = backend
        self._statement_id = statement_id

        self._shutdown_event = asyncio.Event()
        self._link_data_update = asyncio.Condition()
        self._error: Optional[Exception] = None
        self.chunk_index_to_link: Dict[int, "ExternalLink"] = {}

        self._add_links(initial_links)
        self.total_chunk_count = total_chunk_count

        self._worker_task: Optional[asyncio.Task] = None

        logger.debug(
            "AsyncLinkFetcher[%s]: initialized with %d initial link(s); expecting %d total chunk(s)",
            statement_id,
            len(initial_links),
            total_chunk_count,
        )

    def _add_links(self, links: List["ExternalLink"]) -> None:
        """Cache links locally and enqueue them with the download manager."""
        logger.debug(
            "AsyncLinkFetcher[%s]: caching %d link(s) – chunks %s",
            self._statement_id,
            len(links),
            ", ".join(str(link.chunk_index) for link in links) if links else "<none>",
        )
        for link in links:
            self.chunk_index_to_link[link.chunk_index] = link
            self.download_manager.add_link(LinkFetcher._convert_to_thrift_link(link))

    def _get_next_chunk_index(self) -> Optional[int]:
        """Return the next chunk_index to request, or None if we have them all."""
        max_chunk_index = max(self.chunk_index_to_link.keys(), default=None)
        if max_chunk_index is None:
            return 0
        max_link = self.chunk_index_to_link[max_chunk_index]
        return max_link.next_chunk_index

    async def _trigger_next_batch_download(self) -> bool:
        """Fetch the next batch of links from the backend asynchronously."""
        logger.debug(
            "AsyncLinkFetcher[%s]: requesting next batch of links", self._statement_id
        )
        next_chunk_index = self._get_next_chunk_index()
        if next_chunk_index is None:
            return False

        try:
            # Async call to get_chunk_links
            links = await self.backend.get_chunk_links(
                self._statement_id, next_chunk_index
            )
            async with self._link_data_update:
                self._add_links(links)
                self._link_data_update.notify_all()
        except Exception as e:
            logger.error(
                "AsyncLinkFetcher: Error fetching links for chunk %d: %s",
                next_chunk_index,
                e,
            )
            async with self._link_data_update:
                self._error = e
                self._link_data_update.notify_all()
            return False

        logger.debug(
            "AsyncLinkFetcher[%s]: received %d new link(s)",
            self._statement_id,
            len(links),
        )
        return True

    async def get_chunk_link(self, chunk_index: int) -> Optional["ExternalLink"]:
        """
        Return the ExternalLink for the given chunk_index asynchronously.

        This method waits (non-blocking) until the link is available.
        """
        logger.debug(
            "AsyncLinkFetcher[%s]: waiting for link of chunk %d",
            self._statement_id,
            chunk_index,
        )
        if chunk_index >= self.total_chunk_count:
            return None

        async with self._link_data_update:
            while chunk_index not in self.chunk_index_to_link:
                if self._error:
                    raise self._error
                if self._shutdown_event.is_set():
                    raise ProgrammingError(
                        f"AsyncLinkFetcher is shutting down without providing link for chunk index {chunk_index}"
                    )
                await self._link_data_update.wait()

            return self.chunk_index_to_link[chunk_index]

    async def _worker_loop(self) -> None:
        """Entry point for the background task."""
        logger.debug("AsyncLinkFetcher[%s]: worker task started", self._statement_id)
        while not self._shutdown_event.is_set():
            links_downloaded = await self._trigger_next_batch_download()
            if not links_downloaded:
                self._shutdown_event.set()
        logger.debug("AsyncLinkFetcher[%s]: worker task exiting", self._statement_id)
        async with self._link_data_update:
            self._link_data_update.notify_all()

    def start(self) -> None:
        """Start the background worker task."""
        logger.debug("AsyncLinkFetcher[%s]: starting worker task", self._statement_id)
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        """Signal the worker task to stop and wait for completion."""
        logger.debug("AsyncLinkFetcher[%s]: stopping worker task", self._statement_id)
        self._shutdown_event.set()
        if self._worker_task:
            # Wait for the task to complete
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "AsyncLinkFetcher[%s]: worker task did not stop in time, cancelling",
                    self._statement_id,
                )
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass
        logger.debug("AsyncLinkFetcher[%s]: worker task stopped", self._statement_id)


class AsyncSeaCloudFetchQueue:
    """
    Async queue implementation for EXTERNAL_LINKS disposition with ARROW format.

    This queue uses AsyncLinkFetcher for background link fetching and
    AsyncResultFileDownloadManager for parallel downloads.
    """

    def __init__(
        self,
        result_data: "ResultData",
        max_download_threads: int,
        ssl_options: Optional[SSLOptions],
        sea_client: "AsyncSeaDatabricksClient",
        statement_id: str,
        total_chunk_count: int,
        lz4_compressed: bool = False,
        description: List[Tuple] = [],
    ):
        """
        Initialize the async SEA CloudFetchQueue.

        Args:
            result_data: Result data from SEA response
            max_download_threads: Maximum concurrent downloads
            ssl_options: SSL options for downloads
            sea_client: Async SEA client for fetching additional links
            statement_id: Statement ID for the query
            total_chunk_count: Total number of chunks
            lz4_compressed: Whether data is LZ4 compressed
            description: Column descriptions
        """
        self._statement_id = statement_id
        self._lz4_compressed = lz4_compressed
        self._description = description
        self._current_chunk_index = 0
        self._table_row_index = 0

        initial_links = result_data.external_links or []

        # Create async download manager
        thrift_links = [
            LinkFetcher._convert_to_thrift_link(link)
            for link in initial_links
            if link.row_count > 0
        ]

        self._download_manager = AsyncResultFileDownloadManager(
            links=thrift_links,
            max_download_threads=max_download_threads,
            lz4_compressed=lz4_compressed,
            ssl_options=ssl_options,
            statement_id=statement_id,
        )

        # Create async link fetcher if needed
        self._link_fetcher: Optional[AsyncLinkFetcher] = None
        if total_chunk_count > 0:
            self._link_fetcher = AsyncLinkFetcher(
                download_manager=self._download_manager,
                backend=sea_client,
                statement_id=statement_id,
                initial_links=initial_links,
                total_chunk_count=total_chunk_count,
            )
            self._link_fetcher.start()

        # Current table being read
        self._table: Optional["pyarrow.Table"] = None
        self._initialized = False

        logger.debug(
            "AsyncSeaCloudFetchQueue: initialized for statement %s, total chunks: %d",
            statement_id,
            total_chunk_count,
        )

    async def _ensure_initialized(self) -> None:
        """Initialize the first table if not already done."""
        if not self._initialized:
            self._table = await self._create_next_table()
            self._initialized = True

    async def _create_next_table(self) -> Optional["pyarrow.Table"]:
        """Create next table by retrieving the next downloaded file."""
        if self._link_fetcher is None:
            return None

        chunk_link = await self._link_fetcher.get_chunk_link(self._current_chunk_index)
        if chunk_link is None:
            return None

        row_offset = chunk_link.row_offset

        # Get the downloaded file from the async download manager
        downloaded_file = await self._download_manager.get_next_downloaded_file(
            row_offset
        )
        if not downloaded_file:
            logger.debug(
                "AsyncSeaCloudFetchQueue: Cannot find downloaded file for row %d",
                row_offset,
            )
            return None

        # Convert to Arrow table
        arrow_table = create_arrow_table_from_arrow_file(
            downloaded_file.file_bytes, self._description
        )

        self._current_chunk_index += 1
        return arrow_table

    async def next_n_rows(self, n_rows: int) -> "pyarrow.Table":
        """
        Get the next n rows from the queue asynchronously.

        Args:
            n_rows: Number of rows to fetch

        Returns:
            PyArrow Table containing the rows
        """
        await self._ensure_initialized()

        if self._table is None:
            return self._create_empty_table()

        results = self._table.slice(self._table_row_index, n_rows)
        self._table_row_index += results.num_rows

        # If we need more rows and exhausted current table
        while results.num_rows < n_rows and self._table is not None:
            # Get next table
            self._table = await self._create_next_table()
            self._table_row_index = 0

            if self._table is None:
                break

            remaining = n_rows - results.num_rows
            table_slice = self._table.slice(0, remaining)
            self._table_row_index = table_slice.num_rows

            # Concatenate results
            if pyarrow:
                results = pyarrow.concat_tables([results, table_slice])

        return results

    async def remaining_rows(self) -> "pyarrow.Table":
        """
        Get all remaining rows from the queue asynchronously.

        Returns:
            PyArrow Table containing all remaining rows
        """
        await self._ensure_initialized()

        if self._table is None:
            return self._create_empty_table()

        # Get remaining rows from current table
        results = self._table.slice(self._table_row_index)
        self._table_row_index += results.num_rows

        # Get all remaining tables
        while True:
            self._table = await self._create_next_table()
            self._table_row_index = 0

            if self._table is None:
                break

            if pyarrow:
                results = pyarrow.concat_tables([results, self._table])
            self._table_row_index = self._table.num_rows

        return results

    def _create_empty_table(self) -> "pyarrow.Table":
        """Create an empty table with the correct schema."""
        if pyarrow:
            return pyarrow.Table.from_pydict({})
        raise RuntimeError("PyArrow is required for Arrow result format")

    async def close(self) -> None:
        """Close the queue and release resources."""
        if self._link_fetcher:
            await self._link_fetcher.stop()
        self._download_manager.shutdown()
