"""
Async cloud fetch download manager.

This module provides async implementations for downloading cloud fetch results
using asyncio for parallel, non-blocking downloads.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import lz4.frame

try:
    import aiohttp
except ImportError:
    aiohttp = None

from databricks.sql.cloudfetch.downloader import (
    DownloadedFile,
    DownloadableResultSettings,
)
from databricks.sql.types import SSLOptions
from databricks.sql.exc import Error
from databricks.sql.thrift_api.TCLIService.ttypes import TSparkArrowResultLink

logger = logging.getLogger(__name__)


class AsyncResultSetDownloadHandler:
    """
    Async handler for downloading a single cloud fetch result file.

    Uses aiohttp for non-blocking HTTP downloads.
    """

    def __init__(
        self,
        settings: DownloadableResultSettings,
        link: TSparkArrowResultLink,
        ssl_options: Optional[SSLOptions] = None,
        chunk_id: int = 0,
        session_id_hex: Optional[str] = None,
        statement_id: str = "",
    ):
        """
        Initialize the async download handler.

        Args:
            settings: Download settings (compression, timeouts, etc.)
            link: The result link to download
            ssl_options: SSL configuration options
            chunk_id: Chunk identifier for logging
            session_id_hex: Session ID for logging
            statement_id: Statement ID for logging
        """
        if aiohttp is None:
            raise ImportError(
                "aiohttp is required for async cloud fetch. "
                "Install with: pip install databricks-sql-connector[async]"
            )

        self.settings = settings
        self.link = link
        self._ssl_options = ssl_options
        self.chunk_id = chunk_id
        self.session_id_hex = session_id_hex
        self.statement_id = statement_id

    async def run(self) -> DownloadedFile:
        """
        Download the file asynchronously.

        Returns:
            DownloadedFile containing the downloaded and decompressed data
        """
        logger.debug(
            "AsyncResultSetDownloadHandler: starting file download, chunk id %s, offset %s, row count %s",
            self.chunk_id,
            self.link.startRowOffset,
            self.link.rowCount,
        )

        # Check if link is already expired or is expiring
        self._validate_link(self.link, self.settings.link_expiry_buffer_secs)

        start_time = time.time()

        # Create SSL context if needed
        ssl_context = None
        if self._ssl_options:
            ssl_context = self._ssl_options.create_ssl_context()

        # Create timeout
        timeout = aiohttp.ClientTimeout(total=self.settings.download_timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                self.link.fileLink,
                headers=self.link.httpHeaders,
                ssl=ssl_context,
            ) as response:
                if response.status >= 400:
                    text = await response.text()
                    raise Exception(f"HTTP {response.status}: {text}")

                compressed_data = await response.read()

        # Log download metrics
        download_duration = time.time() - start_time
        self._log_download_metrics(
            self.link.fileLink, len(compressed_data), download_duration
        )

        # Decompress if needed
        decompressed_data = (
            self._decompress_data(compressed_data)
            if self.settings.is_lz4_compressed
            else compressed_data
        )

        # Verify size
        if len(decompressed_data) != self.link.bytesNum:
            logger.debug(
                "AsyncResultSetDownloadHandler: downloaded file size %s does not match expected %s",
                len(decompressed_data),
                self.link.bytesNum,
            )

        logger.debug(
            "AsyncResultSetDownloadHandler: successfully downloaded file, offset %s, row count %s",
            self.link.startRowOffset,
            self.link.rowCount,
        )

        return DownloadedFile(
            decompressed_data,
            self.link.startRowOffset,
            self.link.rowCount,
        )

    def _log_download_metrics(
        self, url: str, bytes_downloaded: int, duration_seconds: float
    ) -> None:
        """Log download speed metrics."""
        speed_mbps = (float(bytes_downloaded) / (1024 * 1024)) / duration_seconds
        url_endpoint = url.split("?")[0]

        logger.info(
            "Async CloudFetch download completed: %.4f MB/s, %d bytes in %.3fs from %s",
            speed_mbps,
            bytes_downloaded,
            duration_seconds,
            url_endpoint,
        )

        if speed_mbps < self.settings.min_cloudfetch_download_speed:
            logger.warning(
                "Async CloudFetch download slower than threshold: %.4f MB/s (threshold: %.1f MB/s) from %s",
                speed_mbps,
                self.settings.min_cloudfetch_download_speed,
                url,
            )

    @staticmethod
    def _validate_link(link: TSparkArrowResultLink, expiry_buffer_secs: int) -> None:
        """Validate that the link has not expired."""
        current_time = int(time.time())
        if (
            link.expiryTime <= current_time
            or link.expiryTime - current_time <= expiry_buffer_secs
        ):
            raise Error("CloudFetch link has expired")

    @staticmethod
    def _decompress_data(compressed_data: bytes) -> bytes:
        """Decompress LZ4 frame compressed data."""
        uncompressed_data, bytes_read = lz4.frame.decompress(
            compressed_data, return_bytes_read=True
        )

        # Handle chunked compression
        if bytes_read < len(compressed_data):
            d_context = lz4.frame.create_decompression_context()
            start = 0
            uncompressed_data = bytearray()
            while start < len(compressed_data):
                data, num_bytes, is_end = lz4.frame.decompress_chunk(
                    d_context, compressed_data[start:]
                )
                uncompressed_data += data
                start += num_bytes

        return bytes(uncompressed_data)


class AsyncResultFileDownloadManager:
    """
    Async manager for parallel cloud fetch downloads.

    Uses asyncio.gather() for concurrent downloads with configurable parallelism.
    """

    def __init__(
        self,
        links: List[TSparkArrowResultLink],
        max_download_threads: int,
        lz4_compressed: bool,
        ssl_options: Optional[SSLOptions] = None,
        session_id_hex: Optional[str] = None,
        statement_id: str = "",
        chunk_id: int = 0,
    ):
        """
        Initialize the async download manager.

        Args:
            links: List of result links to download
            max_download_threads: Maximum concurrent downloads
            lz4_compressed: Whether files are LZ4 compressed
            ssl_options: SSL configuration options
            session_id_hex: Session ID for logging
            statement_id: Statement ID for logging
            chunk_id: Starting chunk ID
        """
        self._pending_links: List[Tuple[int, TSparkArrowResultLink]] = []
        self.chunk_id = chunk_id

        for i, link in enumerate(links, start=chunk_id):
            if link.rowCount <= 0:
                continue
            logger.debug(
                "AsyncResultFileDownloadManager: adding file link, chunk id %d, start offset %d, row count: %d",
                i,
                link.startRowOffset,
                link.rowCount,
            )
            self._pending_links.append((i, link))

        self.chunk_id += len(links)

        self._downloaded_files: List[DownloadedFile] = []
        self._max_concurrent = max_download_threads
        self._downloadable_result_settings = DownloadableResultSettings(lz4_compressed)
        self._ssl_options = ssl_options
        self.session_id_hex = session_id_hex
        self.statement_id = statement_id
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def _download_with_semaphore(
        self, chunk_id: int, link: TSparkArrowResultLink
    ) -> DownloadedFile:
        """Download a single file with semaphore-based concurrency control."""
        async with self._semaphore:
            handler = AsyncResultSetDownloadHandler(
                settings=self._downloadable_result_settings,
                link=link,
                ssl_options=self._ssl_options,
                chunk_id=chunk_id,
                session_id_hex=self.session_id_hex,
                statement_id=self.statement_id,
            )
            return await handler.run()

    async def download_all(self) -> List[DownloadedFile]:
        """
        Download all pending files concurrently.

        Uses asyncio.gather() with a semaphore to limit concurrent downloads.

        Returns:
            List of downloaded files in order
        """
        if not self._pending_links:
            return []

        # Create semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

        # Create download tasks
        tasks = [
            self._download_with_semaphore(chunk_id, link)
            for chunk_id, link in self._pending_links
        ]

        logger.debug(
            "AsyncResultFileDownloadManager: starting %d downloads with max %d concurrent",
            len(tasks),
            self._max_concurrent,
        )

        # Download all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        downloaded_files = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "AsyncResultFileDownloadManager: download %d failed: %s",
                    i,
                    result,
                )
                raise result
            downloaded_files.append(result)

        # Clear pending links
        self._pending_links = []

        # Sort by start row offset to ensure correct order
        downloaded_files.sort(key=lambda f: f.start_row_offset)

        logger.debug(
            "AsyncResultFileDownloadManager: completed %d downloads",
            len(downloaded_files),
        )

        return downloaded_files

    async def get_next_downloaded_file(
        self, next_row_offset: int
    ) -> Optional[DownloadedFile]:
        """
        Get the next file starting at the given offset.

        Downloads files lazily as needed.

        Args:
            next_row_offset: Expected starting row offset

        Returns:
            Downloaded file or None if no more files
        """
        # If we have pre-downloaded files, return from those
        if self._downloaded_files:
            file = self._downloaded_files.pop(0)
            if (
                next_row_offset < file.start_row_offset
                or next_row_offset > file.start_row_offset + file.row_count
            ):
                logger.debug(
                    "AsyncResultFileDownloadManager: file does not contain row %d, start %d, row count %d",
                    next_row_offset,
                    file.start_row_offset,
                    file.row_count,
                )
            return file

        # Download next batch
        if self._pending_links:
            # Download a batch
            batch_size = min(self._max_concurrent, len(self._pending_links))
            batch_links = self._pending_links[:batch_size]
            self._pending_links = self._pending_links[batch_size:]

            self._semaphore = asyncio.Semaphore(self._max_concurrent)
            tasks = [
                self._download_with_semaphore(chunk_id, link)
                for chunk_id, link in batch_links
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    raise result
                self._downloaded_files.append(result)

            # Sort by start row offset
            self._downloaded_files.sort(key=lambda f: f.start_row_offset)

            if self._downloaded_files:
                return self._downloaded_files.pop(0)

        return None

    def add_link(self, link: TSparkArrowResultLink) -> None:
        """
        Add a link to the download queue.

        Args:
            link: Result link to add
        """
        if link.rowCount <= 0:
            return

        logger.debug(
            "AsyncResultFileDownloadManager: adding file link, start offset %d, row count: %d",
            link.startRowOffset,
            link.rowCount,
        )
        self._pending_links.append((self.chunk_id, link))
        self.chunk_id += 1

    def shutdown(self) -> None:
        """Clean up resources."""
        self._pending_links = []
        self._downloaded_files = []
