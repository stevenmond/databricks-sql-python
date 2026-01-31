"""
Tests for the async client implementation.

This module contains tests for AsyncConnection, AsyncCursor, and related
async functionality.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock, AsyncMock

# Skip all tests if aiohttp is not installed
pytest.importorskip("aiohttp")

from databricks.sql.async_client import (
    AsyncConnection,
    AsyncCursor,
    AsyncSession,
    async_connect,
)
from databricks.sql.backend.types import SessionId, CommandId, CommandState, BackendType
from databricks.sql.types import SSLOptions
from databricks.sql.exc import InterfaceError, ProgrammingError


class TestAsyncConnection:
    """Test suite for the AsyncConnection class."""

    @pytest.fixture
    def connection_params(self):
        """Return common connection parameters."""
        return {
            "server_hostname": "test-server.databricks.com",
            "http_path": "/sql/warehouses/abc123",
            "access_token": "test-token",
        }

    @pytest.mark.asyncio
    async def test_init(self, connection_params):
        """Test AsyncConnection initialization."""
        conn = AsyncConnection(**connection_params)

        assert conn._server_hostname == "test-server.databricks.com"
        assert conn._http_path == "/sql/warehouses/abc123"
        assert conn._is_open is False
        assert conn.session is None

    @pytest.mark.asyncio
    async def test_open_and_close(self, connection_params):
        """Test opening and closing an async connection."""
        with patch(
            "databricks.sql.async_client.AsyncSeaDatabricksClient"
        ) as mock_backend_class:
            # Setup mock backend
            mock_backend = AsyncMock()
            mock_backend.open_session = AsyncMock(
                return_value=SessionId.from_sea_session_id("test-session-123")
            )
            mock_backend.close_session = AsyncMock()
            mock_backend.close = AsyncMock()
            mock_backend_class.return_value = mock_backend

            with patch(
                "databricks.sql.async_client.get_python_sql_connector_auth_provider"
            ) as mock_auth:
                mock_auth.return_value = MagicMock()

                conn = AsyncConnection(**connection_params)

                # Test open
                await conn.open()
                assert conn._is_open is True
                assert conn.session is not None
                mock_backend.open_session.assert_called_once()

                # Test close
                await conn.close()
                assert conn._is_open is False
                mock_backend.close_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, connection_params):
        """Test AsyncConnection as async context manager."""
        with patch(
            "databricks.sql.async_client.AsyncSeaDatabricksClient"
        ) as mock_backend_class:
            mock_backend = AsyncMock()
            mock_backend.open_session = AsyncMock(
                return_value=SessionId.from_sea_session_id("test-session-123")
            )
            mock_backend.close_session = AsyncMock()
            mock_backend.close = AsyncMock()
            mock_backend_class.return_value = mock_backend

            with patch(
                "databricks.sql.async_client.get_python_sql_connector_auth_provider"
            ) as mock_auth:
                mock_auth.return_value = MagicMock()

                async with AsyncConnection(**connection_params) as conn:
                    assert conn._is_open is True
                    assert conn.session is not None

                # After exiting context, connection should be closed
                assert conn._is_open is False

    @pytest.mark.asyncio
    async def test_cursor_creation(self, connection_params):
        """Test cursor creation from async connection."""
        with patch(
            "databricks.sql.async_client.AsyncSeaDatabricksClient"
        ) as mock_backend_class:
            mock_backend = AsyncMock()
            mock_backend.open_session = AsyncMock(
                return_value=SessionId.from_sea_session_id("test-session-123")
            )
            mock_backend.close_session = AsyncMock()
            mock_backend.close = AsyncMock()
            mock_backend_class.return_value = mock_backend

            with patch(
                "databricks.sql.async_client.get_python_sql_connector_auth_provider"
            ) as mock_auth:
                mock_auth.return_value = MagicMock()

                conn = AsyncConnection(**connection_params)
                await conn.open()

                cursor = conn.cursor()
                assert isinstance(cursor, AsyncCursor)
                assert cursor.connection is conn
                assert cursor.open is True

                await conn.close()

    @pytest.mark.asyncio
    async def test_cursor_from_closed_connection_raises_error(self, connection_params):
        """Test that creating cursor from closed connection raises InterfaceError."""
        conn = AsyncConnection(**connection_params)

        with pytest.raises(InterfaceError):
            conn.cursor()


class TestAsyncCursor:
    """Test suite for the AsyncCursor class."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock async backend."""
        backend = AsyncMock()
        backend.execute_command = AsyncMock()
        backend.get_query_state = AsyncMock(return_value=CommandState.SUCCEEDED)
        backend.get_execution_result = AsyncMock()
        backend.cancel_command = AsyncMock()
        backend.close_command = AsyncMock()
        return backend

    @pytest.fixture
    def mock_connection(self, mock_backend):
        """Create a mock async connection."""
        conn = MagicMock()
        conn._server_hostname = "test-server.databricks.com"
        conn.get_session_id_hex.return_value = "test-session-hex"
        conn.session = MagicMock()
        conn.session.session_id = SessionId.from_sea_session_id("test-session-123")
        conn.session.backend = mock_backend
        conn.lz4_compression = True
        conn.use_cloud_fetch = False
        conn.use_inline_params = False
        conn.open = True
        return conn

    @pytest.fixture
    def cursor(self, mock_connection, mock_backend):
        """Create an AsyncCursor with mocked dependencies."""
        return AsyncCursor(
            connection=mock_connection,
            backend=mock_backend,
            arraysize=100,
            result_buffer_size_bytes=1000,
        )

    @pytest.mark.asyncio
    async def test_init(self, cursor):
        """Test AsyncCursor initialization."""
        assert cursor.open is True
        assert cursor.active_result_set is None
        assert cursor.arraysize == 100
        assert cursor.buffer_size_bytes == 1000

    @pytest.mark.asyncio
    async def test_execute(self, cursor, mock_backend):
        """Test execute method."""
        mock_result_set = MagicMock()
        mock_backend.execute_command.return_value = mock_result_set

        result = await cursor.execute("SELECT 1")

        assert result is cursor
        mock_backend.execute_command.assert_called_once()
        call_args = mock_backend.execute_command.call_args
        assert call_args.kwargs["operation"] == "SELECT 1"
        assert call_args.kwargs["async_op"] is False

    @pytest.mark.asyncio
    async def test_execute_async(self, cursor, mock_backend):
        """Test execute_async method."""
        mock_backend.execute_command.return_value = None

        result = await cursor.execute_async("SELECT 1")

        assert result is cursor
        mock_backend.execute_command.assert_called_once()
        call_args = mock_backend.execute_command.call_args
        assert call_args.kwargs["async_op"] is True

    @pytest.mark.asyncio
    async def test_fetchone(self, cursor, mock_backend):
        """Test fetchone method."""
        mock_result_set = AsyncMock()
        mock_result_set.fetchone = AsyncMock(return_value=("value1", "value2"))
        cursor.active_result_set = mock_result_set

        result = await cursor.fetchone()

        assert result == ("value1", "value2")
        mock_result_set.fetchone.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetchmany(self, cursor, mock_backend):
        """Test fetchmany method."""
        mock_result_set = AsyncMock()
        mock_result_set.fetchmany = AsyncMock(
            return_value=[("v1", "v2"), ("v3", "v4")]
        )
        cursor.active_result_set = mock_result_set

        result = await cursor.fetchmany(2)

        assert len(result) == 2
        mock_result_set.fetchmany.assert_called_once_with(2)

    @pytest.mark.asyncio
    async def test_fetchall(self, cursor, mock_backend):
        """Test fetchall method."""
        mock_result_set = AsyncMock()
        mock_result_set.fetchall = AsyncMock(
            return_value=[("v1",), ("v2",), ("v3",)]
        )
        cursor.active_result_set = mock_result_set

        result = await cursor.fetchall()

        assert len(result) == 3
        mock_result_set.fetchall.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetchone_no_result_set_raises_error(self, cursor):
        """Test fetchone without active result set raises ProgrammingError."""
        cursor.active_result_set = None

        with pytest.raises(ProgrammingError):
            await cursor.fetchone()

    @pytest.mark.asyncio
    async def test_close(self, cursor, mock_backend):
        """Test cursor close method."""
        mock_result_set = AsyncMock()
        mock_result_set.close = AsyncMock()
        cursor.active_result_set = mock_result_set

        await cursor.close()

        assert cursor.open is False
        mock_result_set.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_connection, mock_backend):
        """Test AsyncCursor as async context manager."""
        async with AsyncCursor(
            connection=mock_connection,
            backend=mock_backend,
        ) as cursor:
            assert cursor.open is True

        assert cursor.open is False

    @pytest.mark.asyncio
    async def test_get_query_state(self, cursor, mock_backend):
        """Test get_query_state method."""
        cursor.active_command_id = CommandId.from_sea_statement_id("test-statement")
        mock_backend.get_query_state.return_value = CommandState.RUNNING

        state = await cursor.get_query_state()

        assert state == CommandState.RUNNING
        mock_backend.get_query_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel(self, cursor, mock_backend):
        """Test cancel method."""
        cursor.active_command_id = CommandId.from_sea_statement_id("test-statement")

        await cursor.cancel()

        mock_backend.cancel_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_description_property(self, cursor):
        """Test description property."""
        mock_result_set = MagicMock()
        mock_result_set.description = [("col1", "string", None, None, None, None, None)]
        cursor.active_result_set = mock_result_set

        assert cursor.description == [("col1", "string", None, None, None, None, None)]

    @pytest.mark.asyncio
    async def test_description_no_result_set(self, cursor):
        """Test description property with no active result set."""
        cursor.active_result_set = None
        assert cursor.description is None


class TestAsyncConnect:
    """Test suite for the async_connect function."""

    @pytest.mark.asyncio
    async def test_async_connect(self):
        """Test async_connect function creates and opens connection."""
        with patch(
            "databricks.sql.async_client.AsyncSeaDatabricksClient"
        ) as mock_backend_class:
            mock_backend = AsyncMock()
            mock_backend.open_session = AsyncMock(
                return_value=SessionId.from_sea_session_id("test-session-123")
            )
            mock_backend.close_session = AsyncMock()
            mock_backend.close = AsyncMock()
            mock_backend_class.return_value = mock_backend

            with patch(
                "databricks.sql.async_client.get_python_sql_connector_auth_provider"
            ) as mock_auth:
                mock_auth.return_value = MagicMock()

                conn = await async_connect(
                    server_hostname="test-server.databricks.com",
                    http_path="/sql/warehouses/abc123",
                    access_token="test-token",
                )

                assert conn._is_open is True
                assert conn.session is not None

                await conn.close()


class TestAsyncSession:
    """Test suite for the AsyncSession class."""

    @pytest.mark.asyncio
    async def test_session_open_and_close(self):
        """Test AsyncSession open and close."""
        with patch(
            "databricks.sql.async_client.AsyncSeaDatabricksClient"
        ) as mock_backend_class:
            mock_backend = AsyncMock()
            mock_backend.open_session = AsyncMock(
                return_value=SessionId.from_sea_session_id("test-session-123")
            )
            mock_backend.close_session = AsyncMock()
            mock_backend_class.return_value = mock_backend

            with patch(
                "databricks.sql.async_client.get_python_sql_connector_auth_provider"
            ) as mock_auth:
                mock_auth.return_value = MagicMock()

                session = AsyncSession(
                    server_hostname="test-server.databricks.com",
                    http_path="/sql/warehouses/abc123",
                    access_token="test-token",
                )

                await session.open()
                assert session.is_open is True
                assert session._session_id is not None

                await session.close()
                assert session.is_open is False
