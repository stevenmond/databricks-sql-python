"""
Tests for the async SEA (Statement Execution API) backend implementation.

This module contains tests for the AsyncSeaDatabricksClient class.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock, AsyncMock

# Skip all tests if aiohttp is not installed
pytest.importorskip("aiohttp")

from databricks.sql.backend.sea.async_backend import (
    AsyncSeaDatabricksClient,
    _filter_session_configuration,
)
from databricks.sql.backend.types import SessionId, CommandId, CommandState, BackendType
from databricks.sql.types import SSLOptions
from databricks.sql.auth.authenticators import AuthProvider
from databricks.sql.exc import ServerOperationError, DatabaseError


class TestAsyncSeaBackend:
    """Test suite for the AsyncSeaDatabricksClient class."""

    @pytest.fixture
    def mock_http_client(self):
        """Create a mock async HTTP client."""
        with patch(
            "databricks.sql.backend.sea.async_backend.AsyncSeaHttpClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client._make_request = AsyncMock()
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def sea_client(self, mock_http_client):
        """Create an AsyncSeaDatabricksClient instance with mocked dependencies."""
        server_hostname = "test-server.databricks.com"
        port = 443
        http_path = "/sql/warehouses/abc123"
        http_headers = [("header1", "value1"), ("header2", "value2")]
        auth_provider = AuthProvider()
        ssl_options = SSLOptions()

        client = AsyncSeaDatabricksClient(
            server_hostname=server_hostname,
            port=port,
            http_path=http_path,
            http_headers=http_headers,
            auth_provider=auth_provider,
            ssl_options=ssl_options,
            use_cloud_fetch=False,
        )

        return client

    @pytest.fixture
    def sea_session_id(self):
        """Create a SEA session ID."""
        return SessionId.from_sea_session_id("test-session-123")

    @pytest.fixture
    def sea_command_id(self):
        """Create a SEA command ID."""
        return CommandId.from_sea_statement_id("test-statement-123")

    @pytest.fixture
    def mock_cursor(self):
        """Create a mock cursor."""
        cursor = Mock()
        cursor.active_command_id = None
        cursor.buffer_size_bytes = 1000
        cursor.arraysize = 100
        cursor.connection = Mock()
        return cursor

    def test_extract_warehouse_id(self, sea_client):
        """Test warehouse ID extraction from http_path."""
        assert sea_client.warehouse_id == "abc123"

    def test_extract_warehouse_id_with_endpoints(self, mock_http_client):
        """Test warehouse ID extraction from endpoints path."""
        client = AsyncSeaDatabricksClient(
            server_hostname="test-server.databricks.com",
            port=443,
            http_path="/sql/endpoints/xyz789",
            http_headers=[],
            auth_provider=AuthProvider(),
            ssl_options=SSLOptions(),
        )
        assert client.warehouse_id == "xyz789"

    def test_extract_warehouse_id_invalid_path(self, mock_http_client):
        """Test that invalid http_path raises ValueError."""
        with pytest.raises(ValueError, match="Could not extract warehouse ID"):
            AsyncSeaDatabricksClient(
                server_hostname="test-server.databricks.com",
                port=443,
                http_path="/invalid/path",
                http_headers=[],
                auth_provider=AuthProvider(),
                ssl_options=SSLOptions(),
            )

    @pytest.mark.asyncio
    async def test_open_session(self, sea_client, mock_http_client):
        """Test opening a session."""
        mock_http_client._make_request.return_value = {
            "session_id": "test-session-123"
        }

        session_id = await sea_client.open_session(
            session_configuration={"key": "value"},
            catalog="test_catalog",
            schema="test_schema",
        )

        assert session_id.backend_type == BackendType.SEA
        assert session_id.to_sea_session_id() == "test-session-123"
        mock_http_client._make_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_open_session_no_session_id(self, sea_client, mock_http_client):
        """Test that open_session raises error when no session ID returned."""
        mock_http_client._make_request.return_value = {}

        with pytest.raises(ServerOperationError, match="No session ID returned"):
            await sea_client.open_session(
                session_configuration=None,
                catalog=None,
                schema=None,
            )

    @pytest.mark.asyncio
    async def test_close_session(self, sea_client, mock_http_client, sea_session_id):
        """Test closing a session."""
        mock_http_client._make_request.return_value = {}

        await sea_client.close_session(sea_session_id)

        mock_http_client._make_request.assert_called_once()
        call_args = mock_http_client._make_request.call_args
        assert call_args.kwargs["method"] == "DELETE"

    @pytest.mark.asyncio
    async def test_close_session_invalid_backend(self, sea_client, mock_http_client):
        """Test that close_session raises error for non-SEA session ID."""
        # Create a non-SEA session ID (simulated)
        non_sea_session_id = MagicMock()
        non_sea_session_id.backend_type = BackendType.THRIFT

        with pytest.raises(ValueError, match="Not a valid SEA session ID"):
            await sea_client.close_session(non_sea_session_id)

    @pytest.mark.asyncio
    async def test_cancel_command(self, sea_client, mock_http_client, sea_command_id):
        """Test canceling a command."""
        mock_http_client._make_request.return_value = {}

        await sea_client.cancel_command(sea_command_id)

        mock_http_client._make_request.assert_called_once()
        call_args = mock_http_client._make_request.call_args
        assert call_args.kwargs["method"] == "POST"
        assert "/cancel" in call_args.kwargs["path"]

    @pytest.mark.asyncio
    async def test_close_command(self, sea_client, mock_http_client, sea_command_id):
        """Test closing a command."""
        mock_http_client._make_request.return_value = {}

        await sea_client.close_command(sea_command_id)

        mock_http_client._make_request.assert_called_once()
        call_args = mock_http_client._make_request.call_args
        assert call_args.kwargs["method"] == "DELETE"

    @pytest.mark.asyncio
    async def test_get_query_state(self, sea_client, mock_http_client, sea_command_id):
        """Test getting query state."""
        mock_http_client._make_request.return_value = {
            "statement_id": "test-statement-123",
            "status": {"state": "SUCCEEDED"},
            "manifest": {
                "format": "JSON_ARRAY",
                "schema": {"columns": []},
            },
            "result": {"data_array": []},
        }

        state = await sea_client.get_query_state(sea_command_id)

        assert state == CommandState.SUCCEEDED
        mock_http_client._make_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, sea_client, mock_http_client):
        """Test closing the client."""
        await sea_client.close()

        mock_http_client.close.assert_called_once()


class TestFilterSessionConfiguration:
    """Tests for the _filter_session_configuration function."""

    def test_filter_none_configuration(self):
        """Test filtering None configuration returns empty dict."""
        result = _filter_session_configuration(None)
        assert result == {}

    def test_filter_empty_configuration(self):
        """Test filtering empty configuration returns empty dict."""
        result = _filter_session_configuration({})
        assert result == {}

    def test_filter_supported_configuration(self):
        """Test filtering supported configuration parameters."""
        config = {
            "ANSI_MODE": "true",
            "TIMEZONE": "America/Los_Angeles",
        }
        result = _filter_session_configuration(config)

        # Should convert keys to lowercase
        assert "ansi_mode" in result or "timezone" in result
