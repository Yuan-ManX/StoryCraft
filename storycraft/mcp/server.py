from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from storycraft.config import Settings, load_settings, default_config_path
from storycraft.mcp import register_tools
from storycraft.memory.session_manager import SessionLifecycleManager
from storycraft.utils.logging import get_logger


logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Server Factory
# -----------------------------------------------------------------------------

def create_server(cfg: Settings) -> FastMCP:
    """
    Create and configure MCP server instance.

    This function wires:
      - Server runtime config
      - Session lifecycle manager
      - Tool registration
      - Context injection

    Args:
        cfg: Global Settings object

    Returns:
        Configured FastMCP server instance
    """

    runtime_ctx = cfg

    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[SessionLifecycleManager]:
        """
        Manage session lifecycle during server startup / shutdown.
        """
        logger.info("[MCP] Initializing session lifecycle manager")

        session_manager = SessionLifecycleManager(
            artifacts_root=cfg.project.outputs_dir,
            cache_root=cfg.local_mcp_server.server_cache_dir,
            enable_cleanup=True,
        )

        try:
            yield session_manager
        finally:
            logger.info("[MCP] Cleaning expired sessions")
            session_manager.cleanup_expired_sessions()

    server = FastMCP(
        name=cfg.local_mcp_server.server_name,
        stateless_http=cfg.local_mcp_server.stateless_http,
        json_response=cfg.local_mcp_server.json_response,
        lifespan=lifespan,
    )

    # Register all MCP tools with runtime context
    register_tools.register(server, runtime_ctx)

    return server


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    """
    CLI entrypoint for launching MCP server.
    """
    cfg = load_settings(default_config_path())

    server = create_server(cfg)

    server.settings.host = cfg.local_mcp_server.connect_host
    server.settings.port = cfg.local_mcp_server.port

    logger.info(
        "[MCP] Server starting",
        extra={
            "host": server.settings.host,
            "port": server.settings.port,
            "transport": cfg.local_mcp_server.server_transport,
        },
    )

    server.run(transport=cfg.local_mcp_server.server_transport)


if __name__ == "__main__":
    main()

