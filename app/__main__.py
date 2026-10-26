"""Launcher: start the server with the host/port from the config file.

``uvicorn app.main:app`` works too, but then the port only lives on the command
line. This entry point reads ``APP_HOST`` / ``APP_PORT`` from ``.env`` (or the
environment) so the address is configuration, not tribal knowledge:

    uv run python -m app                 # uses APP_HOST / APP_PORT
    uv run python -m app --port 9000     # one-off override
    uv run python -m app --reload        # development
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="启动 DeepSeek Tool Agent（默认读 .env 里的 APP_HOST / APP_PORT）",
    )
    parser.add_argument(
        "--host",
        default=settings.app_host,
        help=f"监听地址（默认 {settings.app_host}）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.app_port,
        help=f"监听端口（默认 {settings.app_port}，也可配置 APP_PORT）",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="改代码自动重启（开发用）",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error(f"端口必须在 1-65535 之间，收到 {args.port}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log_level)

    import uvicorn

    logger.info("starting on http://%s:%d (reload=%s)", args.host, args.port, args.reload)
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
