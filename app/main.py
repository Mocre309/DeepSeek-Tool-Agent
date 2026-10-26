"""FastAPI application factory — wiring, lifecycle, middleware, static UI."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agent.llm import MissingAPIKeyError
from app.agent.service import AgentService
from app.agent.tools import TOOLS
from app.api.routes import router
from app.core import settings_store
from app.core.config import BASE_DIR, Settings, get_settings
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)

STATIC_DIR = BASE_DIR / "app" / "static"

_DESCRIPTION = """
一个小型但**工程完整**的工具调用 Agent：LangChain 组装 ReAct 循环，DeepSeek 做推理，
FastAPI 暴露 SSE 流式接口。

* `POST /api/chat/stream` —— 逐 token 输出，并把每一次工具调用的入参、结果、耗时
  作为独立事件推给前端
* `POST /api/chat` —— 同样的执行路径，聚合后一次性返回（方便脚本与压测）
* `GET /api/tools` —— 工具注册表，前端据此渲染标签
* `DELETE /api/sessions/{id}` —— 清空某个会话的上下文
""".strip()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    factory = app.state.settings_factory
    service = AgentService(factory=factory)
    app.state.agent = service

    logger.info(
        "startup env=%s model=%s mock=%s",
        settings.environment,
        settings.deepseek_model,
        settings.mock_llm,
    )
    try:
        await service.start()  # eager: config errors surface at boot, not mid-demo
        logger.info("agent graph compiled, %d tools registered", len(TOOLS))
    except MissingAPIKeyError as exc:
        logger.error("agent not ready: %s", exc)
    except Exception:
        logger.exception("unexpected failure while compiling the agent graph")

    try:
        yield
    finally:
        await service.shutdown()
        logger.info("shutdown complete")


def create_app(
    settings: Settings | None = None,
    settings_factory: Callable[[], Settings] | None = None,
) -> FastAPI:
    # Migrate older absolute path overrides so the folder can be moved/copied.
    settings_store.compact_paths()
    # The factory is what lets the settings panel hot-reload the agent: an
    # explicit settings object is pinned, the default re-reads overrides each time.
    if settings_factory is None:
        settings_factory = get_settings if settings is None else (lambda pinned=settings: pinned)
    settings = settings or settings_factory()
    setup_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        description=_DESCRIPTION,
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.settings_factory = settings_factory

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        elapsed_ms = (time.perf_counter() - started) * 1000
        suffix = " (stream open)" if request.url.path.endswith("/stream") else ""
        logger.info(
            "%s %s -> %d in %.0fms rid=%s%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
            suffix,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "服务端内部错误，请查看日志。"})

    app.include_router(router)

    if STATIC_DIR.is_dir():
        # Mounted last so /api/* always wins over the catch-all mount.
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    else:  # pragma: no cover - defensive, static UI is part of the repo
        logger.warning("static dir not found at %s", STATIC_DIR)

    return app


app = create_app()
