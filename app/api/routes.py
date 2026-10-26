"""HTTP routes. Thin by design: validate, delegate to the service, serialize."""

from __future__ import annotations

import re
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse

from app.agent.llm import model_supports_images
from app.agent.service import AgentService, PendingApprovalError
from app.agent.tools import describe_tools
from app.agent.tools.files import is_inside_workspace
from app.api.schemas import (
    ApprovalRequest,
    ArchivedGroupInfo,
    ChatRequest,
    ChatResponse,
    GroupCreate,
    GroupInfo,
    GroupRename,
    GroupRestore,
    HealthResponse,
    HistoryResponse,
    PendingApproval,
    PendingResponse,
    RewriteRequest,
    SessionPatch,
    SessionSummary,
    SettingsPatch,
    SettingsView,
    ToolInfo,
    UndoResponse,
)
from app.core import settings_store

router = APIRouter(prefix="/api", tags=["agent"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # Defensive: stops nginx-style proxies from buffering the stream.
    "X-Accel-Buffering": "no",
}

#: Downloaded files are served by name only — no separators, no traversal.
_FILE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def get_service(request: Request) -> AgentService:
    return request.app.state.agent


@router.get("/health", response_model=HealthResponse, summary="健康检查与就绪状态")
async def health(service: AgentService = Depends(get_service)) -> HealthResponse:
    settings = service.settings
    ready = settings.can_use_llm
    return HealthResponse(
        status="ok" if ready else "degraded",
        ready=ready,
        version=settings.app_version,
        model="mock" if settings.mock_llm else settings.deepseek_model,
        mock=settings.mock_llm,
        tool_count=len(describe_tools()),
        active_sessions=service.active_sessions,
        supports_images=settings.mock_llm or model_supports_images(settings.deepseek_model),
        detail=None if ready else "缺少 DEEPSEEK_API_KEY，且未开启 MOCK_LLM=1",
    )


def _settings_view(settings) -> SettingsView:
    payload = settings_store.describe(settings)
    payload["tools"] = [ToolInfo(**item) for item in describe_tools(settings.disabled_tools)]
    return SettingsView(**payload)


@router.get("/settings", response_model=SettingsView, summary="读取当前运行时配置")
async def read_settings(service: AgentService = Depends(get_service)) -> SettingsView:
    return _settings_view(service.settings)


@router.patch("/settings", response_model=SettingsView, summary="修改配置并热重建 Agent")
async def patch_settings(
    payload: SettingsPatch, service: AgentService = Depends(get_service)
) -> SettingsView:
    patch = payload.model_dump(exclude_none=True)
    if not patch:
        return _settings_view(service.settings)
    try:
        settings_store.apply_patch(patch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await service.reload()
    return _settings_view(service.settings)


@router.get("/tools", response_model=list[ToolInfo], summary="当前可用的工具清单")
async def list_tools(service: AgentService = Depends(get_service)) -> list[ToolInfo]:
    return [ToolInfo(**item) for item in describe_tools(service.settings.disabled_tools)]


@router.post("/chat", response_model=ChatResponse, summary="一次性返回完整回答")
async def chat(payload: ChatRequest, service: AgentService = Depends(get_service)) -> ChatResponse:
    result = await service.run(
        session_id=payload.session_id, message=payload.message, images=payload.images
    )
    return ChatResponse(**result)


@router.post(
    "/chat/stream",
    summary="SSE 流式对话",
    response_description="text/event-stream：token / tool_call / tool_result / usage / error / done",
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def chat_stream(
    payload: ChatRequest, service: AgentService = Depends(get_service)
) -> StreamingResponse:
    async def event_source():
        async for event in service.stream(
            session_id=payload.session_id, message=payload.message, images=payload.images
        ):
            yield event.as_sse()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
        status_code=status.HTTP_200_OK,
    )


@router.get(
    "/sessions",
    response_model=list[SessionSummary],
    summary="会话列表（按最近活跃排序）",
)
async def list_sessions(service: AgentService = Depends(get_service)) -> list[SessionSummary]:
    return [SessionSummary(**asdict(record)) for record in service.sessions()]


@router.patch(
    "/sessions/{session_id}",
    response_model=SessionSummary,
    summary="修改会话的标题 / 分组 / 工作区域",
)
async def patch_session(
    session_id: str, payload: SessionPatch, service: AgentService = Depends(get_service)
) -> SessionSummary:
    try:
        if payload.title is not None:
            service.rename_session(session_id, payload.title)
        if payload.group_name is not None:
            service.set_session_group(session_id, payload.group_name)
        if payload.workspace_dir is not None:
            service.set_session_workspace(session_id, payload.workspace_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record = next(
        (item for item in service.sessions() if item.session_id == session_id), None
    )
    if record is None:
        return SessionSummary(
            session_id=session_id,
            title=payload.title or "新对话",
            turns=0,
            created_at=0.0,
            last_active=0.0,
            workspace_dir=payload.workspace_dir or "",
            group_name=payload.group_name or "",
        )
    return SessionSummary(**asdict(record))


@router.get("/groups", response_model=list[GroupInfo], summary="对话分组列表")
async def list_groups(service: AgentService = Depends(get_service)) -> list[GroupInfo]:
    return [GroupInfo(name=group.name, count=group.count) for group in service.groups()]


@router.post(
    "/groups",
    response_model=GroupInfo,
    status_code=status.HTTP_201_CREATED,
    summary="新建一个空分组",
)
async def create_group(
    payload: GroupCreate, service: AgentService = Depends(get_service)
) -> GroupInfo:
    try:
        group = service.create_group(payload.name)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GroupInfo(name=group.name, count=group.count)


@router.get(
    "/groups/trash",
    response_model=list[ArchivedGroupInfo],
    summary="已删除但可恢复的分组",
)
async def list_archived_groups(
    service: AgentService = Depends(get_service),
) -> list[ArchivedGroupInfo]:
    return [
        ArchivedGroupInfo(name=item.name, count=item.count, deleted_at=item.deleted_at)
        for item in service.archived_groups()
    ]


@router.post(
    "/groups/restore",
    response_model=GroupInfo,
    summary="恢复一个被删除的分组（连同当时在组里的对话）",
)
async def restore_group(
    payload: GroupRestore, service: AgentService = Depends(get_service)
) -> GroupInfo:
    try:
        group = service.restore_group(payload.name)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return GroupInfo(name=group.name, count=group.count)


@router.delete(
    "/groups/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除分组（会存进回收站，可恢复）",
)
async def delete_group(name: str, service: AgentService = Depends(get_service)) -> None:
    service.delete_group(name)


@router.patch(
    "/groups/{name}",
    response_model=GroupInfo,
    summary="重命名分组（组内对话一起改名）",
)
async def rename_group(
    name: str, payload: GroupRename, service: AgentService = Depends(get_service)
) -> GroupInfo:
    try:
        group = service.rename_group(name, payload.name)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return GroupInfo(name=group.name, count=group.count)


@router.post(
    "/sessions/{session_id}/undo",
    response_model=UndoResponse,
    summary="撤回最后一轮对话（连同工具调用一起）",
)
async def undo_turn(session_id: str, service: AgentService = Depends(get_service)) -> UndoResponse:
    try:
        removed = await service.undo(session_id)
    except PendingApprovalError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return UndoResponse(
        session_id=session_id,
        removed=removed,
        messages=await service.history(session_id),
    )


@router.post(
    "/sessions/{session_id}/rewrite",
    summary="重写上一轮：撤回后按新内容重新回答（SSE）",
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def rewrite_turn(
    session_id: str,
    payload: RewriteRequest,
    service: AgentService = Depends(get_service),
) -> StreamingResponse:
    async def event_source():
        async for event in service.rewrite(
            session_id=session_id, message=payload.message, images=payload.images
        ):
            yield event.as_sse()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
        status_code=status.HTTP_200_OK,
    )


@router.get(
    "/sessions/{session_id}/messages",
    response_model=HistoryResponse,
    summary="读取某个会话的历史消息",
)
async def session_messages(
    session_id: str,
    include_tools: bool = False,
    service: AgentService = Depends(get_service),
) -> HistoryResponse:
    messages = await service.history(session_id, include_tools=include_tools)
    return HistoryResponse(session_id=session_id, messages=messages)


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除会话（消息 + 列表项）",
)
async def reset_session(session_id: str, service: AgentService = Depends(get_service)) -> None:
    await service.reset(session_id)


@router.get(
    "/sessions/{session_id}/pending",
    response_model=PendingResponse,
    summary="查看当前挂起的待批准操作",
)
async def session_pending(
    session_id: str, service: AgentService = Depends(get_service)
) -> PendingResponse:
    pending = await service.pending_approval(session_id)
    return PendingResponse(
        session_id=session_id,
        pending=PendingApproval(**pending) if pending else None,
    )


@router.post(
    "/sessions/{session_id}/resume",
    summary="批准 / 拒绝挂起的工具调用，并继续该轮对话（SSE）",
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def resume_session(
    session_id: str,
    payload: ApprovalRequest,
    service: AgentService = Depends(get_service),
) -> StreamingResponse:
    async def event_source():
        async for event in service.resume(
            session_id=session_id, approved=payload.approved, reason=payload.reason
        ):
            yield event.as_sse()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
        status_code=status.HTTP_200_OK,
    )


@router.get(
    "/files/{name}",
    summary="取回 Agent 下载的文件",
    response_class=FileResponse,
)
async def get_downloaded_file(
    name: str, service: AgentService = Depends(get_service)
) -> FileResponse:
    if not _FILE_NAME_RE.fullmatch(name):
        raise HTTPException(status_code=400, detail="文件名不合法")
    root = service.settings.downloads_dir.resolve()
    path = (root / name).resolve()
    # Defence in depth: the regex already forbids separators.
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path, filename=name)


@router.get(
    "/workspace/{path:path}",
    summary="读取 / 下载工作区域里的文件",
    response_class=FileResponse,
)
async def get_workspace_file(
    path: str, service: AgentService = Depends(get_service)
) -> FileResponse:
    root = service.settings.workspace_dir.resolve()
    target = (root / path).resolve()
    # Same containment rule the file tools use — one sandbox, one check.
    if not is_inside_workspace(root, target) or not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(target, filename=target.name)
