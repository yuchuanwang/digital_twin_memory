"""MCP stdio adapter for the memory service; imports do not open a database."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, Generic, Literal, Mapping, TypeVar

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from pydantic import BaseModel, ConfigDict, RootModel

from memory_core import MemoryStore
from memory_models import (
    EmptyRequest, IngestRequest, IngestResult, MemoryConfig, MemoryKey, MemoryRecord,
    MemoryServiceError, Pagination, RecallRequest, StrictModel, validate_request,
)


logger = logging.getLogger("digital_twin_memory")
Payload = TypeVar("Payload")
_EXECUTION_ERRORS = (sqlite3.Error, OSError, RuntimeError, ValueError, TypeError, LookupError)


class ToolSuccess(StrictModel, Generic[Payload]):
    ok: Literal[True] = True
    data: Payload


class RecallSuccess(ToolSuccess[list[MemoryRecord]]):
    pagination: Pagination


class ToolError(StrictModel):
    code: str
    message: str
    retryable: bool
    details: dict[str, int]


class ToolFailure(StrictModel):
    ok: Literal[False] = False
    error: ToolError


class ToolResponse(RootModel[ToolSuccess[Payload] | ToolFailure], Generic[Payload]):
    model_config = ConfigDict(json_schema_extra={"type": "object"})


class RecallResponse(RootModel[RecallSuccess | ToolFailure]):
    model_config = ConfigDict(json_schema_extra={"type": "object"})


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    request: type[StrictModel]
    response: type[BaseModel]
    description: str
    read_only: bool = False

    def as_tool(self, *, hybrid_recall: bool = False) -> types.Tool:
        return types.Tool(
            name=self.name, description=self.description,
            inputSchema=self.request.model_json_schema(),
            outputSchema=self.response.model_json_schema(mode="serialization"),
            annotations=types.ToolAnnotations(
                readOnlyHint=self.read_only, idempotentHint=self.read_only,
                destructiveHint=False,
                openWorldHint=(
                    self.name == "memory_ingest"
                    or (self.name == "memory_recall" and hybrid_recall)
                ),
            ),
        )


TOOL_DEFINITIONS = (
    ToolDefinition(
        "memory_context", EmptyRequest, ToolResponse[dict[str, object]],
        "Inspect the actual project binding, automatic model extraction and storage capabilities.", True,
    ),
    ToolDefinition(
        "memory_recall", RecallRequest, RecallResponse,
        "Read stored records by query or exact memory_id. Queries automatically combine semantic and "
        "keyword recall when embeddings are configured; otherwise they use literal matching. "
        "Use search_mode=keyword for local-only queries. Current is state only; history includes events. "
        "Use pagination.next_offset with the same filters to read more; concurrent writes can shift pages. "
        "Model extraction is not factual verification.", True,
    ),
    ToolDefinition(
        "memory_ingest", IngestRequest, ToolResponse[IngestResult],
        "Automatically send a selected durable fact, preference, explicit decision or completed result. "
        "The server model extracts and immediately stores one unverified memory, or explicitly skips it. "
        "Never send credentials or the full conversation. For updates first recall the ID/version. "
        "Do not blindly retry; there is no deduplication.",
    ),
)


def _result(body: BaseModel, *, is_error: bool = False) -> types.CallToolResult:
    payload = body.model_dump(mode="json")
    return types.CallToolResult(
        content=[types.TextContent(
            type="text", text=json.dumps(payload, ensure_ascii=False, allow_nan=False)
        )],
        structuredContent=payload, isError=is_error,
    )


def _failure(error: MemoryServiceError) -> types.CallToolResult:
    return _result(ToolFailure(error=ToolError(
        code=error.code, message=str(error), retryable=error.retryable, details=error.details,
    )), is_error=True)


def _execute_tool(
    store: MemoryStore, name: str, arguments: Mapping[str, object]
) -> types.CallToolResult:
    if name == "memory_context":
        validate_request(EmptyRequest, arguments)
        return _result(ToolSuccess[dict[str, object]](data=store.context()))
    if name == "memory_ingest":
        return _result(ToolSuccess[IngestResult](
            data=store.ingest(validate_request(IngestRequest, arguments))
        ))
    if name == "memory_recall":
        page = store.recall_page(validate_request(RecallRequest, arguments))
        return _result(RecallSuccess(data=page.data, pagination=page.pagination))
    if name == "memory_forget":
        validate_request(MemoryKey, arguments)
    else:
        raise MemoryServiceError("VALIDATION_ERROR", "Unknown memory tool.")
    raise MemoryServiceError(
        "REVIEW_REQUIRED", "Deletion requires the user's independent admin terminal confirmation."
    )


@asynccontextmanager
async def lifespan(_app: Server[MemoryStore]) -> AsyncGenerator[MemoryStore, None]:
    store = MemoryStore(MemoryConfig.from_env(os.environ))
    store.initialize()
    yield store


def create_server() -> Server[MemoryStore]:
    app: Server[MemoryStore] = Server(
        "digital-twin-memory", version="0.1.0", lifespan=lifespan,
        instructions=(
            "Check memory_context, then recall relevant project knowledge. "
            "Automatically ingest selected durable facts, preferences, explicit decisions and completed results. "
            "The server model extracts and stores without per-record approval, but this is not factual verification. "
            "Do not send secrets or full conversations, blindly retry writes, or bypass deletion confirmation."
        ),
    )

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        hybrid = app.request_context.lifespan_context.context()["retrieval"] == "hybrid"
        return [definition.as_tool(hybrid_recall=hybrid) for definition in TOOL_DEFINITIONS]

    @app.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict[str, object]) -> types.CallToolResult:
        try:
            return await anyio.to_thread.run_sync(
                _execute_tool, app.request_context.lifespan_context, name, arguments
            )
        except MemoryServiceError as error:
            return _failure(error)
        except _EXECUTION_ERRORS as error:
            logger.error("Tool execution failed (%s).", type(error).__name__)
            return _failure(MemoryServiceError("INTERNAL_ERROR", "Tool execution failed; inspect stored state."))

    return app


class SDKDiagnosticFilter(logging.Filter):
    """Malformed protocol frames must not be copied into logs or notifications."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith("mcp."):
            record.msg, record.args = "MCP protocol diagnostic (%s).", (record.levelname,)
            record.exc_info = record.exc_text = record.stack_info = None
        return True


mcp = create_server()


async def run() -> int:
    logging.basicConfig(
        level=logging.WARNING, stream=sys.stderr, format="[memory] %(levelname)s %(message)s"
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(SDKDiagnosticFilter())
    logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.ERROR)
    exit_code = 0
    with anyio.CancelScope() as io_scope:
        async with stdio_server() as (read_stream, write_stream):
            try:
                await mcp.run(read_stream, write_stream, mcp.create_initialization_options())
            except MemoryServiceError as error:
                logger.error("Startup failed [%s]: %s", error.code, error)
                exit_code = 1
                io_scope.cancel()
            except _EXECUTION_ERRORS as error:
                logger.error("Server failed (%s).", type(error).__name__)
                exit_code = 1
                io_scope.cancel()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
