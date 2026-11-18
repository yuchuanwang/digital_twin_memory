"""Strict single-memory inputs and the one stored record shape; no import-time I/O."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Literal, Mapping, TypeVar

from pydantic import (
    AfterValidator, BaseModel, ConfigDict, Field, StringConstraints,
    ValidationError, computed_field, field_validator, model_validator,
)


Kind = Literal["state", "event"]
Scope = Literal["default", "user", "project"]


class MemoryServiceError(Exception):
    def __init__(
        self, code: str, message: str, *, retryable: bool = False,
        details: dict[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.code, self.retryable, self.details = code, retryable, details or {}


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("Value must not be blank.")
    return value


Text = Annotated[
    str, StringConstraints(min_length=1, max_length=2000), AfterValidator(_nonblank)
]
Identifier = Annotated[
    str, StringConstraints(min_length=1, max_length=64), AfterValidator(_nonblank)
]
Version = Annotated[int, Field(ge=1)]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class EmptyRequest(StrictModel):
    pass


class Content(StrictModel):
    content: Text
    source: Text
    evidence: Text


class IngestRequest(StrictModel):
    target: Literal["user", "project"]
    text: Annotated[
        str, StringConstraints(min_length=1, max_length=8000), AfterValidator(_nonblank)
    ]
    source: Text
    memory_id: Identifier | None = None
    expected_version: Version | None = None

    @model_validator(mode="after")
    def require_revision_baseline(self) -> IngestRequest:
        if (self.memory_id is None) != (self.expected_version is None):
            raise ValueError("Provide both memory_id and expected_version for a revision.")
        return self


class ListRequest(StrictModel):
    scope: Scope = "default"
    limit: Annotated[int, Field(ge=1, le=20)] = 10
    offset: Annotated[int, Field(ge=0, le=9223372036854775807)] = 0


class RecallRequest(ListRequest):
    query: Annotated[
        str, StringConstraints(min_length=1, max_length=500), AfterValidator(_nonblank)
    ] | None = None
    memory_id: Identifier | None = None
    view: Literal["current", "history"] = "current"
    search_mode: Literal["auto", "keyword", "hybrid"] = "auto"

    @model_validator(mode="after")
    def require_selector(self) -> RecallRequest:
        if self.query is None and self.memory_id is None:
            raise ValueError("Provide query or memory_id.")
        return self


class MemoryKey(StrictModel):
    memory_id: Identifier


class MemoryRecord(Content):
    memory_id: Identifier
    version: Version
    base_version: Annotated[int, Field(ge=0)]
    project_id: Identifier | None
    kind: Kind
    status: Literal["extracted", "candidate", "approved", "rejected"] = "extracted"
    created_at: str = Field(default_factory=utc_now)
    reviewed_at: str | None = None

    @computed_field
    @property
    def reference(self) -> str:
        return f"{self.memory_id}@{self.version}"


class IngestResult(StrictModel):
    stored: bool
    memory: MemoryRecord | None = None
    reason: str | None = None


class Pagination(StrictModel):
    offset: int
    limit: int
    has_more: bool

    @computed_field
    @property
    def next_offset(self) -> int | None:
        return self.offset + self.limit if self.has_more else None


class MemoryPage(StrictModel):
    data: list[MemoryRecord]
    pagination: Pagination


class MemoryConfig(StrictModel):
    db_path: Path
    project_id: Identifier | None = None
    model_config_path: Path = Path(__file__).with_name("config.json")

    @field_validator("db_path", "model_config_path", mode="before")
    @classmethod
    def normalize_path(cls, value: object) -> Path:
        if not isinstance(value, (str, Path)) or (
            isinstance(value, str)
            and (not value.strip() or value == ":memory:" or value.startswith("file:"))
        ):
            raise ValueError("Use a persistent database file path.")
        try:
            path = Path(value).expanduser()
            if not path.is_absolute() and (
                PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()
            ):
                raise ValueError("Use a path native to this operating system.")
            return path.absolute()
        except (OSError, RuntimeError) as error:
            raise ValueError("Cannot resolve database path.") from error

    @field_validator("project_id", mode="before")
    @classmethod
    def normalize_project(cls, value: object) -> object:
        return None if value == "" else value

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> MemoryConfig:
        if "MEMORY_USE_LLM" in env and env["MEMORY_USE_LLM"].lower() not in ("true", "1", "yes"):
            raise MemoryServiceError("CONFIG_ERROR", "Automatic ingestion requires the server model.")
        try:
            return cls.model_validate({
                "db_path": env.get("MEMORY_DB_PATH", "~/.dt_memory.db"),
                "project_id": env.get("MEMORY_PROJECT_ID"),
                "model_config_path": env.get(
                    "MEMORY_MODEL_CONFIG", str(Path(__file__).with_name("config.json"))
                ),
            })
        except ValidationError as error:
            raise MemoryServiceError("CONFIG_ERROR", "Invalid memory configuration.") from error


Model = TypeVar("Model", bound=StrictModel)


def validate_request(model: type[Model], value: object) -> Model:
    try:
        return model.model_validate(value)
    except ValidationError as error:
        fields = ", ".join(
            ".".join(map(str, item["loc"])) or "request"
            for item in error.errors(include_input=False)
        )
        raise MemoryServiceError("VALIDATION_ERROR", f"Invalid fields: {fields}.") from error
