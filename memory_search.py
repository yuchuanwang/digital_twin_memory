"""Optional embedding recall fused with literal matches; no persistent vector index."""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Annotated, Sequence
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field

from memory_extract import ModelConfig
from memory_models import MemoryRecord, MemoryServiceError


_BATCH_SIZE = 32
_CACHE_LIMIT = 1024
_RRF_OFFSET = 60
Vector = tuple[float, ...]
CacheKey = tuple[str, str, str, str]


class _Embedding(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    index: Annotated[int, Field(ge=0)]
    embedding: Annotated[
        list[Annotated[float, Field(allow_inf_nan=False)]], Field(min_length=1, max_length=8192)
    ]


class _EmbeddingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    data: list[_Embedding]


def _embed(texts: list[str], config: ModelConfig) -> list[Vector]:
    connection = config.embedding
    if connection is None:
        raise MemoryServiceError("CONFIG_ERROR", "Hybrid recall requires an embedding configuration.")
    url = (
        f"{connection.endpoint}/openai/deployments/"
        f"{quote(connection.deployment, safe='')}/embeddings"
    )
    try:
        response = httpx.post(
            url, params={"api-version": connection.api_version},
            headers={"api-key": connection.subscription_key.get_secret_value()},
            json={"input": texts, "encoding_format": "float"},
            timeout=30.0, follow_redirects=False,
        )
    except httpx.RequestError as error:
        raise MemoryServiceError(
            "MODEL_UNAVAILABLE", "Embedding request failed.", retryable=True
        ) from error
    if not response.is_success:
        raise MemoryServiceError(
            "MODEL_UNAVAILABLE", f"Embedding model returned HTTP {response.status_code}.",
            retryable=response.status_code == 429 or response.status_code >= 500,
        )
    try:
        data = _EmbeddingResponse.model_validate(response.json()).data
        if len(data) != len(texts) or {item.index for item in data} != set(range(len(texts))):
            raise ValueError("Embedding response indices do not match inputs.")
        vectors = []
        for item in sorted(data, key=lambda item: item.index):
            norm = math.hypot(*item.embedding)
            if norm == 0 or not math.isfinite(norm):
                raise ValueError("Embedding vector must have a finite nonzero norm.")
            vectors.append(tuple(value / norm for value in item.embedding))
        if len({len(vector) for vector in vectors}) != 1:
            raise ValueError("Embedding dimensions differ.")
        return vectors
    except (ValueError, TypeError) as error:
        raise MemoryServiceError(
            "MODEL_OUTPUT_INVALID", "Embedding model returned invalid vectors or indices."
        ) from error


class MemorySearch:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self._cache: OrderedDict[CacheKey, Vector] = OrderedDict()
        self._lock = Lock()

    def configuration(self) -> ModelConfig | None:
        try:
            self.config_path.stat()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise MemoryServiceError("CONFIG_ERROR", "Cannot inspect model configuration.") from error
        config = ModelConfig.load(self.config_path)
        return config if config.embedding is not None else None

    def _vectors(self, texts: list[str], config: ModelConfig) -> dict[str, Vector]:
        connection = config.embedding
        if connection is None:
            raise MemoryServiceError("CONFIG_ERROR", "Hybrid recall requires an embedding configuration.")
        unique = list(dict.fromkeys(texts))
        prefix = (
            connection.endpoint, connection.deployment, connection.api_version,
        )
        result: dict[str, Vector] = {}
        with self._lock:
            for text in unique:
                key = (*prefix, text)
                if key in self._cache:
                    result[text] = self._cache[key]
                    self._cache.move_to_end(key)
        missing = [text for text in unique if text not in result]
        for offset in range(0, len(missing), _BATCH_SIZE):
            batch = missing[offset:offset + _BATCH_SIZE]
            vectors = _embed(batch, config)
            with self._lock:
                for text, vector in zip(batch, vectors):
                    result[text] = vector
                    key = (*prefix, text)
                    self._cache[key] = vector
                    self._cache.move_to_end(key)
                while len(self._cache) > _CACHE_LIMIT:
                    self._cache.popitem(last=False)
        if len({len(vector) for vector in result.values()}) != 1:
            raise MemoryServiceError(
                "MODEL_OUTPUT_INVALID",
                "Embedding dimensions changed; use a consistent deployment and restart the service.",
            )
        return result

    def rank(
        self, query: str, records: Sequence[MemoryRecord], config: ModelConfig,
    ) -> list[MemoryRecord]:
        if not records:
            return []
        vectors = self._vectors([query, *(record.content for record in records)], config)
        query_vector = vectors[query]
        similarities = [
            max(-1.0, min(1.0, math.fsum(
                left * right for left, right in zip(query_vector, vectors[record.content])
            )))
            for record in records
        ]
        semantic = sorted(
            (index for index, value in enumerate(similarities)
             if value >= config.semantic_similarity_threshold),
            key=lambda index: -similarities[index],
        )
        keyword = [
            index for index, record in enumerate(records)
            if query.casefold() in record.content.casefold()
        ]
        scores: dict[int, float] = {}
        for ranking in (semantic, keyword):
            for rank, index in enumerate(ranking, 1):
                scores[index] = scores.get(index, 0.0) + 1.0 / (_RRF_OFFSET + rank)
        return [records[index] for index in sorted(scores, key=lambda index: (-scores[index], index))]
