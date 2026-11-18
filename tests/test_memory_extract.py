"""Verify exact outbound shape, extraction evidence and explicit provider failures."""

import json
from pathlib import Path

import httpx
import pytest

import memory_extract
from memory_extract import Extracted, ModelConfig, Skipped, extract
from memory_models import MemoryServiceError


@pytest.mark.parametrize("layout", ["legacy", "nested"])
def test_model_call_uses_selected_snippet_and_real_config_fields(model_server, layout: str) -> None:
    path, requests, _ = model_server
    if layout == "nested":
        data = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({"llm": data, "embedding": None}), encoding="utf-8")
    result = extract("Project X uses PostgreSQL.", path)
    assert isinstance(result, Extracted) and result.evidence == "Project X uses PostgreSQL."
    assert len(requests) == 1
    request = requests[0]
    assert request["path"] == "/openai/deployments/fixture-model/chat/completions?api-version=2024-06-01"
    assert request["key"] == "fixture-no-secret"
    assert request["body"]["messages"][-1] == {
        "role": "user", "content": "Project X uses PostgreSQL."
    }
    assert request["body"]["response_format"] == {"type": "json_object"}
    assert "fixture-no-secret" not in repr(ModelConfig.load(path))


def test_smalltalk_is_explicitly_skipped(model_server) -> None:
    path, requests, _ = model_server
    result = extract("Hello!", path)
    assert isinstance(result, Skipped) and result.reason
    assert len(requests) == 1


@pytest.mark.parametrize("body", [
    {}, {"choices": []}, {"choices": [{"message": {"content": None}}]},
    {"choices": [{"message": {"content": "not JSON"}}]},
    {"choices": [{"message": {"content": json.dumps({
        "store": True, "kind": "state", "content": "Invented", "evidence": "not in supplied text",
    })}}]},
    {"choices": [{"message": {"content": '{"store":true,"confirmed":true}'}}]},
])
def test_invalid_or_invented_extraction_is_not_saved(model_server, body) -> None:
    path, requests, control = model_server
    control["body"] = body
    with pytest.raises(MemoryServiceError) as caught:
        extract("A short supplied fact.", path)
    assert caught.value.code == "MODEL_OUTPUT_INVALID"
    assert len(requests) == 1


@pytest.mark.parametrize("status", [302, 401, 429, 500])
def test_http_errors_do_not_echo_provider_content_or_retry(model_server, status: int) -> None:
    path, requests, control = model_server
    control.update(status=status, body={"error": "private-provider-message"})
    with pytest.raises(MemoryServiceError) as caught:
        extract("Selected synthetic text.", path)
    assert caught.value.code == "MODEL_UNAVAILABLE"
    assert caught.value.retryable is (status in (429, 500))
    assert "private-provider-message" not in str(caught.value)
    assert len(requests) == 1


def test_transport_error_is_explicit_and_never_automatically_retried(
    model_server, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise httpx.ReadTimeout("private-network-marker")
    monkeypatch.setattr(memory_extract.httpx, "post", fail)
    with pytest.raises(MemoryServiceError) as caught:
        extract("Selected text.", model_server[0])
    assert caught.value.code == "MODEL_UNAVAILABLE"
    assert "private-network-marker" not in str(caught.value) and len(calls) == 1


def test_config_errors_do_not_echo_key_or_create_resources(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"subscription_key":"private-config-marker"}')
    with pytest.raises(MemoryServiceError) as caught:
        extract("Selected text.", path)
    assert caught.value.code == "CONFIG_ERROR" and "private-config-marker" not in str(caught.value)


@pytest.mark.parametrize("deployment", [None, "", "   "])
def test_missing_embedding_configuration_is_optional(model_server, deployment) -> None:
    path = model_server[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    data["embedding_deployment"] = deployment
    path.write_text(json.dumps(data), encoding="utf-8")
    assert ModelConfig.load(path).embedding is None


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_semantic_threshold_is_a_configuration_error(model_server, threshold) -> None:
    path = model_server[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    data["semantic_similarity_threshold"] = threshold
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(MemoryServiceError) as caught:
        ModelConfig.load(path)
    assert caught.value.code == "CONFIG_ERROR"


@pytest.mark.parametrize("key", [None, ""])
def test_separate_embedding_endpoint_requires_its_own_nonblank_key(model_server, key) -> None:
    path = model_server[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(
        embedding_deployment="fixture-embedding",
        embedding_endpoint="https://another-resource.openai.azure.com",
        embedding_subscription_key=key,
    )
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(MemoryServiceError) as caught:
        ModelConfig.load(path)
    assert caught.value.code == "CONFIG_ERROR"


@pytest.mark.parametrize("field", ["embedding_endpoint", "embedding_api_version"])
def test_legacy_empty_embedding_overrides_are_not_silently_inherited(model_server, field) -> None:
    path = model_server[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(embedding_deployment="fixture-embedding")
    data[field] = ""
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(MemoryServiceError) as caught:
        ModelConfig.load(path)
    assert caught.value.code == "CONFIG_ERROR"
