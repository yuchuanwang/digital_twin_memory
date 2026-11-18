"""Literal and hybrid recall, state/event visibility and scope semantics."""

import json
import math
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from memory_core import MemoryStore
import memory_search
from memory_models import IngestRequest, ListRequest, MemoryServiceError, RecallRequest


@pytest.fixture
def hybrid_store(store: MemoryStore, model_server) -> MemoryStore:
    config_path = model_server[0]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["embedding_deployment"] = "fixture-embedding"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    instance = MemoryStore(store.config.model_copy(update={"model_config_path": config_path}))
    instance.initialize()
    return instance


def test_hybrid_combines_semantic_and_keyword_hits_before_pagination(
    hybrid_store: MemoryStore, model_server,
) -> None:
    query = "preferred programming language"
    texts = [
        "Examples default to Python.", f"{query}: ticket Z9",
        f"The {query} is Python.", "The meeting is on Friday.",
    ]
    model_server[2]["vectors"] = dict(zip([query, *texts], [
        [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.8, 0.6], [-1.0, 0.0],
    ]))
    records = [
        hybrid_store.ingest(IngestRequest(
            target="project", text=text, source="fixture://hybrid",
        )).memory
        for text in texts
    ]
    pages = [
        hybrid_store.recall_page(RecallRequest(query=query, limit=1, offset=offset))
        for offset in range(3)
    ]
    assert [page.data[0] for page in pages] == [records[2], records[0], records[1]]
    assert [page.pagination.has_more for page in pages] == [True, True, False]
    assert pages[-1].pagination.next_offset is None
    assert hybrid_store.context()["retrieval"] == "hybrid"
    assert len(model_server[1]) == 1
    request = model_server[1][0]
    assert request["path"] == "/openai/deployments/fixture-embedding/embeddings?api-version=2024-06-01"
    assert request["key"] == "fixture-no-secret"
    assert request["body"]["encoding_format"] == "float"
    assert set(request["body"]["input"]) == {query, *texts}
    assert hybrid_store.recall(RecallRequest(query=query, search_mode="keyword")) == [
        records[2], records[1],
    ]
    assert len(model_server[1]) == 1
    assert records[0] is not None
    assert hybrid_store.recall(RecallRequest(memory_id=records[0].memory_id)) == [records[0]]
    assert len(model_server[1]) == 1


def test_hybrid_filters_scope_status_and_current_versions_before_embedding(
    hybrid_store: MemoryStore, model_server,
) -> None:
    def save(text: str, **kwargs):
        record = hybrid_store.ingest(IngestRequest(
            target="project", text=text, source="fixture://hybrid-filter", **kwargs,
        )).memory
        assert record is not None
        return record

    old = save("Examples use Python.")
    current = save("Examples now use Rust.", memory_id=old.memory_id, expected_version=1)
    event = save("Decision: used Python for the migration.")
    hidden = save("Rejected preference.")
    with hybrid_store.connection(write=True) as db:
        db.execute("UPDATE memories SET status = 'rejected' WHERE memory_id = ?", (hidden.memory_id,))
    user = hybrid_store.ingest(IngestRequest(
        target="user", text="User-level language preference.", source="fixture://user",
    )).memory
    assert user is not None
    other = MemoryStore(hybrid_store.config.model_copy(update={"project_id": "other"}))
    other.initialize()
    other_record = other.ingest(IngestRequest(
        target="project", text="Another project's preference.", source="fixture://other",
    )).memory
    assert other_record is not None
    query = "favorite coding language"
    model_server[2]["vectors"] = {
        text: [1.0, 0.0] for text in (
            query, old.content, current.content, event.content, hidden.content,
            user.content, other_record.content,
        )
    }
    assert hybrid_store.recall(RecallRequest(query=query, scope="project")) == [current]
    assert set(model_server[1][0]["body"]["input"]) == {query, current.content}
    history = hybrid_store.recall(RecallRequest(query=query, scope="project", view="history"))
    assert {record.reference for record in history} == {
        old.reference, current.reference, event.reference,
    }
    sent = {text for request in model_server[1] for text in request["body"]["input"]}
    assert not sent.intersection({hidden.content, user.content, other_record.content})
    empty = MemoryStore(hybrid_store.config.model_copy(update={"project_id": "empty-project"}))
    empty.initialize()
    count = len(model_server[1])
    assert empty.recall(RecallRequest(query=query, scope="project")) == []
    assert len(model_server[1]) == count


def test_unconfigured_embeddings_keep_keyword_recall_but_reject_explicit_hybrid(
    store: MemoryStore,
) -> None:
    record = store.ingest(IngestRequest(
        target="project", text="Python examples", source="fixture://keyword",
    )).memory
    assert store.context()["retrieval"] == "literal_substring"
    assert store.recall(RecallRequest(query="Python")) == [record]
    with pytest.raises(MemoryServiceError) as caught:
        store.recall(RecallRequest(query="coding language", search_mode="hybrid"))
    assert caught.value.code == "CONFIG_ERROR"


def test_hybrid_similarity_threshold_is_inclusive_and_config_changes_take_effect(
    hybrid_store: MemoryStore, model_server,
) -> None:
    config_path = model_server[0]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["semantic_similarity_threshold"] = 0.5
    config_path.write_text(json.dumps(config), encoding="utf-8")
    query = "language for examples"
    first = hybrid_store.ingest(IngestRequest(
        target="project", text="Python preference", source="fixture://threshold",
    )).memory
    hybrid_store.ingest(IngestRequest(
        target="project", text="Unrelated note", source="fixture://threshold",
    ))
    model_server[2]["vectors"] = {
        query: [1.0, 0.0], "Python preference": [0.5, math.sqrt(0.75)],
        "Unrelated note": [0.49, math.sqrt(1 - 0.49 ** 2)],
    }
    assert hybrid_store.recall(RecallRequest(query=query)) == [first]
    config["semantic_similarity_threshold"] = 0.6
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert hybrid_store.recall(RecallRequest(query=query)) == []
    assert len(model_server[1]) == 1
    config["embedding_deployment"] = None
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert hybrid_store.context()["retrieval"] == "literal_substring"
    assert hybrid_store.recall(RecallRequest(query="Python")) == [first]
    assert len(model_server[1]) == 1


def test_hybrid_batches_all_filtered_candidates_outside_database_transactions(
    hybrid_store: MemoryStore, model_server, monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        hybrid_store.ingest(IngestRequest(
            target="project", text=f"Preference {index}", source="fixture://batch",
        )).memory
        for index in range(35)
    ]
    query = "coding style"
    model_server[2]["vectors"] = {
        text: [1.0, 0.0] for text in [query, *(f"Preference {index}" for index in range(35))]
    }
    active_transactions = []
    original_connection = hybrid_store.connection
    original_post = memory_search.httpx.post

    @contextmanager
    def connection(*, write=False):
        with original_connection(write=write) as db:
            active_transactions.append(db)
            try:
                yield db
            finally:
                active_transactions.remove(db)

    def post(*args, **kwargs):
        assert not active_transactions
        return original_post(*args, **kwargs)

    monkeypatch.setattr(hybrid_store, "connection", connection)
    monkeypatch.setattr(memory_search.httpx, "post", post)
    page = hybrid_store.recall_page(RecallRequest(query=query, offset=30, limit=5))
    assert page.data == list(reversed(records))[30:35]
    assert not page.pagination.has_more
    assert [len(request["body"]["input"]) for request in model_server[1]] == [32, 4]


@pytest.mark.parametrize("status", [401, 429, 500])
def test_configured_embedding_failure_never_falls_back_to_keyword(
    hybrid_store: MemoryStore, model_server, status: int,
) -> None:
    hybrid_store.ingest(IngestRequest(
        target="project", text="Python examples", source="fixture://failure",
    ))
    model_server[2].update(
        embedding_status=status, embedding_body={"private": "provider-secret-marker"},
    )
    with pytest.raises(MemoryServiceError) as caught:
        hybrid_store.recall(RecallRequest(query="Python"))
    assert caught.value.code == "MODEL_UNAVAILABLE"
    assert caught.value.retryable is (status in (429, 500))
    assert "provider-secret-marker" not in str(caught.value)
    assert len(model_server[1]) == 1
    assert len(hybrid_store.recall(RecallRequest(query="Python", search_mode="keyword"))) == 1
    assert len(model_server[1]) == 1


@pytest.mark.parametrize("vectors,indices", [
    ([[1.0, 0.0]], [0]),
    ([[1.0, 0.0], [1.0, 0.0]], [0, 0]),
    ([[0.0, 0.0], [1.0, 0.0]], [0, 1]),
    ([[float("nan"), 1.0], [1.0, 0.0]], [0, 1]),
    ([[True, 1.0], [1.0, 0.0]], [0, 1]),
    ([[1.0], [1.0, 0.0]], [0, 1]),
])
def test_malformed_embeddings_are_explicit_errors(
    hybrid_store: MemoryStore, model_server, vectors, indices,
) -> None:
    hybrid_store.ingest(IngestRequest(
        target="project", text="Python examples", source="fixture://invalid-vector",
    ))
    model_server[2]["embedding_body"] = {"data": [
        {"index": index, "embedding": vector} for index, vector in zip(indices, vectors)
    ]}
    with pytest.raises(MemoryServiceError) as caught:
        hybrid_store.recall(RecallRequest(query="Python"))
    assert caught.value.code == "MODEL_OUTPUT_INVALID"
    assert len(model_server[1]) == 1


def test_embedding_transport_failure_is_redacted_and_not_retried(
    hybrid_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    hybrid_store.ingest(IngestRequest(
        target="project", text="Python examples", source="fixture://timeout",
    ))
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise httpx.ReadTimeout("private-network-marker")

    monkeypatch.setattr(memory_search.httpx, "post", fail)
    with pytest.raises(MemoryServiceError) as caught:
        hybrid_store.recall(RecallRequest(query="Python"))
    assert caught.value.code == "MODEL_UNAVAILABLE"
    assert "private-network-marker" not in str(caught.value) and len(calls) == 1


def test_separate_embedding_connection_does_not_use_chat_credentials(
    hybrid_store: MemoryStore, model_server,
) -> None:
    path, requests, control = model_server
    config = json.loads(path.read_text(encoding="utf-8"))
    deployment = config.pop("embedding_deployment")
    path.write_text(json.dumps({
        "llm": config,
        "embedding": {
            "endpoint": config["endpoint"] + "/embedding-resource",
            "deployment": deployment, "api_version": "2024-12-01-preview",
            "subscription_key": "fixture-embedding-key",
        },
    }), encoding="utf-8")
    record = hybrid_store.ingest(IngestRequest(
        target="project", text="Python examples", source="fixture://separate-resource",
    )).memory
    control["vectors"] = {"coding language": [1.0, 0.0], "Python examples": [1.0, 0.0]}
    assert hybrid_store.recall(RecallRequest(query="coding language")) == [record]
    assert requests[0]["path"] == (
        "/embedding-resource/openai/deployments/fixture-embedding/embeddings"
        "?api-version=2024-12-01-preview"
    )
    assert requests[0]["key"] == "fixture-embedding-key"


def test_default_threshold_recalls_measured_chinese_paraphrase_without_unrelated_note(
    hybrid_store: MemoryStore, model_server,
) -> None:
    query = "惯用编程语言"
    preference = "虚构验收角色的示例代码默认使用 Python。"
    unrelated = "虚构验收角色每周三整理旅行照片。"
    record = hybrid_store.ingest(IngestRequest(
        target="project", text=preference, source="fixture://measured-paraphrase",
    )).memory
    hybrid_store.ingest(IngestRequest(
        target="project", text=unrelated, source="fixture://measured-paraphrase",
    ))
    model_server[2]["vectors"] = {
        query: [1.0, 0.0], preference: [0.4057, math.sqrt(1 - 0.4057 ** 2)],
        unrelated: [0.1745, math.sqrt(1 - 0.1745 ** 2)],
    }
    assert hybrid_store.recall(RecallRequest(query=query)) == [record]
    assert hybrid_store.recall(RecallRequest(query=query, search_mode="keyword")) == []


def test_embedding_response_indices_are_used_even_when_data_is_reordered(
    hybrid_store: MemoryStore, model_server,
) -> None:
    wanted = hybrid_store.ingest(IngestRequest(
        target="project", text="Python examples", source="fixture://unordered",
    )).memory
    hybrid_store.ingest(IngestRequest(
        target="project", text="Travel photographs", source="fixture://unordered",
    ))
    model_server[2]["embedding_body"] = {"data": [
        {"index": 2, "embedding": [1.0, 0.0]},
        {"index": 0, "embedding": [1.0, 0.0]},
        {"index": 1, "embedding": [-1.0, 0.0]},
    ]}
    assert hybrid_store.recall(RecallRequest(query="coding language")) == [wanted]


def test_cached_and_new_embedding_dimensions_cannot_be_mixed(
    hybrid_store: MemoryStore, model_server,
) -> None:
    hybrid_store.ingest(IngestRequest(
        target="project", text="Python examples", source="fixture://changed-dimension",
    ))
    model_server[2]["vectors"] = {
        "first query": [1.0, 0.0], "Python examples": [1.0, 0.0], "second query": [1.0],
    }
    assert len(hybrid_store.recall(RecallRequest(query="first query"))) == 1
    with pytest.raises(MemoryServiceError) as caught:
        hybrid_store.recall(RecallRequest(query="second query"))
    assert caught.value.code == "MODEL_OUTPUT_INVALID"


def test_unreadable_model_configuration_is_not_treated_as_missing(
    model_server, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = model_server[0]
    original_stat = Path.stat

    def stat(path, *args, **kwargs):
        if path == config_path:
            raise PermissionError("private-path-marker")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    with pytest.raises(MemoryServiceError) as caught:
        memory_search.MemorySearch(config_path).configuration()
    assert caught.value.code == "CONFIG_ERROR"
    assert "private-path-marker" not in str(caught.value)


def test_ingestion_current_history_and_events_keep_provenance(store: MemoryStore) -> None:
    first = store.ingest(IngestRequest(
        target="project", text="Database MySQL", source="fixture://first",
    )).memory
    assert first is not None
    assert store.recall(RecallRequest(memory_id=first.memory_id)) == [first]
    event = store.ingest(IngestRequest(
        target="project", text="Decision: migrate to PostgreSQL", source="fixture://decision",
    )).memory
    assert event is not None and event.kind == "event"
    assert store.recall(RecallRequest(memory_id=event.memory_id)) == []
    assert store.recall(RecallRequest(memory_id=event.memory_id, view="history")) == [event]
    replacement = store.ingest(IngestRequest(
        target="project", memory_id=first.memory_id, expected_version=1,
        text="Database PostgreSQL", source="fixture://replacement",
    )).memory
    assert replacement is not None and replacement.version == 2
    assert store.recall(RecallRequest(memory_id=first.memory_id)) == [replacement]
    assert store.recall(RecallRequest(query="MySQL")) == []
    assert store.recall(RecallRequest(query="MySQL", view="history")) == [first]
    assert first.source == "fixture://first" and first.evidence == "Database MySQL"


@pytest.mark.parametrize("text,query", [
    ("项目X用PostgreSQL", "postgresql"), ("并发审查", "并发"),
    ('Literal "quoted" % _', '"quoted" % _'), ("prefix\x00after-NUL", "after-NUL"),
    ("SQL ' OR 1=1 -- is text", "' OR 1=1 --"),
])
def test_literal_queries_treat_operators_as_data(
    store: MemoryStore, text: str, query: str,
) -> None:
    record = store.ingest(IngestRequest(target="project", text=text, source="fixture://literal")).memory
    assert store.recall(RecallRequest(query=query)) == [record]
    assert store.recall(RecallRequest(query="unrelated-no-match")) == []


def test_scope_filtering_precedes_limit(store: MemoryStore, proposal: IngestRequest) -> None:
    saved = store.ingest(proposal).memory
    other = MemoryStore(store.config.model_copy(update={"project_id": "other"}))
    other.initialize()
    for _ in range(3):
        other.ingest(proposal)
    assert store.recall(RecallRequest(query="Database", limit=1)) == [saved]
    user = store.ingest(proposal.model_copy(update={"target": "user"})).memory
    assert store.recall(RecallRequest(query="Database", scope="user")) == [user]
    assert store.recall(RecallRequest(query="Database", scope="project")) == [saved]


def test_history_and_admin_pagination_reach_all_versions_with_stable_ties(
    store: MemoryStore,
) -> None:
    first = store.ingest(IngestRequest(
        target="project", text="History 1", source="fixture://pagination",
    )).memory
    assert first is not None
    for version in range(2, 26):
        store.ingest(IngestRequest(
            target="project", text=f"History {version}", source="fixture://pagination",
            memory_id=first.memory_id, expected_version=version - 1,
        ))
    with store.connection(write=True) as db:
        db.execute("UPDATE memories SET created_at = ?", (first.created_at,))
    for offset, versions, has_more, next_offset in [
        (0, list(range(25, 5, -1)), True, 20),
        (20, list(range(5, 0, -1)), False, None),
        (25, [], False, None),
    ]:
        history = store.recall_page(RecallRequest(
            memory_id=first.memory_id, view="history", limit=20, offset=offset,
        ))
        listing = store.list_page(ListRequest(limit=20, offset=offset))
        assert history == listing
        assert [row.version for row in history.data] == versions
        assert history.pagination.has_more is has_more
        assert history.pagination.next_offset == next_offset
    exact = store.recall_page(RecallRequest(
        memory_id=first.memory_id, view="history", limit=20, offset=5,
    ))
    assert len(exact.data) == 20 and not exact.pagination.has_more
    current = store.recall_page(RecallRequest(memory_id=first.memory_id, offset=1))
    assert current.data == [] and not current.pagination.has_more


def test_pagination_applies_scope_query_and_status_filters_before_offset(
    store: MemoryStore, proposal: IngestRequest,
) -> None:
    expected = [store.ingest(proposal).memory for _ in range(3)]
    other = MemoryStore(store.config.model_copy(update={"project_id": "other"}))
    other.initialize()
    for _ in range(3):
        other.ingest(proposal)
        store.ingest(proposal.model_copy(update={"target": "user"}))
    store.ingest(proposal.model_copy(update={"text": "Unrelated"}))
    pending = store.ingest(proposal).memory
    assert pending is not None
    with store.connection(write=True) as db:
        db.execute("UPDATE memories SET status = 'candidate' WHERE memory_id = ?", (pending.memory_id,))
    page = store.recall_page(RecallRequest(query="Database", scope="project", offset=1, limit=1))
    assert page.data == [expected[1]]
    assert page.pagination.has_more and page.pagination.next_offset == 2
    last = store.recall_page(RecallRequest(query="Database", scope="project", offset=2, limit=1))
    assert last.data == [expected[0]] and not last.pagination.has_more


@pytest.mark.parametrize("arguments", [
    {}, {"query": " "}, {"query": "x", "limit": True},
    {"query": "x", "view": "all"}, {"query": "x", "memory_type": "fact"},
    {"query": "x", "offset": -1}, {"query": "x", "offset": True},
    {"query": "x", "offset": 1.5}, {"query": "x", "offset": 9223372036854775808},
    {"query": "x", "search_mode": "unsupported"},
])
def test_invalid_queries_fail_explicitly(arguments: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RecallRequest.model_validate(arguments)
