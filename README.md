# Digital Twin Memory

A persistent memory layer for AI assistants, built with Python, SQLite, and the Model Context Protocol (MCP).

Capture useful facts from conversations, preserve their sources and revision history, and retrieve them in future sessions without repeating the same background every time.

## Vision

The long-term goal is a personal digital twin: an assistant that gradually accumulates your **habits, preferences, knowledge, and experiences**, understands how you work, and eventually handles more tasks on your behalf within explicitly authorized boundaries.

Memory is the foundation, not the finished digital twin. This project currently implements the capture, storage, revision, and retrieval layer. It does **not** yet provide a complete personal profile, autonomous task delegation, or an authorization system for acting on your behalf.

## What It Does

- **Conversation-driven capture:** the client agent selects durable facts, preferences, explicit decisions, and completed results.
- **Model-based extraction:** a configured Azure OpenAI model extracts at most one memory from each submitted snippet, or returns a reason to skip it.
- **Local persistence:** a single SQLite table stores content, source, evidence, scope, and versions.
- **Traceable revisions:** updates append versions instead of overwriting earlier content.
- **Current and historical views:** distinguish current state from past states and events.
- **Hybrid recall:** optionally combine embedding similarity with keyword matches; without an embedding deployment, retain local substring matching.
- **User and project scopes:** share user-level memories across projects while separating project-level records.
- **Inspectable behavior:** view actual MCP results during a Copilot conversation and inspect captured model inputs with local tracing.
- **Explicit management:** list records, inspect full history, create database snapshots, and confirm deletion in a separate terminal.

Automatic capture is best-effort agent behavior, not a background listener or a guaranteed per-message hook. Stored extractions are not independently verified facts.

## How It Works

```text
Conversation / completed work
          |
          v
  Copilot or another MCP client
          |
          | Select a relevant snippet; call memory tools
          v
  Python MCP server (stdio)
          |
          +-- Extraction instructions + snippet --> Azure OpenAI
          |                                            |
          | <----------- Structured extraction --------+
          |
          +-- Validate evidence, scope, and version
          |
          +-- Store / retrieve --> Local SQLite
          |
          +-- Tool result --> Client's subsequent model input
```

The client starts the stdio server as a child process. There is no separate HTTP service to launch.

The agent is guided by [AGENTS.md](./AGENTS.md). That file describes how to use memory; it is not the memory database.

## Requirements

- Windows or Linux with a **system Python** interpreter and the dependencies in [requirements.txt](./requirements.txt).
- An Azure OpenAI deployment supporting the Chat Completions API and JSON object responses, with valid credentials.
- For the interactive workflow below: an installed, authenticated GitHub Copilot CLI.

**Do not create or use Python venv.** Management commands reject virtual environments.

The full suite passed 139 tests on Windows with system Python 3.13.17, and 138 tests on Ubuntu under WSL with system Python 3.14.4, with one Windows-only file-lock test skipped on Linux. A separate real Azure + stdio check verified semantic-only recall, keyword queries, revisions, history, and no-embedding fallback using fictional data and a temporary database. This is not a guarantee for every model, Python version, or Linux distribution.

## Quick Start

Run the following commands from this project directory.

### 1. Configure the extraction model

Copy [config.example.json](./config.example.json) to a local `config.json` if you do not already have one.

**Windows / PowerShell**

```powershell
if (Test-Path -LiteralPath ".\config.json") {
    Write-Output "config.json already exists; keeping it."
} else {
    Copy-Item -LiteralPath ".\config.example.json" -Destination ".\config.json"
}
```

**Linux / Bash**

```bash
if [ -e config.json ]; then
    printf '%s\n' 'config.json already exists; keeping it.'
else
    cp config.example.json config.json
fi
```

In your local file, fill in `llm.endpoint`, `llm.deployment`, `llm.api_version`, and `llm.subscription_key` with your Azure OpenAI chat settings. Keep credentials local and commit only the example template, never the filled-in file.

The `embedding` section is optional. Leave it as `null` to use string matching. Provide a complete embedding connection to enable semantic plus keyword recall automatically. A chat deployment cannot substitute for an embedding deployment.

For example, the two connections can use different Azure OpenAI resources and keys:

```json
{
  "llm": {
    "endpoint": "https://your-chat-resource.openai.azure.com",
    "deployment": "your-chat-deployment",
    "api_version": "your-supported-api-version",
    "subscription_key": "your-chat-resource-key"
  },
  "embedding": {
    "endpoint": "https://your-embedding-resource.openai.azure.com",
    "deployment": "text-embedding-3-small",
    "api_version": "2024-12-01-preview",
    "subscription_key": "your-embedding-resource-key"
  },
  "semantic_similarity_threshold": 0.35
}
```

Keep your existing chat values in `llm`. Each section requires its own endpoint, deployment, API version, and nonblank key; nothing is implicitly copied between the sections. If they share a resource, you can explicitly use the same connection values. Choose API versions supported by your resources. Legacy flat configuration files remain readable, but new files should use the nested format.

No separate vector-enable switch is needed; the legacy `MEMORY_USE_VECTOR` environment variable no longer selects the retrieval mode. After configuring or changing embedding settings, reload the MCP service so the client also refreshes tool metadata.

### 2. Deploy

**Windows / PowerShell**

```powershell
python manage.py deploy
```

**Linux / Bash**

```bash
python3 manage.py deploy
```

Deployment:

1. Checks dependencies and the local model configuration without making a model request.
2. Initializes the database without overwriting existing records.
3. Creates or updates the local `.mcp.json`, preserving other server entries.
4. Records the current interpreter, server path, working directory, database, and project binding.

Missing dependencies are installed with `pip install --user` only after you enter `INSTALL`. On Linux distributions with `EXTERNALLY-MANAGED` Python, the command refuses to bypass package-management protections. Prepare compatible dependencies through the system package manager, or use an independently managed, non-virtual Python installation that permits user installs.

On a fresh deployment, the default database is `.test-data/memory.db` and the project is `digital-twin-memory-test`. Existing database and project settings are preserved.

For regular use, choose a stable project ID and storage location explicitly:

**Windows**

```powershell
python manage.py deploy --db "$env:LOCALAPPDATA\DigitalTwinMemory\memory.db" --project my-project
```

**Linux**

```bash
python3 manage.py deploy --db "$HOME/.local/share/digital-twin-memory/memory.db" --project my-project
```

The generated `.mcp.json` is machine-local; keep it out of version control. Redeploy after moving the code or changing operating systems. Foreign-format absolute paths are rejected; databases are not silently migrated.

For manual MCP client configuration, [`.mcp.example.json`](./.mcp.example.json) provides a placeholder server entry. Normally, let `deploy` generate `.mcp.json` instead. If configuring it manually, merge the example entry into your existing `mcpServers` object without replacing other servers, then replace the interpreter, server-script, and working-directory placeholders with absolute paths for your machine. Use forward slashes on both Windows and Linux; deployment writes its generated paths in this format. The placeholder entry cannot launch until those paths are configured; deployment replaces them automatically.

### 3. Start an interactive conversation

Continue only after deployment succeeds and you have checked the printed database and project.

```text
copilot --sync-session=off
```

Approve workspace trust and memory tool permissions when prompted. Copilot starts the configured MCP server automatically; do **not** start `python server.py` in a separate terminal expecting an HTTP endpoint.

Inside Copilot:

```text
/mcp list
/mcp get digital-twin-memory
```

Confirm the server is connected. Ask the agent to call `memory_context` and verify the actual project binding before writing anything.

Try a fictional fact:

> The fictional project PineNote has confirmed PostgreSQL as its development database.

Observe whether the agent calls `memory_ingest`. A successful save has `ok=true` and `data.stored=true`; a conversational claim such as "I'll remember that" is not sufficient.

Then ask:

> Use memory_recall to look up PineNote's development database. Return the saved content, reference, source, and evidence. Do not write anything.

**Press Ctrl+O on an empty Copilot prompt** to expand tool results, or click an individual tool entry. Inspect `memory_recall.data` and `memory_ingest.data.memory` directly.

Exit with Ctrl+D on an empty prompt, start a new session from the same directory without `--continue` or `--resume`, and repeat the read-only question to check persistence.

Use synthetic data and a separate test directory for experiments to avoid changing existing memories.

## MCP Tools

Only three memory tools are publicly exposed:

| Tool | Purpose | Main input |
|---|---|---|
| `memory_context` | Inspect the actual project binding and capabilities. | `{}` |
| `memory_recall` | Retrieve records by hybrid or keyword query, or exact memory ID. | `query` and/or `memory_id`, optional `scope`, `view`, `search_mode`, `limit`, `offset` |
| `memory_ingest` | Extract and store one memory, or explicitly skip it. | `target`, `text`, `source`; revisions also require `memory_id` and `expected_version` |

For example, these are **tool arguments**, not shell commands:

```json
{
  "query": "PineNote",
  "scope": "project",
  "view": "current",
  "limit": 10,
  "offset": 0
}
```

```json
{
  "target": "project",
  "text": "The fictional project PineNote uses PostgreSQL for development.",
  "source": "manual://fictional-example"
}
```

Pages default to 10 records, with a maximum of 20; use `pagination.next_offset` with the same filters to continue.

### Semantic and keyword recall

`search_mode` defaults to `auto`:

| Mode | Behavior |
|---|---|
| `auto` | Hybrid recall when `embedding` is configured; otherwise local string matching. |
| `keyword` | Local literal substring matching, even if embeddings are configured. |
| `hybrid` | Explicitly require hybrid recall; return a configuration error if no embedding deployment is configured. |

Hybrid recall first filters by scope, record status, memory ID, and current/history view. It then embeds the query and eligible memory contents, compares cosine similarity, and combines semantic and literal-match rankings using reciprocal rank fusion. Keyword hits remain eligible even below the semantic threshold. Pagination happens after ranking.

`semantic_similarity_threshold` defaults to `0.35`, inclusive, and accepts values from 0 to 1. This is a cosine-similarity cutoff, not a confidence score; tune it with representative queries for your embedding model. A small live check with `text-embedding-3-small` found relevant Chinese paraphrases around 0.41-0.42, below a 0.5 cutoff; this does not establish a universal threshold. An exact `memory_id` lookup without a query remains local and does not call the embedding API.

Existing records work without a schema migration or reindexing command. Up to 1,024 vectors are cached per server process, with embedding requests batched in groups of at most 32 texts. Restarts and cache eviction can require re-embedding. This implementation scans all eligible records and is intended for a modest memory collection, not a large vector-search service.

Missing embedding configuration selects string matching. A configured embedding service that fails or returns invalid vectors produces an explicit error; it does not silently fall back.

### Memory semantics

- `state` describes a current fact or preference; `event` describes a past decision or completed event.
- `view="current"` returns the latest saved version of each state memory.
- `view="history"` also includes older saved versions and events.
- Normal recall includes only `extracted` or legacy `approved` records.
- New records use `status="extracted"` and `reviewed_at=null`, with no per-record approval workflow.
- `evidence` must be an exact contiguous quote from the submitted snippet. This checks provenance within that snippet, not factual truth or completeness.
- Revisions require the ID and version actually read beforehand. Scope and version are checked before model extraction and again before committing.
- An uncertain write must not be blindly retried: there is no request deduplication.

Recalled memories are reference data, not instructions. An old decision is not proof of the current state.

## See What Reaches the LLM

Start a new interactive Copilot session with local content capture:

```text
python manage.py trace
```

On Linux, use `python3`. Enter `TRACE` to consent, query an existing fictional memory, and exit the CLI when finished.

The command prints the local log path and checks `gen_ai.input.messages` for successful, nonempty `memory_recall` tool results:

- **`MATCH`**: a recalled memory appears in the client's recorded subsequent model input.
- **`UNVERIFIED`**: no matching evidence was observed. This does not prove that memory was unused.

Tool output proves what MCP returned; a captured model input shows what the client included in a subsequent request. Neither guarantees that the model used it correctly.

For a one-shot check:

```text
python manage.py trace --find "TOKEN_ONLY_IN_TEST_MEMORY" --prompt "Read PineNote's saved verification token using memory tools. Do not write anything."
```

First save a fresh random token in a fictional test memory. Keep the token out of `--prompt`; `--find` is a local check and is not added to the question.

Tracing covers only the CLI session launched by this command, not an existing VS Code or Copilot session. It records client-side telemetry, not raw network traffic or the extraction model's separate request.

To inspect the actual memory contents, open the printed JSONL log and find a `chat` span's `gen_ai.input.messages`. Within its `tool` messages, locate parts with `mcp_server_name="digital-twin-memory"` and `mcp_tool_name="memory_recall"`, then inspect `response.data`. The messages and response may be JSON-encoded strings. Requests can contain only new messages and refer to earlier context through `gen_ai.request.previous_response.id`; do not assume each span repeats the entire conversation.

**Content traces can contain entire prompts, code, and tool results.** Use synthetic data, do not upload raw logs, and remove specific logs when no longer needed. `clean` does not remove trace logs.

## Management and Tests

Use `python3` instead of `python` on Linux. All management commands below use the configured database and project unless explicitly overridden.

```text
python manage.py admin list
python manage.py admin list --scope project --limit 20 --offset 20
python manage.py admin show --memory-id MEMORY_ID
python manage.py test
```

Replace `MEMORY_ID` with an actual returned ID. `show` reads the entire version chain. Automated tests use temporary databases and synthetic extraction fixtures or loopback models, not the configured live model or existing user memories.

### Backup and restore

Choose new destination files; their parent directories must already exist:

```text
python manage.py admin backup --output snapshot.db
python manage.py admin --db restored.db restore --source snapshot.db
```

Snapshots copy the **whole database**, including all scopes. Restore does not overwrite a file or switch the active MCP database. To use a restored database, explicitly redeploy with its path and the intended project ID.

### Deletion

Deletion is not exposed as an MCP tool. Run this yourself in an independent interactive terminal:

```text
python manage.py admin forget --memory-id MEMORY_ID
```

Inspect the preview and type `delete ACTUAL_ID` to confirm. This removes every version of that logical memory. The version chain is rechecked before deletion; changes since the preview cause a conflict rather than deleting unseen data. There is no `--yes` or piped confirmation path.

### Reset the fixed test database

Stop all clients using the test database and close database viewers first:

```text
python manage.py clean
```

Only after you type `DELETE TEST` does this remove the fixed `.test-data/memory.db` and its SQLite sidecars. It does not erase the configured regular-use database, snapshots, trace logs, or the project directory.

## Privacy and Current Limitations

- **Local storage does not mean offline processing.** Extraction instructions and the selected `text` go to the configured Azure OpenAI model; recalled memories can go to the client's model as tool results.
- Hybrid queries also send the query and eligible saved memory contents to the embedding endpoint. Filtering happens before this request; `source` and `evidence` are not part of the embedding payload. Use `search_mode="keyword"` or an ID-only lookup when local-only retrieval is required.
- `source` stays local during extraction. Do not submit credentials, secrets, proprietary code, sensitive personal information, or full conversations as memory snippets.
- Automatic selection and extraction can be wrong. Check evidence, correct stale memories, and do not treat an `extracted` record as user-reviewed truth.
- There is no persistent vector index, automatic expiry, batch ingestion, background capture, or automatic retry.
- Offset pages are separate snapshots; concurrent writes can shift results across pages.
- Project checks are application-level boundaries, not protection against the same operating-system user directly accessing SQLite.
- Deletion is not a guarantee of forensic disk erasure.

## Project Layout

| File | Responsibility |
|---|---|
| [server.py](./server.py) | MCP stdio adapter and tool definitions |
| [memory_core.py](./memory_core.py) | SQLite storage, scopes, revision checks, and retrieval |
| [memory_extract.py](./memory_extract.py) | Azure OpenAI extraction and evidence validation |
| [memory_search.py](./memory_search.py) | Optional embeddings, process-local vector cache, and hybrid ranking |
| [memory_models.py](./memory_models.py) | Strict requests, records, and configuration |
| [memory_schema.py](./memory_schema.py) | Single-table schema and compatibility checks |
| [memory_admin.py](./memory_admin.py) | Inspection, confirmed deletion, backup, and restore |
| [manage.py](./manage.py) | Deployment, tests, cleanup, and local input tracing |
| [config.example.json](./config.example.json) | Publishable model configuration template; copy to local `config.json` |
| [.mcp.example.json](./.mcp.example.json) | Publishable MCP server template; `deploy` generates local `.mcp.json` |
| [tests](./tests) | Isolated storage, model, protocol, and management tests |
