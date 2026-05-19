# CodeReviewBot

An AI agent that reviews GitHub PRs automatically using AST parsing, vector memory, and GPT-4o. Hooks into GitHub Actions, parses the AST diff, retrieves similar past reviews from a vector store, and posts line-by-line comments flagging bugs, style violations, and security issues.

## Architecture

```
GitHub PR opened
       │
       ▼
POST /webhook/github          ← HMAC-validated
       │
       ▼
diff_extractor.py             ← parse raw patch → ChangedFile objects
       │
       ▼
ast_parser.py                 ← tree-sitter → identify changed functions
       │
       ├── vector_store.py    ← query ChromaDB for similar past reviews
       │
       ▼
reviewer.py (LangChain)       ← GPT-4o generates line comments + summary
       │
       ├── github_client.py   ← post review comments to PR
       ├── slack_client.py    ← notify team channel
       └── vector_store.py    ← store new review in memory
```

## Quick Start

```bash
# 1. Clone and install
git clone <repo>
cd codereviewbot
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your API keys

# 3. Run
make run                  # http://localhost:8000

# 4. Test locally (in another terminal)
make mock-webhook         # sends mock PR payload to local server

# 5. Dashboard
make dashboard            # http://localhost:8501
```

## Docker

```bash
make docker-build
make docker-up            # API on :8000, dashboard on :8501
make docker-logs
make docker-down
```

## GitHub Webhook Setup

1. Go to your repo → Settings → Webhooks → Add webhook
2. Payload URL: `https://your-domain/webhook/github`
3. Content type: `application/json`
4. Secret: same value as `GITHUB_WEBHOOK_SECRET` in `.env`
5. Events: select **Pull requests**

## Project Structure

```
codereviewbot/
├── main.py                  # FastAPI app (router mounting only)
├── config.py                # All env vars via pydantic-settings
├── webhook/
│   ├── handler.py           # POST /webhook/github
│   └── validator.py         # HMAC-SHA256 signature check
├── parser/
│   ├── ast_parser.py        # tree-sitter function extraction
│   └── diff_extractor.py    # unified diff → ChangedFile dataclasses
├── memory/
│   ├── vector_store.py      # ChromaDB read/write
│   └── embedder.py          # OpenAI embeddings wrapper
├── agent/
│   ├── reviewer.py          # LangChain orchestration
│   └── prompts.py           # Prompt templates
├── integrations/
│   ├── github_client.py     # PyGithub wrapper
│   └── slack_client.py      # Slack webhook sender
├── dashboard/
│   └── app.py               # Streamlit metrics UI
├── tests/
│   ├── conftest.py
│   ├── test_parser.py
│   ├── test_agent.py
│   └── fixtures/
│       └── mock_pr_payload.json
└── scripts/
    └── test_local.py        # Send mock webhook for local dev
```

## Testing

```bash
make test           # full suite with coverage
make test-fast      # stop on first failure
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `GITHUB_TOKEN` | Yes | PAT with `repo` + `pull_requests` scopes |
| `GITHUB_WEBHOOK_SECRET` | Yes | Secret configured in GitHub webhook |
| `SLACK_WEBHOOK_URL` | No | Incoming webhook URL (leave blank to disable) |
| `CHROMA_PERSIST_DIR` | No | ChromaDB data directory (default: `./.chroma`) |
| `APP_ENV` | No | `development` or `production` |
| `OPENAI_MODEL` | No | Default: `gpt-4o` |
| `EMBEDDING_MODEL` | No | Default: `text-embedding-3-small` |
