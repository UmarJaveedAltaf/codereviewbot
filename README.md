# [CODEREVIEWBOT]

> AI agent that reviews your GitHub PRs like a senior engineer — before your teammates even open Slack.

![Python](https://img.shields.io/badge/Python-3.11-00ff41?style=flat-square&labelColor=0a0a0f)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-00ff41?style=flat-square&labelColor=0a0a0f)
![LangChain](https://img.shields.io/badge/LangChain-0.3-00ff41?style=flat-square&labelColor=0a0a0f)
![Tests](https://img.shields.io/badge/Tests-202%20passing-00ff41?style=flat-square&labelColor=0a0a0f)

---

## The Problem

Code review is the biggest bottleneck in every engineering team. PRs sit unreviewed for hours. Security holes slip through. The same style mistakes get flagged over and over by the same tired reviewer.

**CodeReviewBot fixes this.** It hooks into your GitHub repos via webhook, reads every PR the moment it opens, and posts line-level comments in under 15 seconds — catching bugs, security issues, and style drift before a single human opens the diff.

---

## What it does

```
$ codereviewbot --repo your-org/your-repo --pr 47

› fetching diff...         done (342 lines, 8 files)
› parsing AST...           done (23 functions analysed)
› querying memory...       12 similar patterns found
› running gemini review... done

!! CRITICAL  auth.py:84        — SQL injection via raw query concat
!  WARNING   api/users.py:201  — bare except swallows all errors
   STYLE     models/card.py:55 — missing type hints on 3 params

› posted 7 comments to PR #47 · slack notified
```

---

## How it's different

Most "AI code review" tools just send the raw diff to an LLM and print the output. This one doesn't.

| Feature | What it means |
|---|---|
| **AST parsing** | Understands code *structure* — complexity scores, docstring detection, class nesting — not just text |
| **Vector memory** | Stores every past review in ChromaDB. Retrieves similar findings before prompting the LLM. Gets smarter with every PR |
| **Structured output** | Forces the LLM to return validated Pydantic models — line number, severity, category, suggestion, code fix. No hallucinated line numbers |
| **Feedback loop** | Tracks which suggestions developers accept vs reject. Acceptance rate visible per repo on the dashboard |
| **Team conventions** | Seed a `conventions.yaml` with your team's rules. The bot references them on every review |
| **HMAC validation** | Every webhook payload verified with SHA-256 before processing |

---

## Tech stack

```
FastAPI          — webhook server + REST API
LangChain        — LLM orchestration + structured output
Gemini 1.5 Flash — review model (gemini-embedding-001 for memory)
ChromaDB         — vector store for past reviews + conventions
tree-sitter      — AST parsing for Python + JavaScript
PyGithub         — posting inline PR comments via GitHub API
Slack SDK        — Block Kit review summaries
Streamlit        — metrics dashboard
Docker           — containerised deployment
GitHub Actions   — CI pipeline (lint → test → build)
```

---

## Architecture

```
GitHub PR opened
      │
      ▼
FastAPI webhook ──► HMAC validator
      │
      ▼
Diff extractor ──► tree-sitter AST parser
      │
      ▼
ChromaDB query ──► similar past reviews + team conventions
      │
      ▼
Gemini LLM agent ──► structured JSON output (Pydantic validated)
      │
      ├──► GitHub API  (inline PR comments)
      ├──► Slack       (Block Kit severity summary)
      └──► ChromaDB    (store review for future memory)
```

---

## Quick start

```bash
git clone https://github.com/UmarJaveedAltaf/codereviewbot
cd codereviewbot
cp .env.example .env          # add GOOGLE_API_KEY + GITHUB_TOKEN
docker-compose up -d          # starts API + dashboard + ChromaDB
python scripts/seed_conventions.py   # load team conventions
```

Add a webhook to your GitHub repo:
- Payload URL: `https://your-host:8000/webhook/github`
- Content type: `application/json`
- Secret: value from your `.env`
- Events: Pull requests

Open a PR. The bot does the rest.

Dashboard available at `http://localhost:8501`

---

## Test suite

```bash
pytest tests/ -v
# 202 passed in 6.6s
```

Covers: webhook validation, HMAC security, AST parsing, vector memory, LLM agent (mocked), GitHub + Slack integrations, Streamlit data loaders.

---

## Dashboard

Four pages — Overview, Repository Breakdown, Review History, Convention Effectiveness.

Tracks: total reviews, acceptance rate per repo, issues by category/severity, issues per day, convention trigger frequency.

---

## Project structure

```
codereviewbot/
├── webhook/          # GitHub webhook receiver + HMAC validator
├── parser/           # tree-sitter AST parser + diff extractor
├── memory/           # ChromaDB vector store + Gemini embeddings
├── agent/            # LangChain LLM agent + structured output
├── integrations/     # GitHub API + Slack clients
├── dashboard/        # Streamlit metrics dashboard
├── scripts/          # Convention seeder + local test runner
├── frontend/         # Landing page
└── tests/            # 202 tests across all modules
```

---

## Roadmap

- [ ] GitHub OAuth + multi-tenant SaaS
- [ ] JavaScript/TypeScript full support
- [ ] Auto-approve low-risk PRs
- [ ] Pricing tiers + team management

---

Built by [Umar Javeed Altaf](https://github.com/UmarJaveedAltaf) · 
