# Project TODOs and Enhancement Ideas

A living list of improvements, experiments, and future work for the RAG Chatbot.

## Product & UX

- [ ] Feedback page: add charts (trend over time, rolling satisfaction)
- [ ] Config UI: switch feedback backend (SQLite/JSON) without env vars
- [ ] Model selection UI (per-session): choose Ollama model, temperature, max tokens
- [ ] Conversation management: rename, pin, archive, delete threads
- [ ] Message actions: copy, regenerate, improve with hint
- [ ] Persona switching and persona management
- [x] Sidebar page to list knowledge base files (uploaded/ingested)

## Feedback & Analytics

- [ ] Filter persistent feedback by time range and keyword
- [ ] Tag feedback with custom labels (e.g., hallucination, incomplete)
- [ ] Export formats: CSV and Parquet in addition to JSON
- [ ] Admin-only actions for clearing data; optional confirmation modals

## RAG Quality & Data

- [ ] Document deduplication and versioning
- [ ] Re-index flow for updated or deleted files
- [ ] Chunking strategies: semantic vs. fixed; overlap tuning
- [ ] Embeddings backend switch (e.g., sentence-transformers variants)
- [ ] Source attributions: show top-k contexts with citations
- [ ] Per-file relevance heatmap to spot weak coverage

## Performance & Reliability

- [ ] Response caching keyed by normalized query + index version
- [ ] Rate limiting and basic abuse prevention
- [ ] Background pre-embedding for large uploads
- [ ] Streaming token display with graceful cancellation

## Architecture & Config

- [ ] Pluggable storage interfaces (feedback, documents, sessions)
- [ ] Centralized settings (dotenv + in-app overrides + profile presets)
- [ ] CLI utilities for maintenance tasks (export, migrate, reindex)

## Testing & Tooling

- [ ] Unit tests for `feedback_manager` (both SQLite and JSON backends)
- [ ] Integration tests for upload → index → retrieve → answer
- [ ] Lint/format hooks and CI
- [ ] Basic load test of retrieval and response latencies

## Security & Privacy

- [ ] Optional user authentication and per-user feedback partitioning
- [ ] PII scrubbing/redaction in logs and stored contexts
- [ ] Data retention policy and configurable purges

## Deployment

- [ ] Dockerfile and compose for Ollama + app + Chroma
- [ ] Prod settings template; health checks and logs
- [ ] Simple monitoring dashboard (metrics + alerts)

## Framework-Ready Abstractions

- [ ] Define storage interfaces and adapters (SQLite, JSON, others)
- [ ] Encapsulate RAG pipeline steps with clear extension points
- [ ] Plugin system for new vector stores and LLM backends
- [ ] Create `prompt_lib.py` for persona-specific prompts and templates
- [ ] Add RAG document "references" surfacing in responses

---

Guidelines: Prefer small, incremental PRs. Reference items by copying the line, and mark with [x] when done.
