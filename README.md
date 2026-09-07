<!-- mcp-name: io.github.Astrix-Labs/papez -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Astrix-Labs/papez/main/docs/assets/papez-wordmark-dark.svg">
    <img alt="Papez: your personal memory for AI" width="360" src="https://raw.githubusercontent.com/Astrix-Labs/papez/main/docs/assets/papez-wordmark-light.svg">
  </picture>
</p>

[![PyPI](https://img.shields.io/pypi/v/papez)](https://pypi.org/project/papez/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/papez)](https://pypi.org/project/papez/)
[![CI](https://github.com/Astrix-Labs/papez/actions/workflows/ci.yml/badge.svg)](https://github.com/Astrix-Labs/papez/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

# Papez

**The intelligence layer for AI memory.**

> Papez doesn't just remember what happened; it remembers why. A scoring engine + causal graph + lifecycle manager for AI agent memory. Speaks MCP natively.

## LoCoMo benchmark (certified)

| System | Score | Protocol |
|---|---|---|
| **Papez** | **85.55 ± 0.37** | Frozen: gpt-4o-mini answerer + judge, temp 0, n=1,540, cats 1–4, 10 runs (July 2026) |
| Zep | 75.14 | Comparable published setup |
| Mem0 | 66.9 | Comparable published setup (Mem0 paper) |

Self-reported vendor figures above ~90 use different answerers/judges and are not comparable — the oracle retrieval ceiling under this frozen protocol is 94.9. Reproduce it yourself: [Astrix-Labs/locomo-harness](https://github.com/Astrix-Labs/locomo-harness) · [full methodology](https://papez.ai/developers/methodology) · [per-run results](https://papez.ai/benchmarks/locomo).

**Hosted product:** [papez.ai](https://papez.ai) — your personal memory for AI, carried across ChatGPT, Claude, and every MCP app · [Pricing](https://papez.ai/pricing) · [Developer docs](https://papez.ai/developers) · [Benchmark methodology](https://papez.ai/developers/methodology) (85.55 on LoCoMo, certified over 10 runs, receipts published)

## What is this

Papez is a scoring engine, causal graph, and lifecycle manager for AI memory. Memories are scored by a multiplicative formula (relevance × connectivity × reactivation), connected in a causal graph, and actively forgotten when they become irrelevant.

This package (`papez`) is the core library: an in-memory causal graph engine with optional JSON persistence, plus a stdio MCP server. It has no database dependency and no REST API. A hosted product built on top of this library — with Postgres, additional storage backends, and a REST/HTTP MCP API — is available separately at `api.papez.ai`; it is not part of this package.

## Why

- **Flat memory doesn't scale.** Dumping everything into a vector store gives you recall with zero understanding. The 500th memory buries the 5 that matter.
- **No forgetting = no intelligence.** Real memory systems forget. Without active pruning, your AI drowns in stale context.
- **No causal reasoning.** Vector similarity can't answer "why did I choose X?" — you need a graph.

Your AI remembers everything but understands nothing. Papez fixes that.

## Quick Start

Requires Python 3.11 or newer.

Install the package. The base install has zero database dependencies — state lives in memory and is optionally persisted to a JSON file.

```bash
pip install papez
```

Optional extras:

```bash
pip install 'papez[openai]'      # OpenAI embeddings
pip install 'papez[local]'       # Local embeddings (sentence-transformers, no API key)
pip install 'papez[anthropic]'   # LLM-based causal inference (consolidation, contradiction detection)
```

Run the stdio MCP server directly:

```bash
python3 -m papez
```

### From source

```bash
git clone https://github.com/Astrix-Labs/papez.git
cd papez
pip install -e '.[dev]'
pytest tests/
```

## Connect to your AI

### Claude Code

```bash
claude mcp add papez -- python -m papez
```

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "papez": {
      "command": "python",
      "args": ["-m", "papez"]
    }
  }
}
```

### Reliability & retries

The stdio server is a single local process. Under load — or during a restart or
redeploy of a hosted transport in front of it — a tool call can transiently fail
or the connection can briefly go unresponsive. Memory writes and reads are not
worth crashing an agent turn over, so **clients should degrade gracefully rather
than treat a memory call as fatal**:

- **The server degrades gracefully too**: a tool exception (or a missing
  required argument) is returned as a structured
  `{"error": "...", "retryable": bool}` payload instead of a protocol-level
  MCP failure, so a memory hiccup never crashes the transport. The
  `retryable` flag encodes the guidance below — `true` only for read tools.
- **Retry idempotent reads** (`memory_recall`, `memory_search`, `memory_traverse`,
  `memory_explain`, `memory_stats`) with a short bounded backoff (e.g. 2–3
  attempts). These have no side effects worth worrying about beyond reactivation
  bookkeeping.
- **Do not blindly retry `memory_store` / `memory_amend`** on an ambiguous timeout
  — a silent success followed by a retry creates a duplicate node. Prefer to
  continue the turn and reconcile on the next `memory_recall`, or pass a stable
  `source_session` so duplicates are easy to spot.
- **Treat memory as best-effort context, not a hard dependency.** If a call fails,
  proceed with whatever context you already have and try again next turn rather
  than aborting. The graph is durable; a missed write is recoverable, a crashed
  agent turn is not.

## MCP Tools

| Tool | Description |
|------|-------------|
| `memory_store` | Store a new memory. Use `related` for writer-specified **typed** edges (`{id, type}`); `related_to` is legacy and always creates `caused_by`. Optional `category`. May return `possible_conflicts` (heuristic hints). |
| `memory_amend` | Record a correction: creates a new memory that **supersedes** an existing one. The old memory is kept (decayed in recall), not deleted. |
| `memory_recall` | Recall memories by natural language query (vector + keyword + graph spreading activation). Supports `verbosity: "concise"` for lightweight payloads. |
| `memory_search` | Filtered vector search by status, category, date (`since`), last-active date (`active_since`), or entity. Pass an **empty query** to enumerate by recency instead (no embedder needed) — with `since`/`active_since` this answers "what's new since I last looked" without knowing what to query for. |
| `memory_traverse` | Walk the causal graph from a node. Returns reachable **nodes and the edges** of the induced subgraph (`source/target/type/weight/created_by`) — a superset of the BFS tree, so paths can be reconstructed. Honors `edge_types`. |
| `memory_explain` | Explain a memory's score. Includes a `score_model` block (formula + live per-force breakdown + staleness note) and `removal_impact`. |
| `memory_stats` | Get memory system statistics |
| `pin_memory` | Pin a memory so it's never forgotten |
| `unpin_memory` | Unpin a previously pinned memory |
| `delete_memory` | Permanently delete a memory |
| `list_core_memories` | List core memories, optionally filtered by category |
| `set_core_preferences` | Set user preferences for core memory categories |
| `promote_to_org` | Promote a private memory to org visibility |

### Writer-specified edges & corrections

`memory_store`'s `related` argument lets the writer set edge semantics instead of
guessing. Each entry is `{"id": "<node-id>", "type": "<edge-type>"}`, directed
`new_node --type--> target` (so `supersedes` means the new node supersedes the
target). Invalid types are rejected **before** the node is created — explicit
writes never half-succeed. `related_to` still exists but always creates
`caused_by`; prefer `related`.

To correct a fact, use `memory_amend(node_id, content, reason=...)`: it stores the
new version, links it `SUPERSEDES → old`, and **keeps the old memory** for audit.
Recall automatically deprioritizes superseded hits and tags them with
`superseded_by`.

When you `memory_store` something that lexically disagrees with an auto-link
candidate (a changed number, a negation), the result may include
`possible_conflicts` — heuristic hints, **not** verified contradictions, and never
materialized as edges. Use them to decide whether to `memory_amend`.

### Concise recall

`memory_recall(query, verbosity="concise")` skips the causal-chain enrichment and
returns only `id / summary / status / score / activation / is_core` (plus
`superseded_by` when set) per hit — much cheaper on tokens for high-frequency
lookups. `verbosity="full"` (the default) is unchanged. Reactivation writes still
occur in both modes (they are governed by `read_only`, not `verbosity`).

See [`docs/scoring.md`](docs/scoring.md) for what `activation` / `decay_score`
actually mean — in short, it is a retention weight that **rises** when a memory is
recalled, not a countdown to deletion.

## How it works

Every memory is scored by three forces multiplied together:

```
decay_score = relevance × connectivity × reactivation
```

- **Relevance** decays over time. Old memories fade unless reinforced.
- **Connectivity** rewards memories with many causal links. Hub memories survive.
- **Reactivation** boosts memories that keep getting recalled. Frequency matters.

Because the formula is multiplicative, a memory must score on *all three* axes to survive. A highly connected but never-accessed memory still decays. A frequently recalled but causally orphaned memory still fades.

`decay_score` (aliased `activation` on every hit) is a **retention weight, not a deletion countdown** — recalling a memory *raises* it, and a low score just means "resting," not "doomed." Deletion requires a low score **and** orphaned **and** unpinned **and** non-core **and** non-org **and** idle (not stored, recalled or reactivated for 30 days, `GENESYS_FORGETTING_MIN_IDLE_DAYS`), all at once. The stdio server runs the rescore-transition-prune pass every 10 minutes (`GENESYS_MAINTENANCE_INTERVAL_S`; 0 disables it). See [`docs/scoring.md`](docs/scoring.md) for the full model and worked numbers.

```
STORE → ACTIVE → DORMANT → FADING → PRUNED
           ↑                    │
           └── reactivation ────┘
                                  (only if score=0, orphan, not pinned)
```

Memories can also be promoted to **core** status — structurally important memories that are auto-pinned and never pruned.

## Benchmark Results

See the [certified LoCoMo results](#locomo-benchmark-certified) at the top of this README: **85.55 ± 0.37** over 10 runs under a frozen protocol (gpt-4o-mini answerer and judge, temperature 0, n=1,540, categories 1–4). Category 5 — adversarial questions with disputed ground truth — is excluded, matching the comparable published setups.

Every run is reproducible: the harness is at [Astrix-Labs/locomo-harness](https://github.com/Astrix-Labs/locomo-harness), with [full methodology](https://papez.ai/developers/methodology) and [per-run results](https://papez.ai/benchmarks/locomo) published. Reproduction scripts for the in-repo scenarios are in [`benchmarks/`](benchmarks/).

## Storage backend

This package ships one storage backend: an in-memory causal graph (`storage/memory.py`) with optional JSON persistence via `GENESYS_PERSIST_PATH`. No database is required.

Additional backends — Postgres/pgvector, FalkorDB, MongoDB, and an Obsidian vault adapter — along with a REST API, OAuth, and multi-user auth, are part of the hosted product at `api.papez.ai` and are not included in this repo.

Want a different storage backend for the open-source library? Implement the provider protocols in [`storage/base.py`](src/papez/storage/base.py) and bring your own.

## Configuration

Copy `.env.example` to `.env` and set:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Unless `GENESYS_EMBEDDER=local` | Embeddings |
| `ANTHROPIC_API_KEY` | No | Enables LLM-based causal inference (consolidation, contradiction detection). Off by default — without it, causal edges only come from edges the caller explicitly declares in `memory_store` plus cosine-similarity linking. |
| `GENESYS_EMBEDDER` | No | `openai` (default) or `local` (sentence-transformers, no API key) |
| `GENESYS_PERSIST_PATH` | No | JSON file path to persist state across restarts (in-memory otherwise) |
| `GENESYS_USER_ID` | No | Default user ID for single-tenant mode |

### Auto-link tuning

Auto-linking connects a newly stored memory to semantically similar existing
memories. If it is too permissive you get a "hairball" — everything ends up ~2
hops from everything, which destroys traversal scoping. Three knobs bound it:

| Variable | Default | Description |
|----------|---------|-------------|
| `GENESYS_AUTOLINK_MIN_SIMILARITY` | embedder-recommended | Cosine floor to create an auto-link. Explicit value wins over the embedder default. |
| `GENESYS_AUTOLINK_MAX_EDGES` | `3` | Max auto-links a single `memory_store` may create. Caps fan-out. |
| `GENESYS_AUTOLINK_MAX_NODE_DEGREE` | `10` | Max `auto_link` edges any single node may *accumulate* as a target. Fan-out alone still lets a hub gain one edge per store forever; this caps the hub itself. |

The floor is **embedder-aware**: an auto-link is permanent graph structure, so its
floor sits *above* the transient recall floor. When `GENESYS_AUTOLINK_MIN_SIMILARITY`
is unset, the effective floor is the embedder's recommendation — **0.6 for OpenAI**
(`text-embedding-3-small`, whose genuine matches cluster ~0.5+) and **0.45 for
local** sentence-transformers (whose genuine matches cluster ~0.2–0.4 but whose
noise pairs have been observed at ~0.44, so only near-duplicate content
auto-links locally). Any unknown embedder falls back to 0.45. Auto-linking also
de-dupes: if a pair is already connected by *any* edge (e.g. a `user_explicit`
`caused_by`), no parallel `auto_link related_to` is created.

The `possible_conflicts` hint on `memory_store` scans with its **own, lower
floor** (`GENESYS_CONFLICT_MIN_SIMILARITY`, defaulting to the recall floor) over
a wider window (`GENESYS_CONFLICT_SCAN_K`, default 8) — so tightening the
auto-link floor never shrinks conflict detection.

### Recall / relevance floors

The same embedder-aware pattern governs recall filtering:

| Variable | Default | Description |
|----------|---------|-------------|
| `GENESYS_RECALL_MIN_SIMILARITY` | embedder-recommended (OpenAI 0.5 / other 0.2) | Cosine floor below which pure vector hits are dropped from `memory_recall`. Keyword hits bypass it. |
| `GENESYS_CORE_INJECT_MIN_SIMILARITY` | embedder-recommended (OpenAI 0.45 / other 0.2) | Floor for injecting auto-promoted core memories into recall results. Pinned memories are always injected. |

### Scoring knobs

The three-force scoring formula and its lifecycle thresholds are all
env-configurable (see [`engine/config.py`](src/papez/engine/config.py) and
[`docs/scoring.md`](docs/scoring.md)): `GENESYS_ACTR_DECAY`,
`GENESYS_RELEVANCE_VECTOR_WEIGHT`, `GENESYS_RELEVANCE_KEYWORD_WEIGHT`,
`GENESYS_MIN_CONNECTIVITY`, `GENESYS_FORGETTING_THRESHOLD`, the `GENESYS_DORMANCY_*`
transition thresholds, and the `GENESYS_CORE_*` promotion weights.

See [`.env.example`](.env.example) for all options.

## Built by

Papez is built by [Rishi Meka](https://github.com/rishimeka) at [Astrix Labs](https://astrixlabs.ai). It came out of frustration with re-explaining project context to Claude every session. The goal is the intelligence layer between your LLM and your memory — fully open source.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[AGPL-3.0-or-later](LICENSE)

> **Note:** Papez releases prior to v0.3.6 were documented as Apache 2.0 in error. The LICENSE file has always contained the AGPLv3 text. From v0.3.6 onward, all documentation correctly references AGPL-3.0-or-later with a Contributor License Agreement.
