"""Centralized engine configuration.

All tunable thresholds in one place. Every value can be overridden via
environment variable. Defaults match the values validated against the
LoCoMo benchmark (85.55, certified) and the neuroscience literature (ACT-R, Rescorla-Wagner).
"""
from __future__ import annotations

import os


def _float(key: str, default: str) -> float:
    return float(os.getenv(key, default))


def _int(key: str, default: str) -> int:
    return int(os.getenv(key, default))


def _float_override(key: str) -> float | None:
    """Return the explicit env override for `key`, or None if unset."""
    val = os.getenv(key)
    return float(val) if val is not None else None


def _bool_live(key: str, default: bool = False) -> bool:
    """Read a boolean env flag *live* (not frozen at import).

    Truthy: 1/true/yes/on (case-insensitive). Reading live lets a process set
    the flag after import and lets tests monkeypatch it without a config reload.
    """
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# ACT-R Scoring (scoring.py)
# ---------------------------------------------------------------------------
ACTR_DECAY_EXPONENT = _float("GENESYS_ACTR_DECAY", "0.5")
RELEVANCE_VECTOR_WEIGHT = _float("GENESYS_RELEVANCE_VECTOR_WEIGHT", "0.7")
RELEVANCE_KEYWORD_WEIGHT = _float("GENESYS_RELEVANCE_KEYWORD_WEIGHT", "0.3")
MIN_CONNECTIVITY = _float("GENESYS_MIN_CONNECTIVITY", "0.1")

# ---------------------------------------------------------------------------
# Status Transitions (transitions.py)
# ---------------------------------------------------------------------------
TAGGED_EXPIRE_HOURS = _int("GENESYS_TAGGED_EXPIRE_HOURS", "24")
ACTIVE_TO_EPISODIC_THRESHOLD = _float("GENESYS_ACTIVE_EPISODIC_THRESHOLD", "0.6")
ACTIVE_TO_EPISODIC_SESSIONS = _int("GENESYS_ACTIVE_EPISODIC_SESSIONS", "3")
DORMANCY_THRESHOLD = _float("GENESYS_DORMANCY_THRESHOLD", "0.15")
DORMANCY_DAYS = _int("GENESYS_DORMANCY_DAYS", "90")
DORMANCY_MAX_REACTIVATIONS = _int("GENESYS_DORMANCY_MAX_REACTIVATIONS", "3")

# ---------------------------------------------------------------------------
# Active Forgetting (forgetting.py)
# ---------------------------------------------------------------------------
FORGETTING_THRESHOLD = _float("GENESYS_FORGETTING_THRESHOLD", "0.01")
# An orphan is only prunable after this many days without a recall. The score
# alone cannot carry "irrelevant": an orphan's connectivity factor is zero by
# construction, so its score is zero the moment it is rescored.
FORGETTING_MIN_IDLE_DAYS = _float("GENESYS_FORGETTING_MIN_IDLE_DAYS", "30")
# Seconds between maintenance passes (rescore, transitions, forgetting) in the
# stdio server. 0 disables the loop; the hosted service uses 600 as well.
MAINTENANCE_INTERVAL_S = _float("GENESYS_MAINTENANCE_INTERVAL_S", "600")

# ---------------------------------------------------------------------------
# Core Memory Promotion (promoter.py)
# ---------------------------------------------------------------------------
CORE_THRESHOLD = _float("GENESYS_CORE_THRESHOLD", "0.55")
CORE_ACTIVATION_WEIGHT = _float("GENESYS_CORE_ACTIVATION_WEIGHT", "0.4")
CORE_HUB_WEIGHT = _float("GENESYS_CORE_HUB_WEIGHT", "0.3")
CORE_SCHEMA_WEIGHT = _float("GENESYS_CORE_SCHEMA_WEIGHT", "0.2")
CORE_STABILITY_WEIGHT = _float("GENESYS_CORE_STABILITY_WEIGHT", "0.1")
AUTO_PROMOTE_CATEGORIES = [
    c.strip()
    for c in os.getenv(
        "GENESYS_AUTO_PROMOTE_CATEGORIES",
        "professional,educational,family,location",
    ).split(",")
]
HUB_SCORE_CAP = _float("GENESYS_HUB_SCORE_CAP", "3.0")
SCHEMA_NEIGHBOR_CAP = _int("GENESYS_SCHEMA_NEIGHBOR_CAP", "10")
STABILITY_CAP = _float("GENESYS_STABILITY_CAP", "3.0")

# ---------------------------------------------------------------------------
# Recall Relevance Filtering (tools.py — memory_recall)
# ---------------------------------------------------------------------------
# Cosine similarity floors are embedder-dependent: OpenAI's
# text-embedding-3-small produces well-separated similarities for genuine
# matches (~0.5+), but local sentence-transformers models and any
# other/unknown embedder (including test doubles with no embedder at all)
# cluster much lower, so a fixed OpenAI-tuned floor silently filters out
# true positives. These constants are the fallback defaults consulted by
# resolve_recall_min_similarity()/resolve_core_inject_min_similarity() below;
# an explicit GENESYS_RECALL_MIN_SIMILARITY / GENESYS_CORE_INJECT_MIN_SIMILARITY
# env var always wins over embedder-based defaults.
RECALL_MIN_SIMILARITY_OPENAI_DEFAULT = 0.5
RECALL_MIN_SIMILARITY_OTHER_DEFAULT = 0.2
CORE_INJECT_MIN_SIMILARITY_OPENAI_DEFAULT = 0.45
CORE_INJECT_MIN_SIMILARITY_OTHER_DEFAULT = 0.2

RECALL_MIN_SIMILARITY_OVERRIDE = _float_override("GENESYS_RECALL_MIN_SIMILARITY")
CORE_INJECT_MIN_SIMILARITY_OVERRIDE = _float_override("GENESYS_CORE_INJECT_MIN_SIMILARITY")

# Zero-results floor for memory_recall. The min-similarity gate above filters
# pure-vector hits that fall below the (embedder-dependent) floor. For a query
# that embeds far from everything — inference/paraphrase questions with little
# lexical overlap — that gate can gut the result set to near-empty, starving
# the answerer of material even though k asked for many. When the surviving
# union is smaller than min(k, RECALL_RESULT_FLOOR), recall backfills with the
# next-best BELOW-threshold vector hits (flagged low_confidence) up to the
# floor. This does NOT change threshold semantics for normal (well-populated)
# results — it only fires when results are already starved. Set to 0 to disable
# the backfill entirely and restore strict threshold-only behavior. A per-call
# `min_results` argument to memory_recall overrides this default.
RECALL_RESULT_FLOOR = _int("GENESYS_RECALL_RESULT_FLOOR", "10")

# Retained for backward compatibility with any external readers of the old
# single-value constants; reflects the OpenAI-tuned default (or the explicit
# env override). Prefer resolve_recall_min_similarity()/
# resolve_core_inject_min_similarity() which are embedder-aware.
RECALL_MIN_SIMILARITY = (
    RECALL_MIN_SIMILARITY_OVERRIDE
    if RECALL_MIN_SIMILARITY_OVERRIDE is not None
    else RECALL_MIN_SIMILARITY_OPENAI_DEFAULT
)
CORE_INJECT_MIN_SIMILARITY = (
    CORE_INJECT_MIN_SIMILARITY_OVERRIDE
    if CORE_INJECT_MIN_SIMILARITY_OVERRIDE is not None
    else CORE_INJECT_MIN_SIMILARITY_OPENAI_DEFAULT
)


def _embedder_recommended(embeddings: object | None, attr: str) -> float | None:
    """Read a `recommended_*_similarity` float off an embedding provider, if present.

    Uses getattr rather than a Protocol/isinstance check so any embedder
    (including test doubles) can opt in; non-numeric or missing values
    (e.g. Mock() attributes auto-created by AsyncMock-based test doubles)
    are treated as "no recommendation".
    """
    value = getattr(embeddings, attr, None)
    return float(value) if isinstance(value, (int, float)) else None


def resolve_recall_min_similarity(embeddings: object | None) -> float:
    """Effective memory_recall similarity floor.

    Precedence: explicit GENESYS_RECALL_MIN_SIMILARITY env var > the
    embedder's own `recommended_min_similarity` > the generic non-OpenAI
    default (0.2).
    """
    if RECALL_MIN_SIMILARITY_OVERRIDE is not None:
        return RECALL_MIN_SIMILARITY_OVERRIDE
    recommended = _embedder_recommended(embeddings, "recommended_min_similarity")
    return recommended if recommended is not None else RECALL_MIN_SIMILARITY_OTHER_DEFAULT


def resolve_core_inject_min_similarity(embeddings: object | None) -> float:
    """Effective core-memory-injection similarity floor.

    Precedence: explicit GENESYS_CORE_INJECT_MIN_SIMILARITY env var > the
    embedder's own `recommended_core_min_similarity` > the generic
    non-OpenAI default (0.2).
    """
    if CORE_INJECT_MIN_SIMILARITY_OVERRIDE is not None:
        return CORE_INJECT_MIN_SIMILARITY_OVERRIDE
    recommended = _embedder_recommended(embeddings, "recommended_core_min_similarity")
    return recommended if recommended is not None else CORE_INJECT_MIN_SIMILARITY_OTHER_DEFAULT

# ---------------------------------------------------------------------------
# Hybrid retrieval fusion — I3-2 (tools.py — memory_recall)
# ---------------------------------------------------------------------------
# The recall union of vector hits + per-term keyword hits is, by default, ranked
# by the vector cosine of the FULL query with only a flat +0.1 nudge for keyword
# membership (RRF_KEYWORD_BONUS). Two persistent failure modes survive that
# scheme (G2-P mechanism annotation, Day 3):
#   * WRONG_INSTANCE_RETRIEVAL — both candidate turns are retrieved but the wrong
#     one outranks (the gold turn matches MORE disambiguating query terms yet its
#     lower full-question cosine buries it; +0.1 can't close the gap).
#   * RETRIEVAL_MISS — a genuine keyword hit sits below the top-k cut because its
#     cosine is low and +0.1 is too small to lift it into the window.
# GENESYS_HYBRID_RRF turns on proper Reciprocal Rank Fusion of two rankings — the
# vector list (by cosine) and a keyword list (by term COVERAGE, i.e. how many
# distinct query terms a node matched, tie-broken by cosine). RRF's reciprocal-
# rank damping lets multi-term agreement outweigh a single strong cosine hit,
# while its bounded contribution stops a common term from dominating. The fused
# score is normalized to [0,1] so it composes unchanged with the existing core
# injection, spreading-activation, and superseded-decay stages. It also stems
# trailing plurals off keyword terms ("cats"->"cat") so a singular gold turn is
# matched. Flag OFF (default) reproduces the +0.1-bonus behaviour byte-for-byte.
HYBRID_RRF_K = _int("GENESYS_HYBRID_RRF_K", "60")
# Legacy flat in-both bonus (used when RRF is OFF; unchanged historical value).
RRF_KEYWORD_BONUS = _float("GENESYS_RRF_KEYWORD_BONUS", "0.1")


def hybrid_rrf_enabled() -> bool:
    """True iff I3-2 RRF fusion + keyword plural-stemming is enabled (env, live)."""
    return _bool_live("GENESYS_HYBRID_RRF", False)


# ---------------------------------------------------------------------------
# Date-anchored reranking — I3-4 (tools.py — memory_recall)
# ---------------------------------------------------------------------------
# For a date-SCOPED question (one carrying an absolute date anchor — a month+year,
# an explicit date, or a date window like "the last two weeks of August 2023"),
# boost retrieved turns whose session date or resolved [event: ... -> YYYY-MM-DD]
# date falls inside the anchor window. Additive and BOOST-ONLY: it can only lift a
# date-matching turn, never demote one, so a query with no parseable absolute
# anchor (all relative-temporal and non-temporal questions) is a strict no-op —
# this is what keeps it off the [event:]-resolution path that carries Temporal's
# net gain. Flag OFF (default) is a byte-for-byte no-op.
DATE_RERANK_BOOST = _float("GENESYS_DATE_RERANK_BOOST", "0.25")


def date_rerank_enabled() -> bool:
    """True iff I3-4 date-anchored reranking is enabled (env, live)."""
    return _bool_live("GENESYS_DATE_RERANK", False)


# ---------------------------------------------------------------------------
# Auto-linking (tools.py — memory_store)
# ---------------------------------------------------------------------------
# An auto-link creates *permanent* graph structure, so its similarity floor
# should sit ABOVE the transient recall floor: link only when two memories are
# clearly the same topic, not merely mutually retrievable. Like the recall
# floors above, the genuine-match band is embedder-dependent — OpenAI's
# text-embedding-3-small genuine matches sit ~0.5+ (recall floor 0.5), so 0.6
# means "clearly the same topic". Local MiniLM (and unknown embedders) cluster
# lower AND noisier: field reports show noise pairs at ~0.44, i.e. ABOVE the
# local genuine-match band top (~0.4), so the non-OpenAI floor sits above both
# (0.45) — under local embeddings an auto-link only forms on near-duplicate
# content, which is the conservative right answer for permanent structure.
# These are the fallback defaults consulted by
# resolve_autolink_min_similarity() below; an explicit
# GENESYS_AUTOLINK_MIN_SIMILARITY env var always wins over embedder-based
# defaults.
#
# Two structural caps bound the "hairball": AUTOLINK_MAX_EDGES caps fan-out
# (how many auto-links a single memory_store may create), and
# AUTOLINK_MAX_NODE_DEGREE caps *accumulation* (how many auto_link edges any
# single node may accrete as the target of later stores) — fan-out alone
# still lets a hub gain one edge per store forever.
AUTOLINK_MIN_SIMILARITY_OPENAI_DEFAULT = 0.6
AUTOLINK_MIN_SIMILARITY_OTHER_DEFAULT = 0.45
AUTOLINK_MIN_SIMILARITY_OVERRIDE = _float_override("GENESYS_AUTOLINK_MIN_SIMILARITY")
AUTOLINK_MAX_EDGES = _int("GENESYS_AUTOLINK_MAX_EDGES", "3")
AUTOLINK_MAX_NODE_DEGREE = _int("GENESYS_AUTOLINK_MAX_NODE_DEGREE", "10")


def resolve_autolink_min_similarity(embeddings: object | None) -> float:
    """Effective memory_store auto-link similarity floor.

    Precedence: explicit GENESYS_AUTOLINK_MIN_SIMILARITY env var > the
    embedder's own `recommended_autolink_min_similarity` > the generic
    non-OpenAI default (0.45).
    """
    if AUTOLINK_MIN_SIMILARITY_OVERRIDE is not None:
        return AUTOLINK_MIN_SIMILARITY_OVERRIDE
    recommended = _embedder_recommended(embeddings, "recommended_autolink_min_similarity")
    return recommended if recommended is not None else AUTOLINK_MIN_SIMILARITY_OTHER_DEFAULT


# ---------------------------------------------------------------------------
# Conflict-hint scan (tools.py — memory_store possible_conflicts)
# ---------------------------------------------------------------------------
# The heuristic conflict scan is deliberately DECOUPLED from the auto-link
# floor: a changed figure between two versions of a fact often sits below the
# strict "clearly the same topic" band (the exact regime where the old 0.3
# auto-link floor used to surface such pairs), so gating the scan on the
# auto-link floor silently shrinks conflict detection whenever that floor is
# raised. The scan reaches down to the recall floor by default and looks at a
# wider vector window (CONFLICT_SCAN_K) than the auto-link fan-out.
CONFLICT_MIN_SIMILARITY_OVERRIDE = _float_override("GENESYS_CONFLICT_MIN_SIMILARITY")
CONFLICT_SCAN_K = _int("GENESYS_CONFLICT_SCAN_K", "8")


def resolve_conflict_min_similarity(embeddings: object | None) -> float:
    """Effective similarity floor for the possible_conflicts heuristic scan.

    Precedence: explicit GENESYS_CONFLICT_MIN_SIMILARITY env var > the
    embedder's own `recommended_min_similarity` (i.e. the recall floor) > the
    generic non-OpenAI recall default (0.2).
    """
    if CONFLICT_MIN_SIMILARITY_OVERRIDE is not None:
        return CONFLICT_MIN_SIMILARITY_OVERRIDE
    recommended = _embedder_recommended(embeddings, "recommended_min_similarity")
    return recommended if recommended is not None else RECALL_MIN_SIMILARITY_OTHER_DEFAULT

# ---------------------------------------------------------------------------
# Edge Staleness
# ---------------------------------------------------------------------------
EDGE_STALE_DAYS = _int("GENESYS_EDGE_STALE_DAYS", "30")
EDGE_STALE_PENALTY = _float("GENESYS_EDGE_STALE_PENALTY", "0.5")

# ---------------------------------------------------------------------------
# Cascade Reactivation (reactivation.py)
# ---------------------------------------------------------------------------
CASCADE_DEPTH = _int("GENESYS_CASCADE_DEPTH", "2")
CASCADE_DECAY_FACTOR = _float("GENESYS_CASCADE_DECAY_FACTOR", "0.3")
DORMANT_REVIVAL_THRESHOLD = _float("GENESYS_DORMANT_REVIVAL_THRESHOLD", "0.1")

# ---------------------------------------------------------------------------
# Ingestion Limits
# ---------------------------------------------------------------------------
MAX_INGEST_FILE_MB = _int("GENESYS_MAX_INGEST_FILE_MB", "100")
