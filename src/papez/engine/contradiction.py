"""Contradiction detection between memories."""
from __future__ import annotations

import re

from papez.models.edge import MemoryEdge
from papez.models.enums import EdgeType, MemoryStatus
from papez.models.node import MemoryNode
from papez.storage.base import EmbeddingProvider, GraphStorageProvider, LLMProvider

_NEGATION_RE = re.compile(r"\b(not|never|no longer|n't|isn't|won't|doesn't)\b", re.IGNORECASE)
_NUM_OR_WORD_RE = re.compile(r"[a-zA-Z]+|[$€£]?\d[\d,.]*%?")

# Anchor words that carry no quantity semantics. "budget is 50000" and
# "latency is 200" must NOT collide on the shared anchor "is" — that
# false-fired numeric_mismatch on numbers measuring different things
# (field report: a 200ms latency memory flagged a $50,000 budget memory).
_ANCHOR_STOPWORDS = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "the", "a", "an",
    "of", "to", "at", "in", "on", "for", "about", "around", "than", "over",
    "under", "by", "with", "as", "and", "or", "now", "currently", "it",
    "its", "roughly", "approximately", "only", "just", "than",
})

# Unit tokens that may follow a number ("6 weeks", "200 ms"). A number's
# unit becomes part of its comparison key so quantities in different units
# never conflict ("6 weeks" vs "8 months" measure different spans).
_UNIT_WORDS = frozenset({
    "ms", "s", "sec", "secs", "seconds", "min", "mins", "minutes", "h",
    "hr", "hrs", "hours", "day", "days", "week", "weeks", "month",
    "months", "year", "years", "k", "m", "b", "percent", "%", "usd",
    "dollars", "eur", "gbp", "people", "engineers", "users", "nodes",
})


def _number_contexts(text: str) -> dict[tuple[str, str], set[str]]:
    """Map each number to (anchor, unit) → set of number tokens.

    - anchor: nearest preceding NON-stopword word ("budget is 50k" → "budget").
    - unit: currency prefix ("$50,000" → "$"), attached suffix ("%"), or the
      following unit word ("6 weeks" → "weeks"); "" when none.
    Numbers with no meaningful anchor key on anchor "".
    """
    contexts: dict[tuple[str, str], set[str]] = {}
    tokens = _NUM_OR_WORD_RE.findall(text)
    last_anchor = ""
    for i, tok in enumerate(tokens):
        first = tok[0]
        if first.isdigit() or first in "$€£":
            unit = ""
            number = tok
            if first in "$€£":
                unit, number = first, tok[1:]
            elif tok.endswith("%"):
                unit, number = "%", tok[:-1]
            elif i + 1 < len(tokens) and tokens[i + 1].lower() in _UNIT_WORDS:
                unit = tokens[i + 1].lower()
            contexts.setdefault((last_anchor, unit), set()).add(number)
        else:
            low = tok.lower()
            if low not in _ANCHOR_STOPWORDS and low not in _UNIT_WORDS:
                last_anchor = low
    return contexts


def heuristic_conflict_signal(text_a: str, text_b: str) -> str | None:
    """Cheap, dependency-free lexical divergence check between two texts.

    Returns ``"numeric_mismatch"`` when the two texts mention *differing*
    numbers in a comparable position — i.e. some shared nearest-preceding
    context word carries different number sets in each text ("costs 50" vs
    "costs 75"). Numbers in unrelated positions (a date in one text, an ID in
    the other) do NOT fire, which keeps the hint from being noise on realistic
    workloads. Returns ``"negation"`` when exactly one text is negated;
    otherwise ``None``. These are hints only — never verified contradictions
    and never materialized as edges.
    """
    ctx_a = _number_contexts(text_a)
    ctx_b = _number_contexts(text_b)
    for key in ctx_a.keys() & ctx_b.keys():
        if ctx_a[key] != ctx_b[key]:
            return "numeric_mismatch"
    neg_a = bool(_NEGATION_RE.search(text_a))
    neg_b = bool(_NEGATION_RE.search(text_b))
    if neg_a != neg_b:
        return "negation"
    return None


async def detect_contradictions(
    new_node: MemoryNode,
    graph: GraphStorageProvider,
    embeddings: EmbeddingProvider,
    llm: LLMProvider,
) -> list[tuple[str, float]]:
    """
    Check if new_node contradicts existing memories.
    1. Vector search for similarity > 0.85
    2. LLM confirmation
    3. Create CONTRADICTS edges for confirmed contradictions
    Returns list of (contradicted_node_id, confidence).
    """
    if not new_node.embedding:
        return []

    # Find highly similar memories (potential contradictions)
    candidates = await graph.vector_search(new_node.embedding, k=20)
    contradictions: list[tuple[str, float]] = []

    for candidate_node, sim_score in candidates:
        # Skip self
        if str(candidate_node.id) == str(new_node.id):
            continue
        # Only check high-similarity pairs
        # FalkorDB cosine distance: lower = more similar; convert to similarity
        similarity = 1.0 - sim_score if sim_score <= 1.0 else sim_score
        if similarity < 0.85:
            continue

        content_a = new_node.content_full or new_node.content_summary
        content_b = candidate_node.content_full or candidate_node.content_summary
        is_contradiction, confidence, reason = await llm.detect_contradiction(content_a, content_b)

        if is_contradiction and confidence > 0.7:
            # Create CONTRADICTS edge
            edge = MemoryEdge(
                source_id=new_node.id,
                target_id=candidate_node.id,
                type=EdgeType.CONTRADICTS,
                weight=confidence,
                reason=reason,
                created_by="llm_contradiction",
            )
            await graph.create_edge(edge)
            contradictions.append((str(candidate_node.id), confidence))

            # If contradicted memory is core, trigger supersession
            if candidate_node.status == MemoryStatus.CORE:
                supersede_edge = MemoryEdge(
                    source_id=new_node.id,
                    target_id=candidate_node.id,
                    type=EdgeType.SUPERSEDES,
                    weight=confidence,
                    reason=reason,
                    created_by="llm_contradiction",
                )
                await graph.create_edge(supersede_edge)
                await graph.update_node(str(candidate_node.id), {
                    "status": MemoryStatus.EPISODIC,
                })

    return contradictions
