"""I3-2 (hybrid RRF fusion + keyword plural-stemming) and I3-4 (date-anchored
reranking) coverage. Both features are env-flagged and default OFF; these tests
exercise the ON path over the real in-memory provider and assert the OFF path is
unchanged."""
from __future__ import annotations

import math

import pytest

from papez.context import current_user_id
from papez.mcp.tools import MCPToolHandler
from papez.retrieval.date_anchor import (
    node_matches_anchor,
    parse_query_date_anchor,
)
from papez.storage.cache import NullCacheProvider
from papez.storage.memory import InMemoryGraphProvider


@pytest.fixture(autouse=True)
def _user_ctx():
    token = current_user_id.set("test-user")
    yield
    current_user_id.reset(token)


class _MapEmbedder:
    """Deterministic embedder mapping an exact string to a unit vector on axis 0
    with a caller-chosen cosine to the query axis. ``embed`` returns the vector
    registered for the text (default: orthogonal to everything)."""

    dim = 8

    def __init__(self):
        self._map: dict[str, list[float]] = {}

    def register(self, text: str, cos_to_axis0: float) -> None:
        c = max(-1.0, min(1.0, cos_to_axis0))
        vec = [0.0] * self.dim
        vec[0] = c
        vec[1] = math.sqrt(max(0.0, 1.0 - c * c))
        self._map[text] = vec

    @property
    def dimension(self):
        return self.dim

    async def embed(self, text):
        if text in self._map:
            return list(self._map[text])
        # unseen text -> orthogonal to the query axis (cosine 0)
        v = [0.0] * self.dim
        v[2] = 1.0
        return v

    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]


async def _handler(embedder):
    graph = InMemoryGraphProvider()
    await graph.initialize("test-user")
    return MCPToolHandler(graph=graph, embeddings=embedder, cache=NullCacheProvider())


def _rank_of(results, needle):
    for i, m in enumerate(results):
        c = (m.get("content_full") or m.get("summary") or "")
        if needle in c:
            return i
    return None


# ---------------------------------------------------------------------------
# I3-2 — RRF lifts a multi-term-coverage gold above high-cosine no-keyword hits
# ---------------------------------------------------------------------------
class TestHybridRRF:
    async def _seed(self):
        emb = _MapEmbedder()
        query = "xalpha xbeta xgamma"
        emb.register(query, 1.0)
        # GOLD: matches all three query terms (coverage 3) but LOW cosine.
        gold = "GOLDMARK xalpha xbeta xgamma details"
        emb.register(gold, 0.30)
        # Four high-cosine fillers sharing NO query term (coverage 0).
        fillers = [f"FILLER{i} unrelated content here" for i in range(4)]
        for i, f in enumerate(fillers):
            emb.register(f, 0.95 - i * 0.02)
        h = await _handler(emb)
        for f in fillers:
            await h.memory_store(f)
        await h.memory_store(gold)
        return h, query

    @pytest.mark.asyncio
    async def test_rrf_off_buries_gold_below_cosine_fillers(self, monkeypatch):
        monkeypatch.delenv("GENESYS_HYBRID_RRF", raising=False)
        h, query = await self._seed()
        res = (await h.memory_recall(query, k=20))["results"]
        gr = _rank_of(res, "GOLDMARK")
        assert gr is not None
        # Flat +0.1 bonus cannot lift a cosine-0.30 gold over cosine-0.95 fillers.
        assert gr > 0, f"expected gold buried without RRF, got rank {gr}"

    @pytest.mark.asyncio
    async def test_rrf_on_lifts_gold_to_top(self, monkeypatch):
        monkeypatch.setenv("GENESYS_HYBRID_RRF", "1")
        h, query = await self._seed()
        res = (await h.memory_recall(query, k=20))["results"]
        gr = _rank_of(res, "GOLDMARK")
        assert gr == 0, f"RRF should rank the coverage-3 gold first, got rank {gr}"


# ---------------------------------------------------------------------------
# I3-2 — keyword plural-stemming. The postgres provider matches a term with a
# literal ILIKE '%term%' (no stemming), so a plural query term "cats" never
# matches a singular "cat" turn. The lib stems the term BEFORE the keyword
# search when the flag is on. (The in-memory provider stems internally, so we
# assert the stem at the two seams that actually matter: the pure function, and
# the exact terms handed to keyword_search.)
# ---------------------------------------------------------------------------
class TestKeywordPluralStem:
    def test_stem_plural_function(self):
        from papez.mcp.tools import _stem_plural
        assert _stem_plural("cats") == "cat"
        assert _stem_plural("classes") == "class"
        assert _stem_plural("books") == "book"
        assert _stem_plural("states") == "state"
        assert _stem_plural("stories") == "story"
        assert _stem_plural("boxes") == "box"
        # never mangles these
        assert _stem_plural("class") == "class"
        assert _stem_plural("bus") == "bus"      # 'us' guard
        assert _stem_plural("this") == "this"    # 'is' guard
        assert _stem_plural("dress") == "dress"  # 'ss' guard

    async def _spy_terms(self, query, monkeypatch, flag):
        if flag:
            monkeypatch.setenv("GENESYS_HYBRID_RRF", "1")
        else:
            monkeypatch.delenv("GENESYS_HYBRID_RRF", raising=False)
        emb = _MapEmbedder()
        emb.register(query, 1.0)
        h = await _handler(emb)
        seen: list[str] = []
        orig = h.graph.keyword_search

        async def _spy(q, *a, **kw):
            seen.append(q)
            return await orig(q, *a, **kw)

        h.graph.keyword_search = _spy
        await h.memory_recall(query, k=10)
        return seen

    @pytest.mark.asyncio
    async def test_flag_on_searches_stemmed_terms(self, monkeypatch):
        seen = await self._spy_terms("list the cats and classes", monkeypatch, flag=True)
        assert set(seen) == {"list", "cat", "class"}, seen

    @pytest.mark.asyncio
    async def test_flag_off_searches_raw_terms(self, monkeypatch):
        seen = await self._spy_terms("list the cats and classes", monkeypatch, flag=False)
        assert set(seen) == {"list", "cats", "classes"}, seen


# ---------------------------------------------------------------------------
# I3-4 — date-anchored reranking
# ---------------------------------------------------------------------------
class TestDateAnchorRerank:
    async def _seed(self):
        emb = _MapEmbedder()
        query = "trip location august 2023"
        emb.register(query, 1.0)
        out = "location report from the session on 2023-03-01 far away"
        emb.register(out, 0.60)  # higher cosine
        inw = "location report from the session on 2023-08-20 nearby"
        emb.register(inw, 0.50)  # lower cosine, but IN the anchor window
        h = await _handler(emb)
        await h.memory_store(out)
        await h.memory_store(inw)
        return h, query

    @pytest.mark.asyncio
    async def test_rerank_off_keeps_cosine_order(self, monkeypatch):
        monkeypatch.delenv("GENESYS_DATE_RERANK", raising=False)
        monkeypatch.delenv("GENESYS_HYBRID_RRF", raising=False)
        h, query = await self._seed()
        res = (await h.memory_recall(query, k=10))["results"]
        assert _rank_of(res, "2023-03-01") < _rank_of(res, "2023-08-20")

    @pytest.mark.asyncio
    async def test_rerank_on_boosts_in_window_turn(self, monkeypatch):
        monkeypatch.setenv("GENESYS_DATE_RERANK", "1")
        monkeypatch.delenv("GENESYS_HYBRID_RRF", raising=False)
        h, query = await self._seed()
        res = (await h.memory_recall(query, k=10))["results"]
        assert _rank_of(res, "2023-08-20") < _rank_of(res, "2023-03-01"), (
            "in-window turn should outrank the higher-cosine out-of-window turn"
        )
        boosted = next(m for m in res if "2023-08-20" in (m.get("content_full") or m.get("summary") or ""))
        assert boosted.get("date_anchor_boosted") is True

    @pytest.mark.asyncio
    async def test_rerank_no_op_without_absolute_anchor(self, monkeypatch):
        monkeypatch.setenv("GENESYS_DATE_RERANK", "1")
        monkeypatch.delenv("GENESYS_HYBRID_RRF", raising=False)
        emb = _MapEmbedder()
        query = "which trip location did they visit"  # no absolute date anchor
        emb.register(query, 1.0)
        out = "location report from the session on 2023-03-01 far away"
        emb.register(out, 0.60)
        inw = "location report from the session on 2023-08-20 nearby"
        emb.register(inw, 0.50)
        h = await _handler(emb)
        await h.memory_store(out)
        await h.memory_store(inw)
        res = (await h.memory_recall(query, k=10))["results"]
        # No parseable anchor -> strict no-op -> cosine order preserved.
        assert _rank_of(res, "2023-03-01") < _rank_of(res, "2023-08-20")
        assert not any(m.get("date_anchor_boosted") for m in res)


# ---------------------------------------------------------------------------
# Date-anchor parser
# ---------------------------------------------------------------------------
class TestParseQueryDateAnchor:
    @pytest.mark.parametrize(
        "q,start,end",
        [
            ("Where was John between August 11 and August 15 2023?", "2023-08-11", "2023-08-15"),
            ("Which hobby did Dave pick up in October 2023?", "2023-10-01", "2023-10-31"),
            ("How many pets did Andrew have, as of September 2023?", "2023-09-01", "2023-09-30"),
            ("during the last week of August 2023?", "2023-08-25", "2023-08-31"),
            ("in the last two weeks of August 2023?", "2023-08-18", "2023-08-31"),
            ("in the week before 16 November 2023?", "2023-11-09", "2023-11-15"),
            ("the first weekend of August 2023?", "2023-08-01", "2023-08-07"),
            ("What movie did Joanna watch on 1 May, 2022?", "2022-05-01", "2022-05-01"),
            ("What did Audrey eat on October 24, 2023?", "2023-10-24", "2023-10-24"),
        ],
    )
    def test_windows(self, q, start, end):
        import datetime as _dt
        got = parse_query_date_anchor(q)
        assert got is not None, f"expected an anchor for {q!r}"
        assert got[0] == _dt.date.fromisoformat(start)
        assert got[1] == _dt.date.fromisoformat(end)

    @pytest.mark.parametrize(
        "q",
        [
            "Which month was John in Italy?",
            "What month did Tim plan on going to Universal Studios?",
            "When did Caroline attend a pride parade in August?",  # month, no year
            "Which year did Evan start taking care of his health?",  # year word, no absolute date
            "Which book was John reading during his recovery from an ankle injury?",
        ],
    )
    def test_no_anchor_returns_none(self, q):
        assert parse_query_date_anchor(q) is None

    def test_node_matches_anchor(self):
        import datetime as _dt
        anchor = (_dt.date(2023, 8, 1), _dt.date(2023, 8, 31))
        assert node_matches_anchor("came back from a trip on 2023-08-20", None, anchor)
        assert not node_matches_anchor("a note dated 2023-03-01", None, anchor)
        # created_at fallback
        assert node_matches_anchor("no iso date here", _dt.datetime(2023, 8, 5), anchor)
