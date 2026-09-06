"""Date-anchor parsing for I3-4 date-anchored recall reranking.

Two pure helpers, no I/O, no LLM:

* ``parse_query_date_anchor(query)`` — pulls an ABSOLUTE date window out of a
  date-scoped question ("in October 2023", "between August 11 and August 15
  2023", "the last two weeks of August 2023", "on 1 May 2022"). Returns an
  inclusive ``(start, end)`` ``date`` pair, or ``None`` when the question carries
  no absolute anchor. An explicit 4-digit YEAR is required for any anchor to be
  emitted — so relative-only temporal questions ("which month was John in
  Italy?", "which year did Evan start?") and every non-temporal question yield
  ``None`` and are left completely untouched by the reranker.

* ``node_dates(text, created_at)`` — collects the candidate dates carried by a
  memory: every ``YYYY-MM-DD`` literal in its content (the ingest header's
  ``session on YYYY-MM-DD`` plus any resolved ``[event: "..." -> YYYY-MM-DD]``
  tag) and the node's ``created_at`` date.

The reranker boosts a memory iff one of its ``node_dates`` falls inside the
query's anchor window. Everything is deterministic; no rule references any gold
answer or dataset statistic.
"""
from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

# "August 15, 2023" / "August 15 2023" / "15 August 2023" / "1 May, 2022"
_MDY_RE = re.compile(rf"\b({_MONTH_ALT})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_DMY_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT}),?\s+(\d{{4}})\b", re.I)
# "October 2023" / "in September 2023" (no day)
_MY_RE = re.compile(rf"\b({_MONTH_ALT})\s+(\d{{4}})\b", re.I)
# "between <date> and <date>"
_BETWEEN_RE = re.compile(r"\bbetween\b(.+?)\band\b(.+)", re.I)


def _month_bounds(year: int, month: int) -> Tuple[date, date]:
    last = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last)


def _single_date(text: str) -> Optional[date]:
    """Parse the first explicit calendar date in ``text`` (needs a year)."""
    m = _MDY_RE.search(text)
    if m:
        mon = _MONTHS[m.group(1).lower()]
        return _safe_date(int(m.group(3)), mon, int(m.group(2)))
    m = _DMY_RE.search(text)
    if m:
        mon = _MONTHS[m.group(2).lower()]
        return _safe_date(int(m.group(3)), mon, int(m.group(1)))
    m = _ISO_DATE_RE.search(text)
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def _safe_date(y: int, mo: int, d: int) -> Optional[date]:
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def parse_query_date_anchor(query: str) -> Optional[Tuple[date, date]]:
    """Return an inclusive ``(start, end)`` window for a date-scoped query, else None."""
    if not query:
        return None
    q = query.strip()
    ql = q.lower()

    # 1. Explicit range: "between X and Y"
    mb = _BETWEEN_RE.search(q)
    if mb:
        left, right = mb.group(1), mb.group(2)
        d2 = _single_date(right)
        # the left side often omits the year ("between August 11 and August 15
        # 2023"): borrow the year (and month, if missing) from the right side.
        d1 = _single_date(left)
        if d1 is None and d2 is not None:
            ml = _MDY_RE.search(left) or _DMY_RE.search(left)
            if ml:
                if ml.re is _MDY_RE:
                    mon, day = _MONTHS[ml.group(1).lower()], int(ml.group(2))
                else:
                    mon, day = _MONTHS[ml.group(2).lower()], int(ml.group(1))
                d1 = _safe_date(d2.year, mon, day)
            else:
                # bare day number on the left ("between the 11th and August 15 2023")
                dm = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", left)
                if dm:
                    d1 = _safe_date(d2.year, d2.month, int(dm.group(1)))
        if d1 and d2:
            return (min(d1, d2), max(d1, d2))

    # 2. "week before <date>" / "weeks before <date>"
    mw = re.search(r"\b(\w+\s+)?weeks?\s+(?:before|prior to|preceding)\b(.+)", ql)
    if mw:
        anchor = _single_date(mw.group(2)) or _single_date(q)
        if anchor:
            n_weeks = 1
            qual = (mw.group(1) or "").strip()
            if qual in ("two", "2"):
                n_weeks = 2
            elif qual in ("three", "3"):
                n_weeks = 3
            start = anchor - timedelta(days=7 * n_weeks)
            return (start, anchor - timedelta(days=1))

    # 3. Modifier + "Month YYYY": "last two weeks of August 2023",
    #    "last week of August 2023", "first week/weekend of August 2023",
    #    "second week of August 2023".
    mmy = _MY_RE.search(q)
    if mmy:
        mon = _MONTHS[mmy.group(1).lower()]
        year = int(mmy.group(2))
        m_start, m_end = _month_bounds(year, mon)
        # look at the words immediately preceding "<Month> <year>"
        pre = ql[: mmy.start()]
        # last N weeks of
        m_last = re.search(r"\blast\s+(two|three|2|3)?\s*weeks?\s+of\s*$", pre)
        if re.search(r"\blast\s+week\s+of\s*$", pre):
            return (max(m_start, m_end - timedelta(days=6)), m_end)
        if m_last:
            n = m_last.group(1)
            days = 14 if n in ("two", "2") else 21 if n in ("three", "3") else 7
            return (max(m_start, m_end - timedelta(days=days - 1)), m_end)
        if re.search(r"\bfirst\s+(week(end)?|few\s+days)\s+of\s*$", pre):
            return (m_start, min(m_end, m_start + timedelta(days=6)))
        if re.search(r"\bsecond\s+week\s+of\s*$", pre):
            return (m_start + timedelta(days=7), min(m_end, m_start + timedelta(days=13)))
        if re.search(r"\bthird\s+week\s+of\s*$", pre):
            return (m_start + timedelta(days=14), min(m_end, m_start + timedelta(days=20)))
        # "first/last weekend of" without "week"
        if re.search(r"\bfirst\s+weekend\s+of\s*$", pre):
            return (m_start, min(m_end, m_start + timedelta(days=6)))
        if re.search(r"\blast\s+weekend\s+of\s*$", pre):
            return (max(m_start, m_end - timedelta(days=6)), m_end)

    # 4. Explicit single calendar date ("on 1 May 2022", "October 24, 2023").
    #    Only when the query names a day — a bare "Month YYYY" is handled in (5).
    if _MDY_RE.search(q) or _DMY_RE.search(q):
        d1 = _single_date(q)
        if d1:
            return (d1, d1)

    # 5. Bare "Month YYYY" ("in October 2023", "as of September 2023") -> whole month.
    if mmy:
        mon = _MONTHS[mmy.group(1).lower()]
        year = int(mmy.group(2))
        return _month_bounds(year, mon)

    # 6. Bare ISO date anywhere.
    m = _ISO_DATE_RE.search(q)
    if m:
        d1 = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d1:
            return (d1, d1)

    return None


def node_dates(text: Optional[str], created_at: Optional[datetime]) -> list[date]:
    """All candidate dates a memory carries: ISO literals in ``text`` + created_at."""
    out: list[date] = []
    if text:
        for m in _ISO_DATE_RE.finditer(text):
            d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if d:
                out.append(d)
    if created_at is not None:
        try:
            out.append(created_at.date())
        except (AttributeError, ValueError):
            pass
    return out


def node_matches_anchor(
    text: Optional[str], created_at: Optional[datetime], anchor: Tuple[date, date]
) -> bool:
    """True iff any of the memory's candidate dates lies within ``anchor``."""
    start, end = anchor
    for d in node_dates(text, created_at):
        if start <= d <= end:
            return True
    return False
