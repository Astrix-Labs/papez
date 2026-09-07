# How scoring works (and why nothing gets deleted at 0.005)

This document exists because the score is easy to misread. If you take one thing
away:

> **`decay_score` is an activation / retention weight, not a countdown to
> deletion.** Retrieving a memory *raises* it. A low score does **not** mean a
> memory is about to be forgotten — forgetting has separate, conjunctive
> conditions that a low score alone never satisfies.

Field feedback from a ~50-node real workload put it well: the most-recalled
nodes carried the highest scores (`0.43`, `0.21`); a written-once,
never-recalled node sat at `~0.005`; and **nothing had been deleted.** That is
the system working as designed. The rest of this doc explains why.

---

## The name

The field name is `decay_score` for backward compatibility, but every recall
hit and `memory_explain` response now also carries an **`activation`** alias
with the identical value. Prefer reading `activation`: it says what the number
means. High activation = strongly retained and easy to surface. Low activation =
resting quietly, still there.

```jsonc
// a recall hit
{
  "id": "…",
  "decay_score": 0.43,   // legacy name
  "activation": 0.43,    // same value, clearer name — read this one
  "score": 0.71          // query-time relevance of THIS hit (see below)
}
```

Note `score` (in a recall result) is a different number: it is the query-time
relevance of that hit to *your current query*. `activation` is the memory's
standing retention weight independent of any one query.

---

## The three-force formula

```
decay_score = relevance × connectivity_factor × activation_factor
```

The formula is multiplicative on purpose. A memory has to earn its keep on
**all three** axes; a zero on any one collapses the product. (See
`engine/scoring.py`, `calculate_decay_score`.)

### Force 1 — Relevance

How well the memory matches the current context.

- **At recall time**, relevance is a hybrid of vector similarity and entity
  overlap: `0.7 × cosine + 0.3 × keyword_overlap`
  (`RELEVANCE_VECTOR_WEIGHT` / `RELEVANCE_KEYWORD_WEIGHT`).
- **At rest** (no query context — e.g. when a background worker recomputes the
  stored score), relevance falls back to a recency term:
  `max(0.1, 1 − days_since_access / 365)`. So an untouched memory's relevance
  floats down slowly toward `0.1` over a year, never to zero.

### Force 2 — Connectivity

How embedded the memory is in the causal graph.

```
raw = log2(1 + causal_weight) / log2(1 + max_causal_weight)
connectivity_factor = raw²
```

- Squaring rewards genuine hubs and punishes barely-connected nodes.
- Clamped to a floor of `MIN_CONNECTIVITY` (0.1) for any connected node, so a
  single good edge keeps a memory alive.
- **An orphan (zero supportive edges) gets `connectivity_factor = 0`** — which
  drives the whole product toward zero. This is the only force that can hard-zero
  a memory, and it is exactly the state forgetting looks for.
- "Supportive" here excludes `CONTRADICTS` and `SUPERSEDES`: a node whose only
  edges are contradictions/supersessions is treated as an orphan (see the
  changelog note on orphan semantics).

### Force 3 — Activation (ACT-R base-level)

How recently and how often the memory has been reactivated. This is the force
the reviewer saw rising as they recalled things.

```
B_i = ln( Σ_j  t_j^(−d) )          # t_j = seconds since the j-th access, d = 0.5
activation_factor = clamp( e^(B_i), 0, 1 )
```

- Every recall appends a fresh timestamp, so the sum grows and recent accesses
  dominate (recent `t_j` are small, and small numbers raised to `−0.5` are
  large). **This is why retrieval raises the score.**
- With no reactivations, the memory has a single timestamp (its creation), and
  `activation_factor` decays smoothly with age — this is the term that pushes a
  write-once memory down to `~0.005`.

---

## Worked numbers (the 0.43 / 0.21 / 0.005 the field saw)

These are illustrative decompositions consistent with the formula, matching the
figures reported from the real workload.

| Memory | relevance | connectivity | activation | **decay_score** | What it is |
|---|---|---|---|---|---|
| A — most recalled | 0.88 | 0.62 | 0.79 | **≈ 0.43** | Frequently retrieved, well-linked hub |
| B — often recalled | 0.80 | 0.55 | 0.48 | **≈ 0.21** | Retrieved regularly, decent connectivity |
| C — write-once, never recalled | 0.60 | 0.10 | 0.083 | **≈ 0.005** | Stored once, still linked, never surfaced |

`A`: `0.88 × 0.62 × 0.79 ≈ 0.431`
`B`: `0.80 × 0.55 × 0.48 ≈ 0.211`
`C`: `0.60 × 0.10 × 0.083 ≈ 0.005`

The gap between A/B and C is driven almost entirely by **Force 3**: A and B keep
getting recalled, so their activation stays high; C has one lonely timestamp, so
its activation term is tiny. Connectivity widens the gap a little; relevance
barely moves it. **Nothing about C's 0.005 means "about to be deleted"** — see
the next section.

---

## `stability` — the spaced-repetition dial

`stability` is a separate, monotonically increasing field that models the
spacing effect: memories reinforced across *spread-out* sessions become more
durable than memories crammed in one burst. Each reactivation nudges it up by
`0.1 / stability` — so early reinforcement moves the needle a lot and later
reinforcement moves it less (diminishing returns, exactly like real spaced
repetition). Stability feeds core-memory promotion (`CORE_STABILITY_WEIGHT`) and
tempers how fast a memory can slide toward dormancy. It is reported alongside the
score in `memory_explain`; it is **not** one of the three multiplicative forces.

---

## Status and pinning override the score

The score is an input to lifecycle decisions, not the decision itself. Two things
outrank it:

1. **Status.** A memory's `status` (`ACTIVE → EPISODIC → DORMANT`, plus `CORE`)
   is its lifecycle layer. `CORE` memories are structurally important — they are
   auto-pinned and never pruned regardless of how their score moves.
2. **Pinning.** `pin_memory` sets `pinned = True` and promotes to `CORE`. A
   pinned memory is exempt from forgetting **at any score**, including 0.

So the effective retention rule is: **status and pin flags decide survival;
`decay_score` only decides ordering and eligibility among the memories that are
even up for pruning.**

---

## Forgetting is conjunctive — this is why 0.005 is safe

A memory is pruned **only if every one of these is simultaneously true**
(`engine/forgetting.py`; rule 5 in `CLAUDE.md`):

```
decay_score < 0.01           (config.FORGETTING_THRESHOLD)
AND is_orphan                (zero supportive edges)
AND NOT pinned
AND status != CORE
AND visibility != ORG
```

Return to memory **C** at `0.005`. Its score *is* below the `0.01` threshold —
the first condition is met. It still is not deleted, because it fails the second:
it has a supportive edge, so `is_orphan` is false. A low score is a *necessary*
condition for forgetting, never a *sufficient* one.

This is the whole point of conjunctive forgetting: a memory has to be
**irrelevant AND causally orphaned AND unpinned AND non-core AND non-org** all at
once. Low activation on its own just means "resting" — the memory sits quietly at
a low score, fully retrievable, until it either gets recalled (activation climbs
again) or genuinely loses all its connections.

---

## Reading a live breakdown with `memory_explain`

`memory_explain` returns a `score_model` block so you never have to reverse-
engineer this again. It includes:

- `formula` — the three-force equation.
- `reading` — a plain-language reminder that higher = more retained and that
  retrieval raises the score.
- `forces.connectivity_factor` and `forces.activation_factor` — **computed live**
  from the node's current edges and reactivation history, so they reflect reality
  now rather than the last time a background worker recomputed the stored
  `decay_score`.
- `staleness_note` — a caveat that the *stored* `decay_score` is refreshed by the
  transitions worker (in hosted deployments), whereas the live forces above are
  fresh.

`relevance` is intentionally *not* given a live number: it is query-dependent and
only contributes at recall time, so there is no single at-rest value for it.

---

## TL;DR for agents

- Read **`activation`**, not `decay_score` (same value, honest name).
- Higher = more strongly retained. **Recalling a memory raises its score.**
- A low score means "resting," not "doomed." Deletion needs low score **and**
  orphaned **and** unpinned **and** non-core **and** non-org, all together.
- Use `memory_explain` → `score_model` when you want the live per-force
  breakdown.
- `stability` and `status`/`pinned` are separate levers; pinning beats any score.

## Why forgetting needs an idle period

An orphan has no supportive edges, so its connectivity factor is zero and its decay score is zero the moment it is rescored. If the score alone gated deletion, every unlinked memory would be pruned on the next maintenance pass. Forgetting therefore also requires that the memory has not been stored, recalled or reactivated for `GENESYS_FORGETTING_MIN_IDLE_DAYS` (default 30). That is what makes "irrelevant" a real condition rather than a side effect of "orphaned".
