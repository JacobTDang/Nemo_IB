---
name: cross-company-readthrough
description: Given an event at Company A (earnings beat/miss, guidance change, product launch, capacity expansion, regulatory action, etc.), identify first-order and second-order beneficiaries and losers across suppliers, customers, competitors, substitutes, and adjacent industries. Produces a probabilistic prediction of the read-through and flags where the market may not yet have connected the dots. Invoke from /equity-deep-research Step 14, or standalone whenever the user says "what does X's news mean for Y" / "who else is affected" / "read-through from."
---

# /cross-company-readthrough — Connecting the graph

The handoff doc flagged this as the single highest-edge companion
skill. The edge is not finding news. Everyone has news. The edge is
connecting:

```
Company event → supplier/customer/competitor graph → second-order
implication → market awareness gap → probabilistic prediction
```

## Inputs

- Event description (free text, e.g., "TSMC says advanced packaging
  capacity remains supply constrained through 2027")
- Primary affected ticker (the company whose news this is)
- Event type if known (earnings_beat, guidance_raise,
  capacity_expansion, supply_constraint, contract_win, etc.)

If invoked from /equity-deep-research Step 14, the event description
is the thesis being researched and the primary ticker is the analyst's
focus name.

## Workflow

### 1. Parse the event

Restate the event in one factual sentence. No editorializing.

Identify:
- Event type (from a controlled list — see "Common event types"
  below)
- Magnitude: small / medium / large / regime-changing
- Time horizon of the impact: next quarter / next year / next 3 years
- Directional bias for the primary: positive / negative / mixed

### 2. Map the relationship graph

For the primary ticker:
- `get_supply_chain(primary)` — surfaces suppliers, customers,
  competitors, substitutes from 10-K Item 1 text
- `get_company_peers(primary)` — direct industry peers
- `get_industry_etfs(theme)` — adjacent names via ETF co-membership

Build a node list with each name's classified relationship to the
primary:
- supplier
- customer
- competitor
- substitute (different product, same buyer)
- adjacent (different industry, correlated through theme)
- complement (a name whose product is consumed together with the
  primary's)

### 3. First-order read-through

For each direct relationship (suppliers, customers, direct
competitors, direct substitutes), classify the impact:

| Name | Relationship | Impact direction | Reasoning |
|------|--------------|------------------|-----------|
| ... | ... | positive / negative / mixed | one-sentence why |

Aim for 4-8 first-order names. Be specific in reasoning — "is a
supplier so benefits" is weak; "supplies HBM3e where 65% of revenue
is concentrated in the primary's product family" is strong.

### 4. Second-order read-through

This is where the edge lives. For each first-order name, ask: who is
THAT company's supplier / customer / competitor?

Examples:
- Primary = NVDA. First-order = TSMC (supplier). Second-order =
  ASML, Disco, KLA (TSMC's equipment suppliers).
- Primary = MSFT. First-order = NVDA (supplier of accelerators).
  Second-order = SK Hynix (HBM supplier to NVDA), Amphenol
  (interconnect into NVDA boards).

Cap second-order at 3-6 names. More than that is noise.

Output each as:
| Name | Path to primary | Impact direction | Reasoning |

### 5. Market interpretation likelihood

For each first- and second-order name, estimate:
- Will the market connect this read-through within: 5 trading days /
  30 days / 90 days / longer?
- Is the read-through already priced in (sell-side notes likely
  highlighting it) or under-appreciated (requires multi-hop reasoning)?

Flag the names where:
- Impact is medium-to-large
- AND market awareness is low (second-order or non-obvious first-order)

These are where the actual edge lives.

### 6. Underappreciated implication

Identify the SINGLE most underappreciated implication. State explicitly:
- The name(s) affected
- The mechanism (why this implication exists)
- Why the market may miss it (multi-hop reasoning, unrelated sector,
  small market cap, etc.)
- The expected time horizon for re-pricing

### 7. Probabilistic prediction

Make a probabilistic prediction with confidence in [0,1]:

> "[Event] increases the probability that [affected ticker] will
> [outcome] within [time horizon], with confidence [0-1]. The market
> may miss this because [reason]."

If confidence is below 0.50, do NOT make the prediction — instead
say "evidence does not support a directional read-through."

### 8. Data needed to confirm

List 2-4 specific observables that would confirm (or disconfirm) the
prediction. Examples:
- "[Affected ticker] reports advanced packaging revenue growth above
  X% next quarter"
- "[Adjacent name] guides capex up by Y% on next earnings call"
- "[Substitute name] discloses pricing pressure in 10-Q MD&A"

## Output

```
## /cross-company-readthrough

**Event** (one sentence):
[restated event]

**Primary affected ticker**: {TICKER}
**Event type**: {type}
**Magnitude**: small / medium / large / regime-changing
**Time horizon**: {next quarter / year / 3-year}
**Primary directional bias**: positive / negative / mixed

**First-order read-through**:
| Name | Relationship | Impact | Reasoning |
|------|--------------|--------|-----------|
| ... | ... | ... | ... |

**Second-order read-through**:
| Name | Path | Impact | Reasoning |
|------|------|--------|-----------|
| ... | ... | ... | ... |

**Where the edge lives** (medium-large impact + low market awareness):
- [name]: [why]
- [name]: [why]

**Underappreciated implication**:
[paragraph]

**Probabilistic prediction**:
[prediction sentence with confidence 0.0-1.0]

**Data needed to confirm**:
- [observable 1]
- [observable 2]
- [observable 3]

**Data gaps**:
- [any relationship that couldn't be classified due to missing supply
  chain data]
```

## Common event types

```
earnings_beat            earnings_miss
guidance_raise           guidance_cut
capacity_expansion       capacity_delay
pricing_increase         pricing_pressure
product_launch           product_delay
regulatory_approval      regulatory_risk
contract_win             contract_loss
customer_churn           supplier_constraint
inventory_build          margin_pressure
management_change        insider_buying
insider_selling          activist_position
short_interest_spike     estimate_revision
litigation_update        macro_shock
```

## Hard rules

- Specificity is the discipline. "Is a supplier so benefits" is not a
  read-through; it's noise. Always cite the mechanism (revenue
  concentration %, product family, contract structure, etc.).
- Cap first-order at 4-8 names and second-order at 3-6. The point is
  the strongest reads, not exhaustive coverage.
- If `get_supply_chain` returns empty (software / services 10-Ks often
  do), use `get_industry_etfs` co-holdings as a substitute and note
  the data substitution in `data_gaps`.
- Do not extend to third-order. If the chain requires that many hops,
  the signal is too weak.
- Probabilistic prediction confidence < 0.50 means no prediction.
  Better to surface "evidence inconclusive" than fabricate a
  probabilistic claim.
- For positive read-throughs to second-order names that are also in
  the primary's own ETF baskets (i.e., correlated by theme), note
  that some of the move is just beta — the read-through edge is the
  IDIOSYNCRATIC portion of the second-order impact.

## When to invoke

- User describes an event and asks "who else is affected"
- /equity-deep-research Step 14 (always)
- After a major earnings print where the read-through is obvious for
  the primary but unclear for the broader supplier/customer chain
- When a 13D / activist filing names a target — the read-through is
  often to peers in the same sub-sector

## Save output

If invoked standalone:
`testing/fixtures/readthrough_<EVENT_OR_TICKER>_<DATE>.md`

If invoked from /equity-deep-research, return the structured output
inline so the caller can fold it into Step 14 of the synthesis.
