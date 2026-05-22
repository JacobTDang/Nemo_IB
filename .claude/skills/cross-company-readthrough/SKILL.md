---
name: cross-company-readthrough
description: Given an event at Company A (earnings beat/miss, guidance change, product launch, capacity expansion, regulatory action, etc.), identify first-order and second-order beneficiaries and losers across suppliers, customers, competitors, substitutes, and adjacent industries. Produces a probabilistic prediction of the read-through and flags where the market may not yet have connected the dots. Invoke from /equity-deep-research Step 14, or standalone whenever the user says "what does X's news mean for Y" / "who else is affected" / "read-through from."
---

# /cross-company-readthrough

Edge comes from connecting: event → graph → second-order implication →
market awareness gap → probabilistic prediction. The news itself is
the least proprietary layer.

## Inputs

- Event description (free text)
- Primary affected ticker
- Event type if known (see Common event types below)

## Workflow

### 1. Parse the event

Restate in one factual sentence. Identify: event type, magnitude
(small / medium / large / regime-changing), time horizon
(next Q / next year / next 3 years), directional bias for primary
(positive / negative / mixed).

### 2. Map the graph

For the primary ticker, call: `get_supply_chain` (suppliers /
customers / competitors from 10-K Item 1), `get_company_peers`
(direct peers), `get_industry_etfs` (adjacent via ETF co-membership).
Classify each name as: supplier / customer / competitor / substitute
(different product, same buyer) / adjacent (different industry,
correlated through theme) / complement (consumed together).

### 3. First-order read-through

For each direct relationship, classify impact (positive / negative /
mixed) with one-sentence mechanism. Aim for 4-8 names. Specificity
matters: "is a supplier so benefits" is noise; "supplies HBM3e where
65% of revenue concentrates in the primary's product family" is signal.

### 4. Second-order read-through

For each first-order name, ask: who is THAT company's supplier /
customer / competitor? Cap at 3-6 names. This is where the edge lives.

Examples:
- NVDA → TSMC (first) → ASML, Disco, KLA (second)
- MSFT → NVDA (first) → SK Hynix, Amphenol (second)

### 5. Market interpretation likelihood

For each first- and second-order name, estimate when the market will
connect this (5 days / 30 days / 90 days / longer) and whether it's
already priced (sell-side notes likely cover it) or under-appreciated
(requires multi-hop reasoning). Flag the medium-to-large impact +
low awareness combinations — that's the edge.

### 6. Underappreciated implication

Identify the SINGLE most underappreciated implication: the names
affected, the mechanism, why the market may miss it, and the time
horizon for re-pricing.

### 7. Probabilistic prediction

`"[Event] increases the probability that [affected ticker] will
[outcome] within [horizon], with confidence [0.0-1.0]. The market may
miss this because [reason]."`

If confidence < 0.50, do NOT predict — say "evidence does not support
a directional read-through."

### 8. Data needed to confirm

List 2-4 specific observables that would confirm or disconfirm.

## Output

```yaml
---
skill: cross-company-readthrough
ticker: <primary or null if multi-name>
verdict: <strong_edge / moderate_edge / weak_signal / inconclusive>
confidence: 0.0-1.0
key_finding: <one sentence on the underappreciated implication>
data_gaps: [<list>]
---

**Event**: [restated]
**Primary**: {TICKER} | **Type**: {type} | **Magnitude**: {size} | **Horizon**: {time} | **Bias**: {dir}

**First-order**:
| Name | Relationship | Impact | Reasoning |

**Second-order**:
| Name | Path | Impact | Reasoning |

**Where the edge lives** (medium-large + low awareness):
- [name]: [why]

**Underappreciated implication**: [paragraph]

**Prediction**: [sentence with confidence]

**Data needed to confirm**:
- [observable 1-4]

**Data gaps**: [any]
```

## Common event types

`earnings_beat / earnings_miss / guidance_raise / guidance_cut /
capacity_expansion / capacity_delay / pricing_increase /
pricing_pressure / product_launch / product_delay /
regulatory_approval / regulatory_risk / contract_win / contract_loss /
customer_churn / supplier_constraint / inventory_build /
margin_pressure / management_change / insider_buying /
insider_selling / activist_position / short_interest_spike /
estimate_revision / litigation_update / macro_shock`

## Hard rules

- Specificity is the discipline. Cite the mechanism (revenue %,
  product family, contract structure) — never just "is a supplier."
- Cap first-order at 4-8, second-order at 3-6. Strongest reads, not
  exhaustive coverage.
- If `get_supply_chain` returns empty (common for software/services),
  fall back to `get_industry_etfs` co-holdings and note the data
  substitution in `data_gaps`.
- Do not extend to third-order. If chain requires that many hops,
  signal is too weak.
- Confidence < 0.50 means no prediction. "Evidence inconclusive"
  beats a fabricated probabilistic claim.
- For positive second-order names also in the primary's ETF baskets:
  note that some move is theme beta — the edge is the idiosyncratic
  portion above theme.

## When to invoke

`/equity-deep-research` Step 14 (always); user asks "who else is
affected" / "read-through from X"; after major earnings or 13D filing
where the primary's read-through is obvious but the supplier/customer
chain isn't. Standalone runs save to
`testing/fixtures/readthrough_<EVENT_OR_TICKER>_<DATE>.md`.
