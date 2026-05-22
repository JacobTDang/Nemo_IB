---
name: postmortem
description: Post-mortem analysis after a position closes. Forces the analyst to write what went wrong (or right but for the wrong reason) and add the lesson to the historical analogues catalog so future research benefits. Use after a position is stopped out, closed at a loss, hit target, or after any thesis closes. Skip for open positions (use /premortem instead).
---

# /postmortem — Learn from the trade

After every closed position — winner or loser — write a structured
postmortem. The goal is NOT to feel good about wins or bad about
losses. The goal is to update the priors used in future research.

## Inputs

- The ticker
- The original thesis (from `latest_thesis(ticker)`)
- The evolution log (from `get_thesis_evolution(thesis_id)`)
- The realized outcome: entry, exit, return %, holding period
- Optional: the alert log from the falsifier watcher (`falsifier_alerts` table)

## The 4 questions

Answer each. Do NOT skip.

1. **Was the thesis correct?** Compare the original assumptions to what
   actually happened. Both can be true:
   - Right thesis, wrong execution (e.g., correct on AI capex acceleration
     but exited 6 months too early)
   - Wrong thesis, right outcome (i.e., right answer for the wrong reason —
     this is the most dangerous category because it confirms bad process)

2. **Were the falsifiers well-chosen?** Pull the original falsifiers from
   the thesis row. Did any trigger? If yes, did the analyst exit when
   they did? If no, were they too lax (never could have fired) or were
   they just lucky?

3. **What changed conviction over time?** Read the evolution log. Look
   for: (a) overreaction to noise, (b) anchoring to entry conviction,
   (c) missing critical data points the falsifier watcher caught.

4. **What's the pattern for next time?** This is the load-bearing
   answer — what is the generalizable lesson, and what historical
   analogue does this trade now belong to?

## Output structure

```
## /postmortem — {TICKER}

**Position summary**: <side> <qty> @ <entry> -> <exit> = <return %>
   over <hold period>
**Original thesis**: <one sentence from thesis row>
**Original confidence**: <0.0-1.0> | **Final confidence**: <0.0-1.0>

### 1. Was the thesis correct?
<honest analysis. "I was right about X, wrong about Y">

### 2. Falsifier quality
<which falsifiers triggered, which didn't, were the right ones in place>

### 3. Conviction trajectory
<key entries from evolution log — what shifted, what should have shifted>

### 4. Pattern + lesson
<the generalizable insight, tagged with structural keywords>

### Update to analogues catalog
<one-paragraph entry to add to knowledge/analogues.md, with appropriate
 structural tags>
```

## Hard rules

- Distinguish "I was right" from "I made money." A losing trade with
  correct process is better than a winning trade with bad process; the
  latter is sample-size luck that will reverse.
- The lesson must be a STRUCTURAL claim, not a ticker-specific one.
  "MSFT was hard to short" is not a lesson; "Mega-cap names with >5%
  buyback yield are hard to short for >6mo regardless of fundamentals"
  IS a lesson.
- Save the postmortem to `testing/fixtures/postmortem_<TICKER>_<DATE>.md`
- Append the new analogue entry to `knowledge/analogues.md` so future
  /deep-research runs benefit.
- After completion, the user is reminded that the next research using
  `get_historical_analogue` will now match against this new entry.

## When to invoke

- After `close_paper_position` returns success
- When the user says "review the NVDA trade" or "what did we learn from
  X"
- Periodically on a sample of older closed positions (lessons take
  multiple iterations to surface)

## When to skip

- For open positions — use /premortem or /deep-research
- Within 1 trading day of a flash move (give time for the pattern to
  reveal itself)
