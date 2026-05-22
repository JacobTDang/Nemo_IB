---
name: red-team-thesis
description: Attack an investment thesis before capital commitment. Restates the thesis, builds the strongest bear case, surfaces what smart shorts would argue, identifies what consensus may already know, and recommends keep / reduce confidence / reduce size / no_position. Use whenever a /equity-deep-research synthesis is about to be acted on or when sizing up. Distinct from /premortem (which writes 3 explicit failure scenarios) in that this attacks the thesis logic itself rather than mapping failure modes.
---

# /red-team-thesis — Adversarial review

The discipline: write the strongest case AGAINST your own thesis
before sizing into it. This is not /premortem — premortem writes
failure scenarios; red-team attacks the bull-case logic itself.

If the red-team argument turns out stronger than the original bull
thesis, the hard rule is to downgrade confidence or output
no_position. Letting the bull thesis stand unchallenged is how
analysts overpay.

## Inputs

- The drafted thesis (one sentence)
- The bull case (3-5 cited points from /equity-deep-research)
- The bear case (3-5 cited points)
- Confidence rating from the analyst
- Variant perception claim (what the analyst thinks consensus is missing)

## Workflow

### 1. Restate the thesis (verbatim)

Quote the thesis sentence exactly as written. This forces accurate
framing — many "weak red teams" attack a strawman version of the
thesis instead of what the analyst actually claimed.

### 2. Build the strongest bear case

Independently of the existing bear case in the synthesis, construct
the strongest argument against the thesis. Use the same data already
gathered (do NOT pull new data; the discipline is "given the same
evidence, can you tell a different story").

Required: at least 3 specific points, each grounded in a tool output
already cited in the synthesis. Specificity matters — "weak business
model" is not a red-team point; "operating margin compressed 280bps
over the last 4 quarters per get_segment_financials" is.

### 3. What would smart shorts argue?

Look at the actual short interest setup from /equity-deep-research
Step 11. If short interest > 15% of float, the bears have already
done work — what's their thesis?

If `get_schedule_13d_filings` returned activist filers, consider their
likely argument (debt restructuring activists vs operational
activists frame differently).

### 4. What does consensus already understand?

The variant perception fails if consensus already knows the same
thing. Test the variant perception claim:
- Is the data analyst-cited actually available in sell-side notes?
- Is the multiple already cheap because the market already discounted
  this risk?
- Are insiders / activists / specialists already positioned ahead of
  the variant view?

Output a yes/no judgment with one sentence of reasoning: "Consensus
already understands X" or "Consensus has not yet repriced for X
because Y."

### 5. Where is the thesis overconfident?

For each bull-case point in the synthesis, ask:
- Is the analyst extrapolating beyond evidence?
- Is the analyst assuming a structural break that hasn't shown up in
  data yet?
- Is the analyst using a peer comp that doesn't actually apply?
- Is the analyst conflating company-wide trends with the specific
  segment where the thesis lives?

Identify the single bull-case point most prone to overconfidence.

### 6. Embarrassing data

Identify the SINGLE observation that would most embarrass the
thesis. Examples:
- "If next quarter's revenue prints below $X, the entire
  ramp narrative breaks."
- "If CFO is replaced in the next 90 days, the going-concern
  resolution looks premature."
- "If competitor Y launches a competing product before the analyst's
  primary product, the entire moat argument fails."

This often surfaces a falsifier the analyst didn't include.

### 7. Recommendation

Choose one:
- **keep** — bull case still stronger than red-team bear; size as
  planned
- **reduce confidence** — bear arguments are real but don't break the
  thesis; lower confidence by 0.05-0.10 and add the red-team points
  to falsifiers
- **reduce size** — bear case has nontrivial probability; cut planned
  size by 50%
- **no_position** — red-team argument is structurally stronger than
  bull case; do not initiate

## Output

```
## /red-team-thesis — {TICKER}

**Restated thesis** (verbatim):
"[exact thesis sentence]"

**Strongest bear case** (independent of synthesis bear case):
1. ...
2. ...
3. ...

**Smart short argument**:
[what does the short side actually believe? grounded in 13F/13D/short
interest data already cited]

**Consensus assessment**:
[has consensus already repriced? yes/no, with one-sentence reasoning]

**Overconfidence risk**:
The bull-case point most prone to overconfidence is: [...]
Reason: [...]

**Embarrassing data point** (single observation that would force a
rewrite):
[...]

**Recommendation**: keep / reduce confidence / reduce size / no_position

**Reasoning** (one paragraph):
[why this recommendation given the above]

**Falsifiers to add**:
- [any new falsifiers surfaced by the red-team that weren't in the
  original list]
```

## Hard rules

- Do not pull new MCP data. The red-team works with the SAME evidence
  the bull thesis used. The discipline is "tell a different story
  given the same facts."
- Do not attack a strawman. Quote the thesis verbatim before
  attacking it.
- If the red-team bear case is stronger than the original bull case,
  the recommendation MUST be `no_position` or `reduce size`. Do not
  rationalize the bull case.
- "Smart short argument" must be grounded in actual positioning data
  (13F, 13D, short interest). If there is no smart short on the
  ticker, say so explicitly — and consider whether the absence of
  shorts is itself a signal (under-followed name? hard to borrow?
  crowded long?).
- Do not output `keep` if you couldn't identify at least one
  overconfidence risk and one embarrassing data point. Every thesis
  has both; failure to find them means the red-team didn't try.

## Save output

If invoked standalone:
`testing/fixtures/redteam_<TICKER>_<DATE>.md`

If invoked from /equity-deep-research, return the structured output
inline so the caller can fold it into Step 17 of the synthesis. The
caller MUST honor the recommendation (do not silently override).
