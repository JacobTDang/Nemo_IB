# Replay: entering at the 8-K instead of the XBRL filing

Run 2026-09-02/03 against the replay-data snapshot. Everything below is
paper-traded over history with point-in-time reads; nothing here is a live
result.

## The question

The XBRL-timed arm (`SIGNAL_VARIANT=ts`) showed no measurable edge: 611
trades, median -36 bps, a drift coefficient whose interval contains zero.
The consensus arm (`cs`) showed a sign but not a size. The two arms differ
in when they know the print: `cs` knows it the day the company announces,
`ts` knows it when the XBRL financials are filed, which for a small filer is
weeks later. Is the missing edge a timing artefact?

`ts_release` answers that directly. It reads diluted EPS from the 8-K
EX-99.1 earnings release the day it is filed, substitutes that one quarter
into the XBRL series, and computes the same standardised surprise as `ts`.
Same signal, same names, same book rules, only `known_at` differs.

## Setup

| | |
|---|---|
| decision dates | 230 sessions, 2025-09-02 .. 2026-07-31 |
| roster | 1,022 issuers, one ticker per CIK, from 1,267 tickers the XBRL arm signalled |
| horizon | 20 sessions, spread window 252 |
| sides | long only; the snapshot carries no borrow rates and the scanner refuses unpriced shorts |
| comparisons | 5, so the Bonferroni t threshold is 2.58 |
| consensus | stashed for the run so the scanner considered every eligible name, restored after |

Three arms were run from one signal build:

- **release as extracted** -- every figure the reader produced. This is what would deploy.
- **release agreeing only** -- restricted to reads that match the later XBRL value within 2.5%. An upper bound with the reader's noise removed; lookahead as a strategy, honest as a bound.
- **xbrl, same prints** -- the XBRL-timed arm already on disk, restricted to the (ticker, quarter) prints the release arm traded. The clean A/B.

## Results

| arm | N | mean net bps | median | hit | t | 95% CI of mean | drift per sigma | drift CI |
|---|---|---|---|---|---|---|---|---|
| ts (XBRL-timed), full | 611 | +15.2 | -35.5 | 48% | 0.39 | [-62, +92] | +16.1 | [-17, +49] |
| cs (consensus), full | 213 | +121.6 | -10.6 | 49% | 1.87 | [-6, +249] | +147.5 per tail | [+3, +292] |
| release as extracted | 392 | +89.7 | -0.6 | 50% | 1.82 | [-7, +187] | +34.0 | [-6, +74] |
| release agreeing only | 350 | +89.9 | +1.4 | 50% | 1.78 | [-9, +189] | +27.9 | [-16, +71] |
| xbrl, same prints | 348 | +18.0 | -54.8 | 46% | 0.34 | [-85, +121] | +16.8 | [-28, +62] |

No arm clears the Bonferroni threshold on its own. The release arm's
coefficient is a sign, not a size: about 2,213 trades at this dispersion
would pin it to within 50%.

The reader's noise costs nothing measurable: the agreeing-only arm has the
same mean as the as-extracted arm on 42 fewer trades.

### The paired A/B

348 prints were traded by both the release arm and the XBRL arm, all on the
same side. Release minus XBRL, per print:

| | |
|---|---|
| mean gain | +61.2 bps |
| median gain | 0.0 |
| 95% CI | [+12.5, +110.0] |
| t | 2.46 |

Significant unadjusted, just under the 2.58 threshold adjusted. The median
of zero says the gain lives in a subset, and splitting by how many days the
release preceded the XBRL filing shows which:

| release earlier by | N | release mean | XBRL mean | gain mean | gain median | gain CI | t |
|---|---|---|---|---|---|---|---|
| 0-1 days | 235 | +93.0 | +89.2 | +3.8 | 0.0 | [-28, +36] | 0.24 |
| 2-7 days | 35 | +47.0 | -62.5 | +109.5 | +107.3 | [-98, +317] | 1.03 |
| 8+ days | 78 | +52.4 | -160.1 | +212.5 | +102.9 | [+45, +380] | 2.48 |

Where the two entries happen on the same day, they earn the same. Where the
XBRL entry is a week or more late, it loses 160 bps a trade on prints the
release entry makes +52 on. That is a dose response, and it is the strongest
support 348 pairs can give the timing hypothesis: the XBRL arm's null was the
late entries losing, not the signal failing.

Most prints do not have the gap. Across 13,855 matched prints the median
release precedes its XBRL filing by one day (p10 zero, p90 seventeen);
large filers lodge the 10-Q with the release. The gain is concentrated in
the tail of small filers with a long gap.

### By announcement timing

Release arm, absolute returns:

| timing | N | mean | median | hit | t (threshold 2.94 at 15 comparisons) |
|---|---|---|---|---|---|
| after close | 194 | +155.8 | +27.4 | 52% | 2.23 |
| before open | 190 | +25.5 | -14.3 | 48% | 0.35 |
| during hours | 8 | +13.3 | -129.1 | 38% | 0.08 |

The paired gain over XBRL is larger for before-open releases (+78, CI
[+25, +131], t 2.90) than after-close (+46, t 1.13). A secondary cut; noted,
not acted on.

## The reader

`research/release_eps.py`, on the 1,022-name roster:

| | |
|---|---|
| releases examined | 24,308 |
| figure read and agrees with later XBRL | 12,926 |
| figure read and disagrees | 2,466 |
| refused (no figure) | 8,916 |
| agreement, of reads | 84% |
| read rate, of releases | 63% |
| names where every release was refused | 102 of 995 |

On the nine names used to build the reader (AEIS, AGNC, AFL, AA, A, PLAB,
KO, WFC, MSFT) it reads 97% and agrees on 86%. The roster is harder. Why it
refused, most common first:

| reason | releases |
|---|---|
| a diluted per-share phrase, no dollar figure beside it | 2,721 |
| no diluted per-share figure named in the release | 2,445 |
| no diluted per-share table row with a period header | 2,037 |
| every diluted figure qualified as non-GAAP or adjusted | 927 |
| the 8-K lists documents and none is an EX-99 exhibit | 747 |
| fetch failed, cache full (issue #98) | 18 |
| ambiguous table mapping | 8 |
| no release text | 7 |

A refusal is a print the release arm does not trade until the XBRL arrives
(`sue_ts_release` falls back to the XBRL series), so the read rate bounds
how much of the timing gain a deployment captures. The 18 cache failures
are releases lost to the tmpfs incident, not to the reader.

## Caveats

- 392 trades. The release arm's own interval contains zero; only the paired difference, and within it only the long-gap bucket, is significant.
- Long only. The snapshot has no borrow rates, so every short was refused across every arm. Whether the timing gain holds on the short side is untested.
- One year of decision dates, all in 2025-09 .. 2026-07.
- The agreeing-only arm uses the later XBRL value to select reads, which is lookahead. It is a bound on the reader, not a strategy.
- The roster is the XBRL arm's signalled names, so it inherits that arm's universe and screen; the CIK collapse (#91) was applied in the driver, not by the production screen at the time of the run.
- The run was interrupted twice by the host sleeping and once by the cache filling; the signal build resumed from per-name checkpoints each time, and the 126 names that had failed on the full cache were re-read from scratch.

## What it means for deploying `ts_release`

The nightly scanner only asks names with a print recorded in the signal
window, so a release-timed scan reads a few dozen 8-K histories a night,
not the universe. Cold, the reader averaged 13.5 seconds a name across the
roster against SEC; warm, a full 33-release history re-reads in 6 seconds
and the nightly call (`sue_ts_release`) answers in 1.6. The filing cache
now bounds itself by what the filesystem charges (#98), and the whole
roster's cache measured 166MB, so a warm cache fits under the production
cap.

Switching is one setting, `SIGNAL_VARIANT=ts_release`, and the paper book
records the variant on every order so a book spanning the change stays
auditable. Forward paper trading is the out-of-sample test this replay
cannot provide.

## Raw output

The signal build, the per-release audit with every read and its reason,
and the three arms' full results (`replay_release_*.json`,
`replay_xbrl_same_prints.json`) were produced by a driver run inside the
image against the snapshot volume; they are not committed. The numbers
above were computed from those files by a one-off comparison script.
