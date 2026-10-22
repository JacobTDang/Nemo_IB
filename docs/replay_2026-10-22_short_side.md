# Replay: measuring the short side with flat borrow

Run 2026-10-22 against the replay-data snapshot. Everything below is
paper-traded over history with point-in-time reads; nothing here is a live
result.

## The question

In the September 2026 replay (`docs/replay_2026-09-03_release_timing.md`),
every arm traded long only. The snapshot volume carried no borrow rates, and
the scanner refuses an unpriced short rather than charging zero. This meant half
of the signal—negative earnings surprises—was completely unmeasured (issue #106).

If negative surprises show post-earnings announcement drift after borrow costs,
the book was leaving edge uncaptured. If borrow costs erase the drift, or if
negative surprises show no drift, a two-sided book pays carry without benefit.

This study measures the short side of the XBRL-timed (`ts`) arm across the same
230 decision dates as the September replay (2025-09-02 .. 2026-07-31).

## Setup

| | |
|---|---|
| decision dates | 230 sessions, 2025-09-02 .. 2026-07-31 |
| universe | 870 distinct tickers across the 230 sessions (803 eligible per session) |
| horizon | 20 sessions, spread window 252 |
| borrow rate | flat 3.0% annualised (`--borrow-rate 0.03`) for Arm 1; charges ~24.2 bp per 20-session hold |
| comparisons | 5, so the Bonferroni t threshold is 2.58 for the combined book; each side is one more split, so 10 and 2.81 per side |
| consensus | stashed for the run so the scanner considered every eligible name, restored after |

Two passes were run from the precomputed XBRL signal table:

- **Pass A (long-only baseline)**: `--borrow-rate None`. Unpriced shorts are refused, matching the September replay configuration.
- **Pass B (two-sided book)**: `--borrow-rate 0.03`. Every short is priced with a flat 3.0% annual borrow fee.

## Results

| arm / side | N | mean net bps | median | hit | t | 95% CI of mean | drift per sigma | drift CI |
|---|---|---|---|---|---|---|---|---|
| ts short side (Pass B) | 83 | +21.8 | +184.5 | 55.4% | 0.19 | [-207.5, +251.1] | -5.8 | [-83.1, +71.6] |
| ts long side (Pass B) | 542 | +0.4 | -47.1 | 46.9% | 0.01 | [-81.8, +82.5] | +7.8 | [-27.5, +43.1] |
| ts combined book (Pass B) | 625 | +3.2 | -35.5 | 48.0% | 0.08 | [-74.2, +80.6] | +5.1 | [-26.8, +37.1] |
| ts long-only baseline (Pass A) | 542 | +0.0 | -50.8 | 46.7% | 0.00 | [-82.2, +82.2] | +7.7 | [-27.6, +43.1] |
| September ts full reference | 611 | +15.2 | -35.5 | 48.0% | 0.39 | [-62.0, +92.0] | +16.1 | [-17.0, +49.0] |

### Long book comparison

The flat borrow rate on the short side did not distort the long book:
- Both Pass A and Pass B generated exactly 542 long trades.
- 538 of the 542 long trades are the same in both passes, with the same net return. The other four are the same names (INVH, RCL, TEVA, UBSI) entered one session later in Pass B, on dates when the two-sided book also took a short. Mean net +0.0 bp vs +0.4 bp, median -50.8 bp vs -47.1 bp, drift per sigma +7.7 bp vs +7.8 bp.
- Both agree with September's finding that the XBRL-timed long signal has no statistically significant edge (t = 0.01; 95% CI contains zero).
- The September count (611) is not comparable with this run's. Its XBRL arm signalled 1,267 tickers; this run's signal table covers the 870 tickers in the snapshot's universe rows for the window. Pass A is the like-for-like baseline.

### Short book analysis

83 short trades cleared the expected edge and cost gates:
- **Returns**: Mean net return was +21.8 bps, with a median of +184.5 bps and a hit rate of 55.4% (46 winning trades out of 83).
- **Financing cost**: Over the 20-session (~29 calendar day) holding period, shorts paid an average of 24.17 bps in borrow carry fees. Gross returns averaged +50.5 bps before borrow and execution costs.
- **Statistical significance**: With high dispersion (standard error of 117.0 bps), the t-statistic is +0.19, well below the per-side Bonferroni threshold of 2.81 (and below 2.00 unadjusted). The 95% confidence interval of the mean is wide: [-207.5, +251.1] bps.
- **Drift coefficient**: The estimated drift coefficient is -5.8 bps per sigma (95% CI [-83.1, +71.6]), which spans zero. Like the long side, the XBRL-timed short signal does not clear significance at this sample size.

## Arm 2: Real historical borrow rates

Arm 2 evaluated backfilling historical borrow rates using `python -m research.borrow --backfill`.

- **Module capability**: `research/borrow.py` accepts CSV files (`--from-csv`) with `ticker` and `annual_rate` columns, and timestamps rows at the close of `--as-of` when `--backfill` is specified.
- **Data availability**: The repository and current environment hold no historical borrow rate source. Borrow rates are not provided by free feeds (such as yfinance or SEC EDGAR).
- **Vendor policy**: Adding an external borrow data source (e.g., an institutional broker loan file or a paid vendor API) requires repository owner sign-off.
- **Status**: Arm 2 is blocked pending an approved external borrow rate data source.

## Caveats

- **3% flat rate assumption**: The 3.0% rate is an assumption. Easy-to-borrow names typically trade at general collateral (~0.3%), while distressed or hard-to-borrow small caps often command borrow fees of 10% to 30% or more. If negative surprises are concentrated in hard-to-borrow names, a flat 3% underestimates true borrow costs.
- **Sample size**: 83 short trades over 230 sessions is an average of 0.36 shorts per session. A much larger sample is required to establish statistical significance.
- **Survivorship bias**: The snapshot bars cover companies that traded through August 2026. Delisted small caps are not included, which introduces survivorship bias into historical simulations.
- **One-year window**: All decision dates fall within 2025-09 .. 2026-07.

## Raw output

Replay comparison outputs (`replay_comparison.json`, `ts_signals_870.json`) were produced against the snapshot copy and remain outside version control.
