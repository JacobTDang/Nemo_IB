# Historical Analogues — Investment Pattern Library

Curated reference of historical market setups that recur. Each entry has
structural tags for pattern matching, the actual outcome, and the lesson
the period left investors with. Used by `get_historical_analogue` to
match a current thesis to its closest prior-period rhymes.

Tag schema (canonical):
- **capex_cycle** | **capex_peak** | **capex_trough** — where in the
  spending cycle the affected industry sits
- **valuation_expansion** | **valuation_compression** — multiple regime
- **margin_expansion** | **margin_compression** — operating leverage
- **supply_constrained** | **supply_glut** — capacity vs demand
- **tech**, **energy**, **financials**, **commodities**, **consumer**,
  **real_estate** — sector
- **rate_rising** | **rate_falling** | **rate_held** — monetary regime
- **dollar_strong** | **dollar_weak** — FX regime
- **growth_acceleration** | **growth_deceleration**
- **regulatory_tightening** | **regulatory_loosening**
- **concentrated_buyers** | **diversifying_buyers** — customer base
- **insider_selling** | **insider_buying**
- **retail_frenzy** | **institutional_distribution**

Magnitude schema (per-entry, parsed by `get_historical_analogue`):
- `**Magnitude:**` line carries `-XX% peak-to-trough drawdown over YY
  months. (bear analogue)` for bust periods, or `+XX% peak gain over
  YY months. (bull analogue)` for rally periods. Use `(setup)` when
  the entry describes a precondition without a completed price move
  yet. The numeric magnitude is what `/equity-deep-research` Step 15
  uses to calibrate bear case price targets — invoking an analogue
  forces the bear PT to imply at least 60% of the drawdown magnitude.

---

## 1. 1999-2000: Dot-Com Bubble

**Period:** Q4 1998 to Q1 2000 (peak), through Q3 2002 (trough)
**Tags:** tech, capex_peak, valuation_expansion, supply_constrained,
  rate_rising, retail_frenzy, growth_acceleration
**Magnitude:** -86% peak-to-trough drawdown over 31 months (Cisco
  CSCO from $80 March 2000 to $8 Oct 2002; NDX -78% over the same
  window). (bear analogue)

**Setup:** Internet adoption inflection. Tech capex surged as telecom
build-out (fiber, switches, routers) and corporate IT prepared for "the
new economy". Cisco / Lucent / Nortel hit peak revenue and peak multiples
simultaneously. Multiples expanded from ~30x to >100x EV/EBITDA. Retail
participation peaked late (1999-2000). Fed raised rates through 2000.

**What worked:** Picks-and-shovels suppliers (Cisco, Sun Microsystems)
peaked WITH the cycle and crashed with it. The contrarian winner was
SELLING into the peak: hedge funds that shorted Nortel in late 1999
made multiples of their capital.

**What broke it:** Capex digestion. By 2001 enterprises had bought
years of IT inventory. Telecom fiber turned out to be 99% dark for
years. EV/EBITDA compressed from 100x to 8x. Stocks fell 80-95%.

**Lesson:** When CapEx ratios reach unprecedented levels AND multiples
are also at peaks, you are at a CapEx peak, not a CapEx cycle. The
return on the marginal dollar of capex is what matters, not the
absolute capex level.

**Modern rhyme triggers:** Hyperscaler AI capex rising +40% YoY +
EV/EBITDA at top quintile + customer base concentrated in 5 firms.

---

## 2. 2007-2008: Housing / Financial Crisis

**Period:** Peak Q2 2007, trough Q1 2009
**Tags:** financials, real_estate, valuation_compression, regulatory_loosening,
  rate_rising, supply_constrained, insider_selling
**Magnitude:** -83% peak-to-trough drawdown over 17 months (XLF
  financials ETF from ~$38 May 2007 to ~$6 March 2009; S&P 500
  -57% over the same window). (bear analogue)

**Setup:** Subprime mortgage origination peaked 2005-06. Housing prices
peaked Q2 2006 then plateaued. Wall Street levered into structured
products (CDOs, CMOs) through 2007. Lehman, Bear Stearns, Citi all
levered 30:1+. Insider selling at major investment banks accelerated
mid-2007. Fed raised rates through 2006.

**What worked:** Short subprime via CDS (Paulson, Burry). Owning gold
through the crisis. Long Berkshire Hathaway preferred deals (BAC, GS).

**Lesson:** Leverage + concentrated counterparty risk + opaque
derivatives + insider selling = system-wide failure mode. Watch insider
sell ratios at banks; watch CDS spreads on counterparties.

**Modern rhyme triggers:** Bank insider selling clustering + commercial
real estate writedowns rising + private credit defaults rising.

---

## 3. 2014-2016: Oil & Shale Collapse

**Period:** Peak June 2014, trough Feb 2016 (WTI from $107 to $26)
**Tags:** energy, commodities, supply_glut, capex_peak, valuation_compression,
  dollar_strong
**Magnitude:** -71% peak-to-trough drawdown over 20 months (XOP
  E&P ETF; WTI crude itself fell 76% over the same window; many
  pure-play E&Ps fell 80-95%). (bear analogue)

**Setup:** US shale output doubled 2011-2014. OPEC refused to cut.
Saudi Arabia chose to defend market share, not price. WTI fell 75%.
Energy capex collapsed from $700B to $400B globally. Many E&P names
(CHK, WLL, EOG initially) fell 60-90%.

**What worked:** Owning the IDM / integrated majors (XOM, CVX) over the
pure-plays. Buying the survivors at the bottom (CXO, PXD). Avoiding
service companies (HAL, SLB) which had no pricing power. Shorting the
weakest E&Ps with debt covenants.

**Lesson:** Commodity supply is sticky on the downside — shale producers
keep pumping at any price above marginal cost to service debt. Watch
debt covenants, not P&L losses. The leverage + commodity-price combo is
deadly; pure-play with weak balance sheet is uninvestable in a glut.

**Modern rhyme triggers:** Capex rising while end-product price falling
+ debt-funded growth + low-cost producer entering market.

---

## 4. 2010-2012: Smartphone / Mobile Cycle

**Period:** iPhone 4 (2010) through iPhone 5S (2013)
**Tags:** tech, consumer, capex_cycle, supply_constrained, growth_acceleration,
  margin_expansion
**Magnitude:** +233% peak gain over 36 months (AAPL from ~$30 mid-2010
  to ~$100 split-adjusted late 2012; TSM and Samsung memory both +100%
  over the window). (bull analogue)

**Setup:** Smartphone adoption inflected to >50% of mobile phones.
Apple ramped chip orders to TSMC, Samsung memory orders, Foxconn
assembly. Suppliers ramped capacity. Carriers subsidized handsets and
absorbed the cost into multi-year contracts.

**What worked:** Long supply chain (TSM, Samsung memory, Synaptics).
Long Apple. Long carriers later (AT&T, VZ) as data monetized. Long
infrastructure (CSCO at the right entry).

**Lesson:** When a new platform's adoption curve inflects from 10% to
40% household penetration, suppliers with capacity get pricing power
for 3-5 years. Watch lead times and capacity utilization, not just
demand. The winning trade is to identify which supplier has the largest
share gain potential, not to own the brand.

**Modern rhyme triggers:** AI compute adoption at <30% in enterprise +
HBM/CoWoS lead times stretching out + customer prepayments rising.

---

## 5. 2020-2021: COVID Cloud Acceleration

**Period:** March 2020 trough through Q4 2021 peak; bust phase
  through Oct 2022
**Tags:** tech, growth_acceleration, valuation_expansion, margin_expansion,
  rate_falling, retail_frenzy
**Magnitude:** -70% peak-to-trough drawdown over 24 months on the
  bust phase (WCLD cloud ETF from ~$69 Feb 2021 to ~$24 Oct 2022;
  marquee names SHOP / DDOG / NET fell 75-85%). (bear analogue —
  use for the post-pull-forward exhaustion case, not the 2020-21
  rally itself)

**Setup:** Remote work pulled forward 3-5 years of cloud adoption into
24 months. Snowflake IPO'd at peak frenzy. SaaS multiples hit 30x+
revenue. Tiger Global, Coatue funded everything. Fed at zero rates.
Inflation initially called "transitory."

**What worked:** Long every SaaS / cloud / e-commerce name through 2020,
exit through 2021. Best returns: SHOP, DDOG, NET, MELI. Worst tail risk:
the pull-forward exhaustion in 2022 when growth normalized.

**Lesson:** Pull-forward demand looks like permanent demand for one
cycle. The exit point is when YoY comparisons flip to "normalized" and
guidance starts including "tough comps" language. CFO tone shifted
visibly Q3 2021 in many names.

**Modern rhyme triggers:** "Tough comps" appearing in earnings releases
+ growth decelerating from 50%+ to 20-30% + multiple still elevated.

---

## 6. 2008-2011: China Infrastructure Stimulus

**Period:** Late 2008 stimulus through 2011 (commodity peak), 2014
demand collapse
**Tags:** commodities, real_estate, growth_acceleration, supply_constrained,
  dollar_weak
**Magnitude:** +200% peak gain over 30 months (copper $1.50 → $4.50;
  BHP / RIO / FCX all roughly tripled off late-2008 lows by mid-2011).
  (bull analogue)

**Setup:** China's 4T RMB stimulus drove commodity demand. Copper went
from $1.50 to $4.50. Iron ore peaked. BHP, RIO, Vale, Freeport ramped
capacity. Many commodity names doubled or tripled.

**What worked:** Long mining majors 2009-2011. Long Australia / Brazil
exposed names. Long ag inputs (Potash, Mosaic).

**Lesson:** Demand-side stimulus produces a 24-36 month commodity rally
when supply is capacity-constrained. Watch Chinese property starts and
PMI manufacturing as leading indicators. The supply response (capex
ramp) takes 3-5 years; by then demand has normalized and you're back
in glut.

**Modern rhyme triggers:** Government infrastructure spending +
capacity-constrained commodities + low capex history.

---

## 7. 2015-2020: Energy Capex Drought

**Period:** 2015-2020 (after the 2014 collapse)
**Tags:** energy, capex_trough, valuation_compression, supply_constrained
  (future)
**Magnitude:** N/A — this is a precondition / setup analogue, not a
  completed price move. The drawdown belongs to entry #3
  (2014-2016 collapse); the subsequent uranium / copper rallies
  (2018-2024) belong to a follow-on analogue not yet catalogued.
  (setup)

**Setup:** After the 2014-16 crash, oil majors slashed capex. Long-cycle
projects (deepwater, oil sands, Russia frontier) were canceled. Reserve
replacement ratios fell below 100% for years. Oil services bankruptcies
peaked 2016-17. Capex stayed depressed even as oil recovered to $60-70.

**What set up the next move:** By 2020, the supply curve had moved up
+5-10 USD/bbl on the marginal barrel because cheap reserves had been
depleted. Then COVID demand recovery hit a supply-tight world.

**Lesson:** Long-cycle commodity capex troughs set up multi-year
spikes 5-7 years later. Watch reserve replacement ratios, drilling
permits, and the marginal cost curve.

**Modern rhyme triggers:** Uranium (2018-2022 capex drought + AI
data-center power demand inflection), copper (2014-2022 drought +
EV transition).

---

## 8. 2018-2019: Memory Bust

**Period:** 2018 peak through Q2 2019 trough (DRAM ASP fell ~50%)
**Tags:** tech, commodities, supply_glut, valuation_compression,
  capex_peak
**Magnitude:** -52% peak-to-trough drawdown over 7 months (MU from
  ~$63 May 2018 to ~$30 Dec 2018; SK Hynix similar; DRAM ASP itself
  -50%). (bear analogue)

**Setup:** Micron and SK Hynix had ramped DRAM capacity 2016-2018
chasing peak prices. Hyperscaler buyers (AWS, Azure, GCP) had pulled
forward inventory. When demand normalized, DRAM ASP collapsed. Micron
EPS went from $11 to $2 in 12 months.

**What worked:** Short Micron / SK Hynix at the peak (2018). Long them
6-9 months later at the trough (Q1 2020).

**Lesson:** Memory is a pure commodity. When ASPs are at peak and capex
ratios are at peak, you are 6-12 months from a glut. Watch wafer-start
data, customer inventory days, and second-tier capacity announcements.

**Modern rhyme triggers:** HBM ASPs rising + Samsung / SK Hynix / Micron
adding HBM capacity + AI capex pace decelerating.

---

## 9. 2007: Subprime Auto / Consumer Credit

**Period:** Q1 2007 through 2009 (subprime auto delinquencies peaked
2009)
**Tags:** financials, consumer, regulatory_loosening, insider_selling
**Magnitude:** -70% peak-to-trough drawdown over 18 months (COF
  Capital One from ~$80 mid-2007 to ~$8 early-2009; consumer-credit
  cohort 60-85% range). (bear analogue)

**Setup:** Subprime auto lending hit record originations 2005-07.
Securitization of auto loans peaked. Capital One, Santander Consumer,
AmeriCredit all had peak exposures.

**Modern rhyme triggers:** BNPL (Affirm, Klarna) defaults rising +
auto loan delinquencies rising + credit card 60+day delinquencies at
multi-year highs.

---

## 10. 2021-2022: SPAC / Reflexivity Bust

**Period:** Q4 2020 peak, through 2022 collapse
**Tags:** tech, valuation_expansion, retail_frenzy, regulatory_loosening
**Magnitude:** -85% peak-to-trough drawdown over 20 months (de-SPAC
  index from Feb 2021 peak through Oct 2022; marquee names LCID -85%,
  NKLA -97%, QS -90%). (bear analogue)

**Setup:** SPAC issuance hit $160B in 2020-21. Pre-revenue companies
(Lucid, Nikola, QuantumScape) went public at $10B+ valuations.
Reflexivity drove fundamental decision-making: companies guided to
hockey-stick revenue ramps because SPAC merger required it.

**Lesson:** When companies guide on outcomes that require their share
price to stay elevated, the cycle is self-fulfilling on the way up
and self-destructing on the way down. Watch for "guidance based on
order-book conversion" combined with "order book is non-binding LOIs".

**Modern rhyme triggers:** Any AI / EV / quantum SPAC with capex
spending committed against non-binding customer commitments.

---

## 11. 2023-2024: AI Hyperscaler Capex Cycle

**Period:** Mid-2023 (NVDA Q1 FY24) through ongoing
**Tags:** tech, capex_peak, growth_acceleration, valuation_expansion,
  supply_constrained, concentrated_buyers
**Magnitude:** N/A — cycle still active as of writing. No completed
  drawdown to calibrate against. For analyst purposes, use the
  composite of analogues 1 (Cisco, -86%), 4 (smartphone, +233%
  upside) and 8 (memory, -52%) to bound bear / bull cases. (setup)

**Setup:** ChatGPT triggered hyperscaler AI capex acceleration. Big 4
CSPs (MSFT, GOOGL, META, AMZN) capex went from $130B to $300B in 24
months. NVDA Datacenter revenue grew from $4B/qtr to $35B/qtr. HBM3e
became supply-constrained at SK Hynix. TSMC CoWoS packaging became the
bottleneck.

**Status (as of writing):** Cycle still active. Watch:
- HBM / CoWoS capacity additions (when supply catches up, pricing
  pressure begins)
- Hyperscaler capex guidance — first quarter of "tough comps" language
  is the inflection
- Customer concentration — NVDA's top 4 customers = ~40% of revenue
  is a 1999-Cisco-level concentration risk

**Most relevant analogues:** 1999 dot-com (capex peak + concentrated
buyers), 2018 memory bust (supply addition response), 2010-12
smartphone cycle (winning suppliers in adoption inflection).

---

## How to use this catalog

Match a current thesis to historical periods by structural tags, not by
surface narrative. The 2024 AI capex cycle is NOT just "like the
internet" — it's like the 1999 dot-com AND the 2010 smartphone cycle
AND the 2018 memory bust depending on which structural feature you
emphasize. The art is identifying which mechanism is most likely to
drive the next 12 months.

Highest-value tags for matching:
1. Where in the capex cycle (peak / trough / mid)
2. Buyer concentration (concentrated / diversified)
3. Multiple regime (expanding / compressing)
4. Supply elasticity (constrained / glut)
5. Insider behavior (buying / selling)
