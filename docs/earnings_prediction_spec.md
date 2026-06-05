# Earnings Prediction Data Pipeline — Spec Sheet

Goal: predict revenue, EPS, and guidance direction for watchlist companies
before the earnings call, using a combination of alternative data, supply
chain signals, and macro indicators. The output is a structured pre-earnings
brief with a surprise probability vs consensus and a confidence-weighted
estimate.

---

## Signal Taxonomy

Signals are grouped into six categories. Each entry specifies what it
predicts, the data source, the access method, update cadence, how far ahead
of earnings it is useful, and which company types benefit most.

**Prediction targets:**
- `REV` — revenue / top-line surprise
- `MARGIN` — gross or operating margin surprise
- `EPS` — bottom-line beat/miss
- `GUIDE` — guidance raise/cut direction
- `VOLUME` — unit volume / demand
- `COST` — input cost / supply chain cost

**Signal quality tiers:**
- `HIGH` — directly measures something close to the financial metric
- `MEDIUM` — proxy signal with meaningful correlation
- `LOW` — noisy / directional only

---

## Category 1 — Product Demand / Consumer Signals

These proxy actual revenue before it's reported.

### 1.1 Google Trends
- **Predicts:** REV, VOLUME
- **Quality:** MEDIUM
- **Source:** Google Trends (public, no auth)
- **Access:** `pytrends` Python library — `TrendReq.build_payload([keywords], timeframe='today 3-m')`
- **Query approach:**
  - Brand name: `"iPhone"`, `"Tesla"`, `"ChatGPT"`
  - Product-level: `"iPhone 16"`, `"RTX 5090"`, `"Cybertruck"`
  - Intent-level: `"buy iPhone"`, `"iPhone deal"`, `"Samsung Galaxy S25"` (competitor)
  - Geographic breakdown: pull by country for international revenue exposure
- **Cadence:** daily (API is near-real-time with ~3-day lag)
- **Lead time:** signals visible 4–8 weeks before earnings
- **Best for:** consumer electronics, automotive, consumer internet, retail
- **Limitations:** relative index (0–100), not absolute volume; seasonality must be
  normalized against prior-year same period; only captures search intent, not conversion
- **Implementation note:** compare current quarter trend index vs same quarter
  last year. If index is up 20%+ YoY for the same search terms that correlated
  with revenue the prior year, flag as bullish revenue signal.

### 1.2 App Store Rankings and Download Velocity
- **Predicts:** REV, VOLUME
- **Quality:** MEDIUM-HIGH for mobile-first companies
- **Source:** Apple App Store (iTunes RSS feed), Google Play Store (scraping)
- **Access:**
  - Apple: `https://rss.applemarketingtools.com/api/v2/us/apps/top-free/50/apps.json`
    (top 50 free/paid/grossing by category, no auth)
  - Google Play: scrape `https://play.google.com/store/apps/top` with BeautifulSoup
  - Third-party aggregator: `https://serpapi.com/google-play-store-api` (paid, but
    limited free tier available)
- **Cadence:** daily scrape
- **Lead time:** 2–6 weeks
- **Best for:** Spotify, Netflix, Duolingo, Snap, Meta apps, gaming companies
- **Key metric:** rank trend over 30/60/90 days within category; grossing chart
  position (revenue proxy); review count velocity (new reviews/day)

### 1.3 Amazon Product Performance
- **Predicts:** REV, VOLUME, MARGIN (pricing power)
- **Quality:** MEDIUM-HIGH for consumer goods companies that sell on Amazon
- **Source:** Amazon product pages + BSR (Best Seller Rank) data
- **Access:**
  - BSR is embedded in the product page HTML — scrape with requests + BeautifulSoup
  - `keepa.com` API (free tier: 250 calls/day) returns historical price and BSR
    time series per ASIN
  - `rainforest-api.com` (paid) returns structured product data
- **Cadence:** daily for top ASINs; weekly for broader coverage
- **Key metrics:**
  - BSR trend (lower = selling better)
  - Review count velocity (proxy for unit sales)
  - Price stability / discount frequency (margin signal — discounting = demand weakness)
  - `Questions answered` count change (engagement proxy)
- **Best for:** consumer electronics (AAPL accessories, AMZN Echo/Kindle, SONO),
  CPG (P&G, KO, PEP), retail brands (NKE, LULU, CROX)
- **Limitations:** channel mix — a company selling 10% on Amazon and 90% direct
  has less signal here

### 1.4 Web Traffic (SimilarWeb / Semrush proxy)
- **Predicts:** REV (for web-native businesses)
- **Quality:** MEDIUM
- **Source:**
  - SimilarWeb: free tier gives monthly visits, bounce rate, engagement for any domain
    at `https://www.similarweb.com/website/<domain>/` (scraping required for bulk)
  - Semrush: free tier gives organic traffic estimates
  - `similarweb-python` unofficial library exists
- **Cadence:** monthly data (1-month lag)
- **Best for:** e-commerce (SHOP, ETSY, AMZN), SaaS (SNOW, CRM, NOW), media (NFLX)
- **Key metric:** YoY monthly unique visitor growth in quarter; time-on-site trend;
  conversion-adjacent metrics (cart page traffic, checkout page traffic if available)

### 1.5 Foot Traffic Data
- **Predicts:** REV, VOLUME (same-store sales proxy)
- **Quality:** HIGH for retail / restaurant / gym
- **Source:**
  - Placer.ai (paid, ~$500/mo for research tier)
  - SafeGraph (now Dewey Data — academic access available)
  - Google Popular Times (embedded in Maps — scrapeable via `populartimes` Python library)
- **Access:** `pip install populartimes` → `populartimes.get_id(api_key, place_id)`
  Returns hourly visit distribution per location
- **Cadence:** weekly rollup
- **Best for:** MCD, SBUX, TGT, WMT, DG, ROST, LULU, Planet Fitness, AMC
- **Note:** Google Popular Times data is directional (relative busyness), not absolute
  count. Trend change YoY is the signal, not absolute level.

---

## Category 2 — Supply Chain and Manufacturing

These measure what companies are building/buying before they ship.

### 2.1 US Import/Export Records (Customs Data)
- **Predicts:** REV, VOLUME, COST
- **Quality:** HIGH — directly measures physical goods flow
- **Source:** US Census Bureau via Customs and Border Protection manifest data
- **Access:**
  - Census.gov trade API: `https://api.census.gov/data/timeseries/intltrade/imports`
    (free, requires free API key at `api.census.gov/data/key_signup.html`)
  - Endpoint example: `GET /data/timeseries/intltrade/imports?get=GEN_VAL_MO,CTY_CODE&COMM_LVL=HS6&COMM=851712&time=2026-01`
    (HS code 851712 = smartphones)
  - Bill of Lading data (shipper/consignee level): `ImportGenius` (paid ~$200/mo),
    `Panjiva` (S&P Global, institutional), `ImportYeti` (free, limited)
  - `ImportYeti.com` has a free search UI; no official API but JSON endpoints
    can be reversed from the browser
- **Cadence:** monthly (Census publishes ~5 weeks after month end)
- **Key signals:**
  - AAPL: look for Foxconn/Pegatron/Hon Hai shipments from Taiwan/China to US
  - NVDA: look for shipments of H100/H200/B200 class to hyperscaler data centers
  - TSLA: look for vehicle component imports from Giga Shanghai
- **HS codes to track by sector:**
  - Smartphones: 8517.12
  - Computers/laptops: 8471.30
  - GPUs/processors: 8542.31
  - EV batteries: 8507.60
  - Displays: 9013.80
  - Semiconductors (general): 8542

### 2.2 Supplier Earnings as Leading Indicators
- **Predicts:** REV, VOLUME, COST, MARGIN
- **Quality:** HIGH — directly measures demand from the upstream side
- **Source:** We already have Finnhub `get_earnings_calendar` + `get_financial_statements`
- **Access:** already built — this is a workflow pattern, not a new data source
- **Key supplier-to-downstream mappings:**

  | Supplier | Reports | Downstream read-through |
  |---|---|---|
  | TSMC (TSM) | Monthly revenue + quarterly | AAPL, NVDA, AMD, QCOM, AVGO |
  | Samsung Electronics | Quarterly preliminary | AAPL (displays/memory), SK Hynix comps |
  | SK Hynix | Quarterly | NVDA (HBM), memory pricing |
  | ASML | Quarterly | Intel, TSMC, Samsung — semiconductor capex |
  | Foxconn (2317.TW) | Monthly revenue | AAPL hardware volume |
  | Murata Mfg (6981.T) | Quarterly | AAPL components, EV battery |
  | Corning (GLW) | Quarterly | AAPL display glass, fiber |
  | Lam Research (LRCX) | Quarterly | TSMC/Samsung capex |
  | Applied Materials (AMAT) | Quarterly | Semi capex broadly |
  | Flex Ltd (FLEX) | Quarterly | Server OEM (HPE, Dell, cloud) |
  | Celestica (CLS) | Quarterly | Hyperscaler server demand |
  | Jabil (JBL) | Quarterly | AAPL, healthcare devices |
  | ON Semiconductor | Quarterly | Automotive EV (TSLA, GM) |
  | Allegro MicroSystems | Quarterly | Automotive, EV |
  | WD / Seagate | Quarterly | Hyperscaler storage capex |

- **Cadence:** quarterly (with TSMC/Foxconn monthly releases as bonus signals)
- **Workflow:** when any supplier reports, auto-run `/cross-company-readthrough`
  for all downstream names on the watchlist

### 2.3 Shipping and Freight Rates
- **Predicts:** REV timing (is product moving?), COST (freight as % of COGS)
- **Quality:** MEDIUM
- **Source:**
  - Freightos Baltic Index (FBX): `https://fbx.freightos.com/` — free data for spot
    container rates on 12 major trade lanes
  - Baltic Dry Index (BDI): `https://www.balticexchange.com/en/data/indices.html`
    (dry bulk — industrial commodities)
  - FRED series `BAMLH0A0HYM2` won't help here, but FRED has `DCPN3M` (commercial
    paper) which proxies supply chain financing stress
  - Xeneta: paid API for contract rates
- **Access:** Freightos publishes a public JSON endpoint (reverse-engineer from
  fbx.freightos.com network tab); alternatively scrape the page weekly
- **Key lanes to watch:**
  - Trans-Pacific Eastbound (China→US): AAPL, TSLA, consumer goods importers
  - Asia-Europe: luxury brands (LVMH, Kering), European industrials
  - Trans-Atlantic: automotive
- **Cadence:** weekly spot rates; monthly contract rates
- **Interpretation:** spike in rates = strong demand (goods are moving); rate
  collapse = inventory glut or demand weakness

### 2.4 Port Throughput
- **Predicts:** REV (volume of goods entering US market)
- **Quality:** MEDIUM
- **Source:**
  - Port of LA: `https://www.portofla.org/port-info/statistics/` (monthly PDF/Excel)
  - Port of Long Beach: `https://www.polb.com/port-info/news-and-press/statistics/`
  - Port of NY/NJ, Savannah, Houston: similar public stats pages
- **Access:** direct download of published Excel/PDF monthly stats; parse with
  `pandas.read_excel` or `pdfplumber`
- **Cadence:** monthly (published ~3 weeks after month end)
- **Key metric:** TEUs (twenty-foot equivalent units) processed YoY change
- **Best for:** any company with significant import volume through US ports

### 2.5 Manufacturing PMI and Industrial Data
- **Predicts:** MARGIN, COST, GUIDE (for industrial companies)
- **Quality:** MEDIUM-HIGH for industrials, LOW for pure tech
- **Source:** FRED (already integrated)
- **FRED series to add:**
  - `MANEMP` — manufacturing employment
  - `IPMAN` — industrial production manufacturing
  - `PCUOMFGOMFG` — PPI for all manufacturing (input cost proxy)
  - `ISM/MAN_PMI` — not on FRED but ISM publishes at `ismworld.org` (free with
    registration); scrape the monthly release
  - `PCU334413334413` — PPI for semiconductors (cost signal for chip companies)
  - `PCU336111336111` — PPI for automobile manufacturing
- **Cadence:** monthly

---

## Category 3 — Workforce and Hiring Signals

Hiring patterns telegraph where management is investing and expecting growth.

### 3.1 Job Posting Volume by Division
- **Predicts:** GUIDE, REV (medium-term), investment direction
- **Quality:** MEDIUM
- **Source:**
  - LinkedIn (scraping — against ToS but widely practiced; use with caution)
  - Indeed: `https://www.indeed.com/jobs?q=<company+name>&l=` — scrapeable
  - Glassdoor: similar structure
  - **Best approach:** `jobs.lever.co/<company>` and `boards.greenhouse.io/<company>`
    — most tech companies post on these ATS platforms with public JSON APIs:
    `https://api.lever.co/v0/postings/<company>?mode=json`
    `https://boards-api.greenhouse.io/v1/boards/<company>/jobs`
    Both return structured JSON with department, location, title — no auth required.
- **Cadence:** weekly snapshot; track 4-week rolling count by department
- **Key signals:**
  - Sales/GTM headcount growth → revenue confidence
  - Engineering headcount surge in specific product area → product launch incoming
  - Finance/accounting hiring freeze → cost discipline / margin focus
  - Mass layoff announcement → guidance cut risk (cross-reference layoffs.fyi)
- **`layoffs.fyi` data:** `https://layoffs.fyi/` — scrapeable; tracks tech layoffs
  with company, date, headcount. FRED and LinkedIn won't catch this; layoffs.fyi
  does within days of announcement.

### 3.2 LinkedIn Headcount Trend
- **Predicts:** GUIDE, cost structure
- **Quality:** MEDIUM
- **Source:** LinkedIn company pages publish employee count (updates ~monthly)
- **Access:** LinkedIn has an official API (requires partnership approval), but the
  company headcount is embedded in the public company page HTML. Scrape with
  Playwright/Selenium (requires login session). Alternatively, `proxycurl.com`
  API (paid, $0.01/credit) returns structured LinkedIn data without scraping.
- **Cadence:** monthly
- **Implementation note:** calculate QoQ % change in headcount by department
  (engineering vs sales vs operations) as a leading indicator for the earnings
  trajectory.

---

## Category 4 — Competitor and Market Share

### 4.1 Competitor Earnings Readthrough (automated)
- **Predicts:** REV, MARGIN, GUIDE
- **Quality:** HIGH
- **Source:** Finnhub earnings calendar (already have)
- **Workflow:** for each watchlist company, pre-compute a list of direct
  competitors (from `get_company_peers`) and key suppliers (from `get_supply_chain`).
  When any of those companies reports, auto-trigger `/cross-company-readthrough`.
  Store the readthrough output in a new `preearnings_signals` table with the
  source company, the impacted watchlist ticker, and the directional flag.

### 4.2 Pricing Intelligence
- **Predicts:** MARGIN, VOLUME (elasticity signal)
- **Quality:** MEDIUM-HIGH
- **Source:** company and competitor product pages
- **Access:** scrape the company's own product pricing page weekly.
  Compare vs competitor pricing for same SKU class. Track:
  - Has the company raised prices without a new product (pricing power → margin expand)?
  - Has a competitor cut prices (competitive pressure → margin risk)?
  - Is the product being discounted more than seasonally expected (demand weakness)?
- **Implementation:** maintain a `pricing_snapshots` table per product/URL.
  Weekly diff the prices. Alert on: own price increase >5%, competitor cut >10%,
  own discount >15% outside normal sale periods.
- **Key pages to track:**
  - AAPL: `apple.com/shop/buy-iphone`
  - NVDA: AIB partner pages (ASUS, MSI, Gigabyte RTX listings on Newegg/B&H)
  - TSLA: `tesla.com/model3` (price has been a key signal — cuts signal demand weakness)
  - MSFT: `microsoft.com/en-us/microsoft-365/business/compare-all-plans`

### 4.3 Market Share Proxies (Free)
- **Predicts:** REV trend (market position)
- **Quality:** LOW-MEDIUM
- **Source:**
  - Steam Hardware Survey: `store.steampowered.com/hwsurvey` — GPU market share
    (NVDA vs AMD). Published monthly, scrape the JSON embedded in the page.
  - StatCounter GlobalStats: `gs.statcounter.com` — browser (GOOG Chrome share),
    OS (MSFT Windows vs AAPL macOS), mobile vendor
  - IDC/Gartner: quarterly reports published as press releases (scrape headlines)
  - Counterpoint Research: similar — press release scraping for smartphone shipments
  - Canalys: channel shipment data for PC, smartphone, cloud
- **Cadence:** monthly (Steam); quarterly (IDC/Gartner/Canalys)

---

## Category 5 — International Signals

Material events that appear in foreign regulatory filings before US press picks
them up.

### 5.1 EU Regulatory Filings (DG Competition / EASA / EMA)
- **Predicts:** GUIDE, M&A, new product approvals
- **Quality:** HIGH when relevant (not always)
- **Source:**
  - EU merger database: `https://ec.europa.eu/competition/mergers/cases/`
    — search by company name, returns JSON for cases mentioning that company
  - EUR-Lex official journal: `https://eur-lex.europa.eu/search.html` — full text
    search of EU regulatory publications
  - EMA (drug approvals): `https://www.ema.europa.eu/en/medicines/` — REST API
    `https://www.ebi.ac.uk/chembl/api/data/molecule/` for drug pipeline
- **Access:** all public, no auth; JSON/XML APIs available
- **Cadence:** monitor weekly for any mention of watchlist company names

### 5.2 China SAMR Filings (M&A, Anti-Monopoly)
- **Predicts:** deal completion risk, international expansion
- **Quality:** HIGH when relevant
- **Source:** `http://www.samr.gov.cn/fldj/tzgg/` — China State Administration
  for Market Regulation, merger review announcements
- **Access:** requires scraping Chinese-language pages; use `requests` +
  `BeautifulSoup` + `googletrans` or `deepl` API for translation
- **Cadence:** weekly check
- **Relevance:** any acquisition by a watchlist company that requires Chinese
  regulatory approval will appear here. China often approves last (longest review),
  so SAMR filing = deal is 6-12 months from close

### 5.3 Japan JFTC and FSA
- **Predicts:** deal activity, financial regulatory changes
- **Source:** `https://www.jftc.go.jp/en/pressreleases/` (English section),
  `https://www.fsa.go.jp/en/news/`
- **Access:** scrape press release lists weekly

### 5.4 Taiwan Stock Exchange Monthly Revenue
- **Predicts:** REV, VOLUME (for companies with Taiwan-listed suppliers)
- **Quality:** HIGH
- **Source:** `https://mops.twse.com.tw/mops/web/t05st10_ifrs` — Taiwan MOPS
  (Market Observation Post System). Every listed Taiwan company must file monthly
  revenue by the 10th of the following month.
- **Access:** POST request to MOPS with company code returns HTML table; parse
  with BeautifulSoup. TSMC = 2330, Foxconn = 2317, MediaTek = 2454,
  ASE Group = 3711
- **Cadence:** monthly, published by 10th of following month
- **This is the single highest-value free data source for semiconductor earnings.**
  TSMC's monthly revenue for March is published April 10 — NVDA doesn't report
  until May. That's 3-4 weeks of advance signal.

### 5.5 Korean Customs and Samsung Preliminary Results
- **Predicts:** memory chip pricing, AAPL display demand
- **Source:**
  - Samsung publishes preliminary earnings (revenue + operating profit estimate)
    ~2 weeks before full results. Highly reliable for DRAM/NAND pricing direction.
  - Korean Customs Service publishes monthly export/import data by product category.
    `https://unipass.customs.go.kr/ets/` (English portal available)
- **Cadence:** monthly (customs); quarterly preliminary (Samsung)

---

## Category 6 — Sentiment and Qualitative

### 6.1 Earnings Call Tone Analysis (already have)
- `extract_call_sentiment` — already built, tracks CFO/CEO tone shift
- Extend: track against prior 4 quarters to build a sentiment trend line

### 6.2 Patent Filing Velocity
- **Predicts:** GUIDE (future product investment), competitive moat
- **Quality:** LOW (long lead time, 18-month publication lag)
- **Source:**
  - USPTO PatentsView API: `https://patentsview.org/apis/api-endpoints/patents`
    (free, no auth) — returns structured patent data by assignee
  - WIPO PATENTSCOPE: `https://patentscope.wipo.int/search/en/search.jsf`
    — international filings, REST API available
  - Google Patents: scrapeable
- **Access:** `GET https://api.patentsview.org/patents/query?q={"assignee_organization":"NVIDIA"}&f=["patent_date","patent_title"]`
- **Key metric:** filing velocity in a new technical area (e.g., NVDA filing
  surge in networking patents before InfiniBand integration, AAPL VR patents
  before Vision Pro). Lag is 12-18 months, so this is guidance/product cycle
  intelligence, not current-quarter signal.

### 6.3 Social Sentiment (Reddit / StockTwits)
- **Predicts:** consumer sentiment, retail investor positioning
- **Quality:** LOW (noisy) — useful for consumer brands, not B2B
- **Source:**
  - Reddit: `https://www.reddit.com/r/wallstreetbets/.json` — public JSON feed;
    also Pushshift API (now limited) for historical
  - StockTwits: `https://api.stocktwits.com/api/2/streams/symbol/AAPL.json` (free)
  - Twitter/X: v2 API (basic free tier: 500k reads/month)
- **Cadence:** daily or real-time
- **Note:** sentiment alone is weak signal. Most useful as a contrarian indicator
  (extreme bullish retail sentiment near earnings = positioning risk) or as a
  spike detector for breaking news spreading on social before news wires.

### 6.4 News Sentiment Scoring
- **Predicts:** short-term price reaction, analyst revision trigger
- **Source:** we already have `get_company_news` (Finnhub) and `obb_news_company`
- **Extend:** pipe headlines into a sentiment scorer. Options:
  - `transformers` library with `ProsusAI/finbert` model (financial BERT,
    free, runs locally) — classifies news as positive/neutral/negative
  - `VADER` (NLTK, free) — rule-based, less accurate but faster
  - FinBERT via Hugging Face: `pipeline("text-classification", model="ProsusAI/finbert")`
- **Output:** sentiment score per article, rolling 7-day sentiment index per ticker,
  flagging when sentiment drops sharply ahead of earnings

---

## Category 7 — Options Market Signals

Options market often prices in information before it's public.

### 7.1 Implied Volatility Term Structure
- **Predicts:** market-implied probability of earnings surprise
- **Quality:** HIGH (the market's own estimate)
- **Source:** `obb_options_chain` (already working)
- **Key metrics:**
  - IV crush size (implied move = (front-month ATM straddle price) / spot price)
  - Skew: put IV vs call IV at equivalent delta — put skew widening ahead of
    earnings = institutional hedging = bears are paying up
  - Term structure: if near-term IV > 60-day IV, earnings is the dominant risk event
- **Already accessible:** `mcp__nemo_openbb__obb_options_chain` + `mcp__nemo_financial__get_options_metrics`

### 7.2 Unusual Options Activity
- **Predicts:** directional bet by informed participants
- **Quality:** MEDIUM-HIGH (not always informed; can be hedges)
- **Source:**
  - Full options chain via `obb_options_chain` — already have
  - Filter for: single-name calls/puts with OI spike >5x 20-day average OI;
    premium >$1M; expiry within 30 days of earnings
  - `unusualwhales.com` publishes aggregated unusual activity (scrape the
    public feed or use their API — limited free tier)
  - `Barchart.com/options/unusual-activity` — scrapeable
- **Implementation:** compute 20-day rolling average OI per strike/expiry;
  flag rows where today's OI is >3x the average with significant premium

---

## Data Pipeline Architecture

### Tables to add to state/schema.py

```sql
-- Pre-earnings signals: all collected signals for a ticker ahead of earnings
CREATE TABLE IF NOT EXISTS preearnings_signals (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  ticker            TEXT NOT NULL,
  earnings_date     TEXT NOT NULL,          -- YYYY-MM-DD
  signal_category   TEXT NOT NULL,          -- demand / supply_chain / hiring /
                                            --   competitor / international /
                                            --   sentiment / options
  signal_name       TEXT NOT NULL,          -- e.g. google_trends, tsmc_monthly_rev
  direction         TEXT NOT NULL,          -- bullish / bearish / neutral
  magnitude         REAL,                   -- 0.0-1.0 normalized strength
  raw_value         TEXT,                   -- JSON blob of the raw data point
  source_url        TEXT,
  collected_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  days_before_earnings INTEGER             -- negative = before, positive = after
);

-- Supplier readthrough log: tracks which supplier reports triggered readthrough
CREATE TABLE IF NOT EXISTS supplier_readthroughs (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  supplier_ticker   TEXT NOT NULL,
  downstream_ticker TEXT NOT NULL,
  supplier_report_date TEXT NOT NULL,
  direction         TEXT NOT NULL,          -- bullish / bearish / neutral
  key_findings      TEXT,                   -- prose summary
  confidence        REAL,
  triggered_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Pricing snapshots: weekly price scrape per product/competitor
CREATE TABLE IF NOT EXISTS pricing_snapshots (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  ticker            TEXT NOT NULL,
  product_name      TEXT NOT NULL,
  price             REAL,
  currency          TEXT DEFAULT 'USD',
  source_url        TEXT NOT NULL,
  is_discounted     INTEGER DEFAULT 0,
  discount_pct      REAL,
  snapped_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### New MCP Server: `nemo_altdata`

Wraps the free alternative data sources that are too slow or complex to
call inline. Tools:

| Tool name | Signal | Source |
|---|---|---|
| `get_google_trends` | Consumer demand | pytrends |
| `get_taiwan_monthly_revenue` | Supplier volume | MOPS API |
| `get_shipping_rates` | Freight cost/volume | Freightos scrape |
| `get_port_throughput` | Import volume | Port of LA/LB |
| `get_job_postings_count` | Hiring signal | Lever/Greenhouse JSON |
| `get_ipo_alt_data` | Not IPO — import records | Census trade API |
| `get_amazon_product_rank` | Consumer demand | Keepa + scrape |
| `get_options_unusual_activity` | Informed positioning | Barchart scrape |
| `get_patent_velocity` | Long-term investment | PatentsView API |
| `get_social_sentiment` | Retail sentiment | StockTwits API |
| `get_finbert_sentiment` | News sentiment | FinBERT local model |

### New Skill: `/preearnings-research`

Runs automatically 28, 14, and 7 days before each watchlist company's earnings
date. Steps:

1. Confirm earnings date from `get_earnings_calendar`
2. Pull supplier list from `get_supply_chain`; check which suppliers have already
   reported this quarter → auto-run readthrough for each
3. Pull Google Trends for brand + top 3 products (vs same quarter last year)
4. Pull Taiwan MOPS monthly revenue for relevant suppliers
5. Pull shipping rate index for primary trade lane
6. Pull job posting count vs 90 days ago (Lever/Greenhouse)
7. Pull options IV term structure + skew vs 30-day average
8. Pull pricing snapshot vs 30 days ago
9. Pull FinBERT sentiment score on last 30 days of news
10. Synthesize: weight signals by quality tier and category; produce a
    structured summary with:
    - Revenue surprise probability (beat / in-line / miss)
    - Margin direction
    - Guidance direction
    - Top 3 supporting signals (cited with source)
    - Top 2 risks to the prediction
    - Confidence: 0.0-1.0

### Cadence and Triggering

| Signal | Collection cadence | Trigger mechanism |
|---|---|---|
| Google Trends | Weekly | Cron: every Sunday |
| Taiwan MOPS | Monthly (10th) | Cron: 11th of each month |
| Shipping rates | Weekly | Cron: every Monday |
| Job postings | Weekly | Cron: every Sunday |
| Supplier earnings | Event-driven | Earnings calendar watcher |
| Options IV | Daily (2 weeks pre-earnings) | Days-before-earnings gate |
| Amazon BSR | Weekly | Cron: every Saturday |
| Pricing snapshot | Weekly | Cron: every Wednesday |
| News sentiment | Daily | Same as event news daemon |
| Port throughput | Monthly | Cron: 25th of each month |

---

## Priority Build Order

**Phase 1 — Highest signal, lowest cost (build first):**
1. Taiwan MOPS monthly revenue scraper (TSMC, Foxconn, MediaTek)
2. Supplier earnings auto-readthrough (extends existing cross-company-readthrough)
3. Google Trends integration (pytrends)
4. Options IV implied move calculator (extends existing obb_options_chain)

**Phase 2 — Moderate signal, moderate effort:**
5. Job postings count via Lever/Greenhouse public APIs
6. Shipping rate scraper (Freightos)
7. Amazon BSR via Keepa API
8. FinBERT news sentiment (local model)

**Phase 3 — Useful but lower priority:**
9. US Customs trade data (Census API)
10. Pricing snapshot scraper
11. Port throughput downloader
12. EU/China/Japan regulatory scraper

**Phase 4 — Paid data (if budget allows):**
13. Placer.ai foot traffic (retail names)
14. Proxycurl LinkedIn headcount
15. ImportGenius customs (shipper-level, vs Census aggregate)

---

## Key Constraints

- **Rate limits:** pytrends blocks aggressively on bulk requests — use 1s sleep
  between calls, randomize user agent, use VPN rotation if needed
- **ToS:** LinkedIn scraping is against ToS; Lever/Greenhouse is explicitly public
- **Latency:** Census trade data has 5-week lag; MOPS has 10-day lag; both are
  still very useful relative to earnings
- **Model dependency:** FinBERT requires ~440MB local model download; runs on
  CPU acceptably (2-3s per article)
- **Data normalization:** all trend signals must be normalized against the same
  period last year to remove seasonality before comparison
- **Currency:** international revenue companies need FX adjustment on international
  signals (a Chinese demand signal in RMB needs USD/CNY rate applied)
