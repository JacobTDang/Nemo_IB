"""A dollar figure in a headline is not automatically capital expenditure.

    get_capex_announcements(ticker="NVDA", days=90)
        -> total_announced_usd 1,620,400,000,000     ($1.62 TRILLION)
           signal "bullish"

NVIDIA's capital expenditure is single-digit billions. The tool was out by two
orders of magnitude, and every component of the error was a real figure read as
the wrong kind of money:

    $750B   deals NVIDIA is signing WITH CUSTOMERS -- revenue, not capex
    $500B   third-party capital mobilised by Apollo / BlackRock / KKR
            -- not NVIDIA's money at all
    $250B   a lease GUARANTEE
    $105B   a financing BACKSTOP for OpenAI's Ohio site
    $10B    a Naver / Brookfield joint venture

AVGO had the same shape -- $341.5B, of which $230B was new bookings, $80B a
DEBT RAISE, $30B Apple buying chips FROM Broadcom, and $1.5B (the Fort Collins
upgrade) the only actual capital expenditure. TSM returned $365B by adding
$265B to $100B when the $265B article says in as many words that it already
includes the $100B.

Summing distinct figures instead of one per article -- the fix in
test_capex_total_is_a_total.py -- was a real improvement and does not touch
this. The figures being summed are not capex.

The rule this file holds the tool to: a figure enters a total labelled capex
only when the text says the company is SPENDING it on something physical, and
never when the text says it is being financed, guaranteed, raised, mobilised
from someone else, invested in another company's equity, paid TO the company by
a customer, or restated as a running programme total. A figure that cannot be
placed is not capex either -- it is unplaced, and an unplaced figure never
enters the total. Where nothing can be placed the tool has to say so: a news
corpus cannot assert that no capex was announced, so the answer is no total,
not a zero and certainly not $1.62tn.

An announcement of SPENDING and an announcement of REVENUE are opposite
signals about the same company. A caller has to be able to tell them apart
from the response, which means every figure carries its category and the words
that put it there.

The corpus below is verbatim from live DuckDuckGo news calls for NVDA, AVGO
and TSM -- titles, bodies, dates and urls as returned. The defect lives in how
real headlines are read, so invented ones would not exercise it. Because the
corpus is a snapshot, the date window is opened wide rather than left to expire
the fixtures as they age.
"""
import pytest

import tools.altdata_server.server as alt

# The captured corpus is a snapshot; a real lookback would silently empty it
# as the articles age past the cutoff and turn every test below green for the
# wrong reason.
_NO_DATE_FILTER = 100_000


class _FakeDDGS:
    """The ddgs context manager, minus the network."""

    def __init__(self, results):
        self._results = results
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def news(self, query, max_results=10, timelimit=None):
        self.queries.append(query)
        return list(self._results)


def _install(monkeypatch, results):
    import sys
    import types
    module = types.ModuleType("ddgs")
    fake = _FakeDDGS(results)
    module.DDGS = lambda *a, **k: fake
    monkeypatch.setitem(sys.modules, "ddgs", module)
    return fake


# ---------------------------------------------------------------------------
# Live corpora, verbatim
# ---------------------------------------------------------------------------

_NVDA_CORPUS = [
    {"title": "Who wins most in NVIDIA’s $500 billion private capital deal?",
     "body": "NVIDIA’s deal to mobilize over $500 billion in third-party capital with "
             "Apollo, BlackRock, Blackstone, Brookfield, Goldman Sachs, and KKR reframes "
             "how AI factories are built and financed. By turning ...",
     "date": "2026-08-11T05:00:00+00:00",
     "url": "https://finance.yahoo.com/technology/ai/articles/wins-most-nvidia-500-billion-133032066.html"},
    {"title": "Nvidia deepens South Korea AI push with $1 billion NAVER investment",
     "body": "NVIDIA Corp NVDA plans to invest about $1 billion in NAVER Corp as part of a "
             "broader effort to expand artificial intelligence datacenter infrastructure "
             "in South Korea. Announced Saturday during South ...",
     "date": "2026-07-27T05:00:00+00:00",
     "url": "https://www.msn.com/en-us/money/technology/nvidia-deepens-south-korea-ai-push-with-1-billion-naver-investment/ar-AA28L9Pv"},
    {"title": "Naver and NVIDIA triple Korea's AI factory to 200 megawatts in $10 billion "
              "deal with Brookfield",
     "body": "South Korea's national AI factory just tripled in planned scale. NAVER, "
             "NVIDIA, and Brookfield Asset Management announced on July 25 that they will "
             "expand NAVER's NVIDIA DSX AI factory at the GAK ...",
     "date": "2026-07-26T05:00:00+00:00",
     "url": "https://www.msn.com/en-us/money/general/naver-and-nvidia-triple-korea-s-ai-factory/ar-AA28I88Y"},
    {"title": "NVIDIA’s $3 Billion Bet on the Power Behind AI",
     "body": "NVIDIA Corporation (NASDAQ:NVDA)’s latest push into the infrastructure "
             "supporting AI chips extends all the way to the power grid. According to a "
             "report f...",
     "date": "2026-08-19T20:17:44+00:00",
     "url": "https://finance.yahoo.com/technology/ai/articles/nvidia-3-billion-bet-power-202314612.html"},
    {"title": "Verkada takes Nvidia investment to expand its physical AI platform",
     "body": "Physical security company Verkada Inc. has taken an investment from Nvidia "
             "Corp. and signed a technical partnership with the chipmaker, the two said "
             "today, in a deal meant to speed up the artificial ...",
     "date": "2026-07-01T05:00:00+00:00",
     "url": "https://siliconangle.com/2026/07/01/verkada-takes-nvidia-investment-expand-physical-ai-platform/"},
]

_AVGO_CORPUS = [
    {"title": "Broadcom Lands $230 Billion in New Deals, Yet Trades at Steep Discount",
     "body": "A sign is posted in front of a Broadcom office on December 12, 2024 in San "
             "Jose, California. Justin Sullivan/Getty Images Broadcom Inc. (NASDAQ: AVGO) "
             "closed July 25 by signing the largest ...",
     "date": "2026-07-28T05:00:00+00:00",
     "url": "https://www.techtimes.com/articles/321808/20260728/broadcom-lands-230-billion-new-deals.htm"},
    {"title": "Broadcom could raise up to $80 billion as Anthropic AI buildout drives "
              "massive financing push",
     "body": "The senior tranche, which would have priority for repayment, is expected to "
             "total roughly $45 billion, while a junior tranche ...",
     "date": "2026-08-24T20:17:45+00:00",
     "url": "https://www.msn.com/en-us/money/markets/broadcom-could-raise-up-to-80-billion/ar-AA2aOP0a"},
    {"title": "Broadcom to expand U.S. facility after $30 billion chip deal with Apple",
     "body": "CNBC's MacKenzie Sigalos reports on Apple and Broadcom's $30 billion chip "
             "deal. Got a confidential news tip? We want to hear from you. Sign up for "
             "free newsletters and get more CNBC delivered to your ...",
     "date": "2026-07-08T05:00:00+00:00",
     "url": "https://www.cnbc.com/video/2026/07/08/broadcom-expands-us-facility-after-30-billion-chip-deal-with-apple.html"},
    {"title": "The Apple/Broadcom Deal: What It Is And, More Importantly, What It Is Not",
     "body": "Broadcom secured a new multi-year, $30B supply agreement with Apple, "
             "extending their RF component partnership. The agreement extends the "
             "long-standing Apple/Broadcom collaboration until 2031, ...",
     "date": "2026-07-08T05:00:00+00:00",
     "url": "https://seekingalpha.com/article/4920672-the-applebroadcom-deal"},
    {"title": "Apple shares new details of chip supply deal with AVGO – iPhone maker "
              "commits $1.5B to expand Broadcom’s manufacturing",
     "body": "Broadcom previously disclosed in an SEC filing that the agreement with Apple "
             "will run through 2031. The details of the agreement come weeks after Apple "
             "raised prices across its product portfolio, ...",
     "date": "2026-07-08T05:00:00+00:00",
     "url": "https://www.msn.com/en-in/money/technology/apple-shares-new-details-of-chip-supply-deal-with-avgo/ar-AA27tmSK"},
    {"title": "Apple Expands Broadcom Deal Beyond $30 Billion, Shares Rise 5%",
     "body": "The above button links to Coinbase. Yahoo Finance is not a broker-dealer or "
             "investment adviser and does not offer securities or cryptocurrencies for "
             "sale or facilitate trading. Coinbase pays us for ...",
     "date": "2026-07-08T05:00:00+00:00",
     "url": "https://finance.yahoo.com/markets/stocks/articles/apple-expands-broadcom-deal-beyond-162211002.html"},
]

_TSM_CORPUS = [
    {"title": "TSMC eyes 4 more Arizona chip plants after $100 billion expansion",
     "body": "The new investment, announced June 16, brings Taiwan Semiconductor "
             "Manufacturing Co.'s U.S. investment to $265 billion.",
     "date": "2026-07-16T05:00:00+00:00",
     "url": "https://www.azcentral.com/story/money/business/energy/2026/07/16/tsmc-eyes-4-more-arizona-chip-plants/90941055007/"},
    {"title": "Taiwan Semiconductor Manufacturing just showed the artificial intelligence "
              "(AI) build-out is alive and well with this jaw-dropping announcement",
     "body": "Taiwan Semiconductor just announced another $100 billion investment to expand "
             "its foundry operations in Arizona. The stock has sold off over the past few "
             "weeks. TSMC trades at an attractive forward ...",
     "date": "2026-07-19T05:00:00+00:00",
     "url": "https://www.msn.com/en-us/money/topstocks/taiwan-semiconductor-manufacturing-just-showed/ar-AA28fj2o"},
    {"title": "Taiwan Semiconductor commits to investing another $100 billion in the "
              "United States",
     "body": "Taiwan Semiconductor just announced an additional $100 billion investment in "
             "U.S. semiconductor fabrication plants. The company's Arizona fabs produce "
             "some of its most advanced microchips. The move ...",
     "date": "2026-07-16T05:00:00+00:00",
     "url": "https://www.msn.com/en-us/money/general/taiwan-semiconductor-commits-to-investing-another-100-billion/ar-AA284ryO"},
    {"title": "Taiwan computer chipmaker TSMC pledges another $100 billion to expand US "
              "chipmaking capacity",
     "body": "HONG KONG (AP) — Major Taiwan computer chipmaker TSMC said Thursday it "
             "plans to spend another $100 billion on expanding its manufacturing capacity "
             "in the United States. The latest commitment appears ...",
     "date": "2026-07-16T05:00:00+00:00",
     "url": "https://www.msn.com/en-us/money/companies/taiwan-computer-chipmaker-tsmc-pledges-another-100-billion/ar-AA281n1U"},
    {"title": "Another $100 billion",
     "body": "In its earnings announcement, Taiwan Semiconductor Chairman C.C. Wei "
             "announced the company is putting some of its record profits to work in the "
             "U.S. by committing an additional $100 billion to its ...",
     "date": "2026-07-16T05:00:00+00:00",
     "url": "https://www.fool.com/investing/2026/07/16/taiwan-semiconductor-commits-to-investing-another-100-billion/"},
]


@pytest.fixture
def nvda(monkeypatch):
    _install(monkeypatch, _NVDA_CORPUS)
    return alt._fetch_capex_announcements("NVDA", "NVIDIA Corporation", _NO_DATE_FILTER)


@pytest.fixture
def avgo(monkeypatch):
    _install(monkeypatch, _AVGO_CORPUS)
    return alt._fetch_capex_announcements("AVGO", "Broadcom Inc", _NO_DATE_FILTER)


@pytest.fixture
def tsm(monkeypatch):
    _install(monkeypatch, _TSM_CORPUS)
    return alt._fetch_capex_announcements(
        "TSM", "Taiwan Semiconductor Manufacturing Company", _NO_DATE_FILTER)


def _category_of(payload, amount):
    """The verdict on one dollar figure, whichever article carried it."""
    figs = {f["amount_usd"]: f for f in payload["figures"]}
    assert amount in figs, (
        f"${amount:,.0f} is not in the payload at all; found "
        f"{sorted(figs, reverse=True)}")
    return figs[amount]["category"]


# ---------------------------------------------------------------------------
# Money paid TO the company is not money spent BY it
# ---------------------------------------------------------------------------

def test_a_customer_deal_is_not_capital_expenditure(avgo):
    """"Broadcom Lands $230 Billion in New Deals" is order intake. It is the
    single largest figure in the AVGO corpus and it was two-thirds of the
    $341.5B "capex" total. Revenue and capex are opposite signals: one says
    customers are paying Broadcom, the other says Broadcom is paying for
    plant. Adding them together answers neither question."""
    assert _category_of(avgo, 230e9) == "customer_or_partner_deal"


def test_a_customer_buying_chips_is_not_the_seller_spending(avgo):
    """Apple's $30B chip commitment is Broadcom REVENUE. The headline that
    carried it -- "Broadcom to expand U.S. facility after $30 billion chip deal
    with Apple" -- also contains real expansion language, which is exactly how
    it was read as capex: the article-wide keyword scan saw "expand" and
    "facility" and attached them to the biggest number on the page."""
    assert _category_of(avgo, 30e9) == "customer_or_partner_deal"


def test_third_party_capital_is_not_the_companys_money(nvda):
    """$500B "mobilized in third-party capital with Apollo, BlackRock,
    Blackstone, Brookfield, Goldman Sachs, and KKR" is other people's balance
    sheets. It never touches NVIDIA's capital expenditure and it was the
    largest single line in the $1.62tn."""
    assert _category_of(nvda, 500e9) == "third_party_capital"


def test_a_debt_raise_is_not_capital_expenditure(avgo):
    """"Broadcom could raise up to $80 billion" is a liability, not an asset.
    Money raised may later be spent, but counting the raise AND the spend
    counts the same dollars twice, and counting a raise on its own reports
    borrowing as investment."""
    assert _category_of(avgo, 80e9) == "financing"


def test_a_tranche_inside_a_financing_is_financing_too(avgo):
    """The $45B senior tranche is a slice of the same $80B raise. Both figures
    are in the same article and the old total would have taken the larger; a
    per-figure reading has to keep the smaller one out of capex as well, or
    fixing the headline figure just moves the error down a line."""
    assert _category_of(avgo, 45e9) == "financing"


def test_an_equity_investment_in_another_company_is_not_capex(nvda):
    """"plans to invest about $1 billion in NAVER Corp" buys shares in NAVER.
    The same sentence goes on to say "to expand artificial intelligence
    datacenter infrastructure", which is precisely the phrase a capex
    classifier keys on -- so the money spent on someone else's equity reads as
    money spent on plant unless the object of the investment is looked at."""
    assert _category_of(nvda, 1e9) == "equity_or_ma"


def test_a_joint_venture_deal_is_not_the_companys_own_capex(nvda):
    """The $10B AI-factory expansion is NAVER's site, with Brookfield's
    capital, announced by three parties. It may well be a capex event for
    someone; it is not $10B of NVIDIA capital expenditure, and a tool that
    cannot say whose money it is must not put it in NVIDIA's total."""
    assert _category_of(nvda, 10e9) == "customer_or_partner_deal"


# ---------------------------------------------------------------------------
# What is left is the total
# ---------------------------------------------------------------------------

def test_the_total_sums_only_figures_that_are_capital_expenditure(avgo):
    """Of AVGO's $341.5B, exactly one figure is a company spending its own
    money on physical capacity: the $1.5B Fort Collins upgrade. That is the
    number, and it is 0.4% of what the tool used to report."""
    assert _category_of(avgo, 1.5e9) == "capital_expenditure"
    assert avgo["capex_total_usd"] == 1.5e9


def test_the_basis_says_which_figures_were_left_out_and_why(avgo):
    """A corrected total that does not show its working is as unauditable as
    the wrong one. The reader has to see that $230B of bookings and $80B of
    debt were seen, categorised and deliberately excluded -- not missed."""
    basis = avgo["capex_total_basis"]
    assert basis
    excluded = avgo["amounts_by_category"]
    assert excluded["customer_or_partner_deal"]["total_usd"] == 260e9
    assert excluded["financing"]["total_usd"] == 125e9


def test_no_capex_figure_means_no_total_rather_than_a_zero(nvda):
    """Not one figure in the NVDA corpus is NVIDIA spending on plant, which is
    correct: NVIDIA's capex does not make headlines. A zero would claim the
    corpus proves none was announced, which a news search cannot do. So the
    total is withheld and the payload says why."""
    assert nvda["capex_total_usd"] is None
    assert nvda["capex_figure_count"] == 0
    assert nvda["capex_total_basis"]


def test_the_figures_are_still_reported_when_the_total_is_withheld(nvda):
    """Refusing the aggregate must not throw the evidence away. A list of
    announcements with amounts, categories and sources is the useful part; it
    is the fabricated $1.62tn on top of it that was not."""
    assert nvda["announcement_count"] == 5
    by_cat = nvda["amounts_by_category"]
    assert by_cat["third_party_capital"]["total_usd"] == 500e9
    assert 500e9 in by_cat["third_party_capital"]["amounts_usd"]


def test_an_unplaceable_figure_never_enters_the_total(nvda):
    """"NVIDIA's $3 Billion Bet on the Power Behind AI" -- a bet on what? The
    text does not say whether NVIDIA is building, buying or backing anything.
    Unplaced is the honest answer and unplaced money is not capex."""
    assert _category_of(nvda, 3e9) == "unclassified"
    assert nvda["capex_total_usd"] is None


# ---------------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------------

def test_a_running_programme_total_is_not_a_new_announcement(tsm):
    """"brings Taiwan Semiconductor Manufacturing Co.'s U.S. investment to $265
    billion" is the cumulative programme, and the $100B announced that week is
    inside it. Adding the two gave $365B for $100B of news."""
    assert _category_of(tsm, 265e9) == "cumulative_total"
    assert _category_of(tsm, 100e9) == "capital_expenditure"
    assert tsm["capex_total_usd"] == 100e9


def test_the_containment_is_named_rather_than_silently_dropped(tsm):
    """Dropping the $265B without saying so would hide the largest number in
    the corpus. The caller needs both: $100B was announced, against a
    programme the same article puts at $265B."""
    assert tsm["containment_detected"] is True
    assert tsm["cumulative_program_usd"] == 265e9
    assert "265" in tsm["capex_total_basis"] or "cumulative" in tsm["capex_total_basis"]


def test_one_announcement_restated_by_five_outlets_is_still_one(tsm):
    """Five of the TSM articles carry the same $100B. The distinct-figure rule
    from test_capex_total_is_a_total.py still has to hold once the figures are
    classified, or the classification fix would reintroduce the counting bug."""
    assert tsm["capex_figure_count"] == 1
    assert tsm["capex_total_usd"] == 100e9


# ---------------------------------------------------------------------------
# The signal
# ---------------------------------------------------------------------------

def test_the_signal_is_not_derived_from_revenue_or_financing(nvda):
    """"bullish" off $500B of somebody else's capital and a $10B joint venture
    is a verdict with nothing under it. A signal derived from a total that is
    two orders of magnitude wrong is worse than no signal."""
    assert nvda["signal"] == "data_gap"
    assert nvda["signal_basis"]


def test_a_real_capex_announcement_still_produces_a_signal(tsm):
    """The refusal must not swallow the case the tool exists for. TSMC
    announcing $100B of fab construction is a genuine capex event and reads
    bullish."""
    assert tsm["signal"] == "bullish"
    assert tsm["signal_basis"]


def test_the_avgo_signal_rests_on_the_fort_collins_upgrade_alone(avgo):
    """AVGO still reads bullish, but for $1.5B of plant rather than $230B of
    bookings. Same verdict, and now a defensible one -- which is the point:
    the basis has to name the figures it used."""
    assert avgo["signal"] == "bullish"
    assert "1,500,000,000" in avgo["signal_basis"] or "1.5" in avgo["signal_basis"]


# ---------------------------------------------------------------------------
# The shape of the answer
# ---------------------------------------------------------------------------

def test_every_figure_carries_its_category_and_the_words_behind_it(avgo):
    """A category with no evidence is an assertion. The matched phrase is what
    lets a caller overrule the classifier instead of trusting it."""
    figs = avgo["figures"]
    assert figs
    for f in figs:
        assert f["category"], f
        assert f["evidence"], f
        assert f["context"], f


def test_the_undifferentiated_total_is_gone(nvda, avgo, tsm):
    """`total_announced_usd` mixed capex, revenue, debt and guarantees under
    one name. Renaming it is deliberate: a caller reading the old key must get
    a KeyError, not a number a hundred times smaller than it expects."""
    for payload in (nvda, avgo, tsm):
        assert "total_announced_usd" not in payload
        assert "largest_announcement_usd" not in payload


def test_a_row_says_what_kind_of_money_its_biggest_number_is(avgo):
    """The row-level `max_amount_usd` was the field callers read as "the capex
    in this article". On the Apple headline it was $30B of revenue. It now
    carries its category beside it and cannot be mistaken for a spend."""
    row = next(r for r in avgo["announcements"] if r["largest_figure_usd"] == 230e9)
    assert row["figure_category"] == "customer_or_partner_deal"
    assert row["capex_amount_usd"] is None


def test_the_rows_the_total_was_built_from_survive_truncation(avgo):
    """Rows are capped at 8 and were sorted by size, so the one $1.5B row the
    total rests on sorts below six larger non-capex headlines. A total whose
    evidence has been paged away cannot be checked."""
    assert avgo["announcements"][0]["capex_amount_usd"] == 1.5e9


# ---------------------------------------------------------------------------
# The case the tool exists for
# ---------------------------------------------------------------------------

def test_a_plain_capex_announcement_is_still_recognised(monkeypatch):
    """The whole point of tightening the classifier is that it must still fire
    on the easy case: a named company, a spend verb and a physical asset in the
    same clause."""
    _install(monkeypatch, [
        {"title": "Intel to invest $20 billion in new Ohio factory",
         "body": "Intel said it will build the site.", "date": "", "url": "u"},
    ])
    out = alt._fetch_capex_announcements("INTC", "Intel Corporation", 90)

    assert out["capex_total_usd"] == 20e9
    assert out["signal"] == "bullish"
    assert _category_of(out, 20e9) == "capital_expenditure"


def test_a_cancelled_plant_is_capex_and_reads_bearish(monkeypatch):
    """Direction is a separate axis from category. A shelved fab is a capital
    expenditure announcement -- a negative one -- and must not be laundered
    into "unclassified" by a classifier that only recognises spending. A
    cancellation is the case a capex tool most needs to get right and the one
    least likely to appear in a corpus captured during a build-out, so it is
    put here by hand."""
    _install(monkeypatch, [
        {"title": "Intel scraps $20 billion Ohio fab, halting construction",
         "body": "Intel said the project is shelved.", "date": "", "url": "u"},
    ])
    out = alt._fetch_capex_announcements("INTC", "Intel Corporation", 90)

    assert _category_of(out, 20e9) == "capital_expenditure"
    assert out["signal"] == "bearish"


def test_another_companys_spending_is_not_this_companys_capex(monkeypatch):
    """The relevance filter keeps an article because it mentions
    "semiconductor" somewhere; that is not the same as the figure belonging to
    TSMC. Samsung's $520B has to fail attribution, or the tool reports a
    competitor's programme as its subject's capex."""
    _install(monkeypatch, [
        {"title": "Samsung and SK Hynix to build four new chip plants as South Korea "
                  "unveils $520 billion plan",
         "body": "Samsung Electronics Co., Ltd. and SK Hynix plan to build two new "
                 "semiconductor fabrication plants.", "date": "", "url": "u"},
    ])
    out = alt._fetch_capex_announcements(
        "TSM", "Taiwan Semiconductor Manufacturing Company", 90)

    assert out["capex_total_usd"] is None
    assert _category_of(out, 520e9) != "capital_expenditure"


def test_a_suppliers_spending_is_not_the_companys_spending(monkeypatch):
    """Live NVDA, verbatim. "Nvidia" is in the clause, "invest" is in the
    clause, "facility" is in the clause -- and the company doing all three is
    King Yuan Electronics. Requiring the name somewhere in the sentence is not
    the same as requiring it to be the spender, and the difference put $1.4B of
    a supplier's capex into NVIDIA's total."""
    _install(monkeypatch, [
        {"title": "Nvidia supplier King Yuan Electronics to invest up to $1.4 billion "
                  "in US facility",
         "body": "", "date": "", "url": "u"},
    ])
    out = alt._fetch_capex_announcements("NVDA", "NVIDIA Corporation", 90)

    assert _category_of(out, 1.4e9) != "capital_expenditure"
    assert out["capex_total_usd"] is None


def test_the_role_word_can_sit_on_either_side_of_the_name(monkeypatch):
    """The same King Yuan story, the other way round -- live NVDA, verbatim.
    "the supplier to chipmaker Nvidia" puts the relationship BEFORE the name
    instead of after it, and a guard that only looks forward from the name
    lets $1.4B of somebody else's fab straight back into the total."""
    _install(monkeypatch, [
        {"title": "Nvidia supplier to build US plant",
         "body": "Taiwanese chip-testing company King Yuan Electronics (KYEC) plans to "
                 "invest up to $1.4 billion to establish a facility in the United "
                 "States, the supplier to chipmaker Nvidia said on Friday.",
         "date": "", "url": "u"},
    ])
    out = alt._fetch_capex_announcements("NVDA", "NVIDIA Corporation", 90)

    assert _category_of(out, 1.4e9) != "capital_expenditure"
    assert out["capex_total_usd"] is None


def test_money_being_weighed_is_not_money_announced(monkeypatch):
    """Live NVDA, verbatim. Nvidia WEIGHS an investment in SB Energy while
    OpenAI EXPANDS an Ohio data centre -- three subjects in one headline, and
    the classifier took the company from the first, the verb from the third and
    the asset from the third. A capex announcement is announced; a figure under
    consideration is not one, whoever is considering it."""
    _install(monkeypatch, [
        {"title": "Nvidia Weighs $3 Billion SB Energy Investment As OpenAI Expands "
                  "Ohio Data Center Push",
         "body": "", "date": "", "url": "u"},
    ])
    out = alt._fetch_capex_announcements("NVDA", "NVIDIA Corporation", 90)

    assert _category_of(out, 3e9) != "capital_expenditure"
    assert out["capex_total_usd"] is None


def test_a_lookup_that_found_nothing_answers_the_capex_fields(monkeypatch):
    """The no-results branch is a separate return and must not start raising
    KeyError on a caller reading the fields the success branch promises. It
    withholds the total for the same reason NVDA does: no article is not the
    same finding as no capex."""
    _install(monkeypatch, [])
    out = alt._fetch_capex_announcements("KO", "Coca-Cola Company", 90)

    assert out["success"] is False
    assert out["reason"] == "no_results"
    assert out["capex_total_usd"] is None
    assert out["capex_total_basis"]
