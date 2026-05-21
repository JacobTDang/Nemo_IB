"""HEAVY end-to-end stress test for the RAG retrieval layer.

This is the quality gate for the RAG layer. It exercises four dimensions:

  1. Bootstrap correctness   - the bootstrap script runs cleanly and the
                                corpus has the expected doc_type spread.
  2. Ground-truth retrieval  - 30+ curated Q&A pairs; measure P@1, P@5,
                                and MRR across categories (filings,
                                analyst, analogues, rules, negatives,
                                paraphrases, cross-document).
  3. Adversarial robustness  - paraphrases, ambiguous one-word queries,
                                out-of-corpus negatives, polarity flips,
                                cross-document multi-doc spread.
  4. Volume + latency        - ingest 2000+ synthetic chunks and measure
                                ingestion throughput; query latency
                                percentiles; vector-index correctness.

The final report is written to testing/fixtures/rag_stress_report.md as
a human-readable engineering document, regardless of whether the gates
pass. Synthetic stress chunks (STRESS_VOL_ prefix) are deleted at end.

Hard rules honored from the spec:
  - Does not touch production code, only this test file and the report.
  - Cleans up STRESS_VOL_ / STRESS_TEST_ docs at end.
  - Leaves the bootstrapped corpus intact for downstream phases.
  - Generates the report even when gates fail.
  - Targets sub-10-minute total runtime by batching embeddings.

Run via:
  ./.venv/Scripts/python.exe testing/test_rag_heavy_stress.py
"""
from __future__ import annotations

import json
import os
import random
import statistics
import string
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.rag import store  # noqa: E402
from agent.rag.embedder import embed, embed_batch  # noqa: E402
from agent.rag.ingest import ingest_document, ingest_extractor_output  # noqa: E402
from agent.rag.search import rag_search  # noqa: E402
from state.schema import get_connection, init_schema  # noqa: E402

# Try to import the SEC extractors. They hit the network; if unavailable
# the test still runs but Section 1 step 2 will skip those ingestions.
try:
    from tools.web_search_server.sec_utils import (  # noqa: E402
        extract_mda,
        extract_risk_factors,
        get_earnings_releases,
        get_supply_chain,
    )
    _HAVE_SEC_EXTRACTORS = True
except Exception as exc:  # pragma: no cover
    _HAVE_SEC_EXTRACTORS = False
    _SEC_IMPORT_ERR = str(exc)


# ---------------------------------------------------------------------------
# Harness — same shape as the other rag tests
# ---------------------------------------------------------------------------

_results = {"pass": 0, "fail": 0, "failures": []}

# Test prefixes we own. Any doc we insert during this run uses one of
# these so cleanup is targeted.
_VOL_PREFIX = "STRESS_VOL_"
_STRESS_PREFIX = "STRESS_TEST_"

# Container for the final report payload.
_REPORT: Dict[str, Any] = {
    "started_at":      None,
    "finished_at":     None,
    "bootstrap":       {},
    "corpus_stats":    {},
    "qa_metrics":      {},
    "adversarial":     {},
    "volume":          {},
    "gates":           {},
    "top_failures":    [],
}


def _check(name: str, condition: bool, hint: str = "") -> None:
    if condition:
        _results["pass"] += 1
        print(f"  PASS  {name}")
    else:
        _results["fail"] += 1
        _results["failures"].append((name, hint))
        print(f"  FAIL  {name}  --  {hint}")


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _cleanup_prefix(prefix: str) -> int:
    """Delete every doc_id starting with `prefix`. Returns count deleted."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT doc_id FROM rag_chunks WHERE doc_id LIKE ?",
            (prefix + "%",),
        ).fetchall()
        ids = [r["doc_id"] for r in rows]
    finally:
        conn.close()
    n = 0
    for did in ids:
        try:
            store.delete_by_doc_id(did)
            n += 1
        except Exception:
            pass
    return n


# ---------------------------------------------------------------------------
# Section 1 — Bootstrap + corpus expansion
# ---------------------------------------------------------------------------

def section_1_bootstrap() -> Dict[str, Any]:
    _section("1. Bootstrap + corpus expansion")
    info: Dict[str, Any] = {}

    # Step 1: run the bootstrap script via subprocess. It is idempotent.
    bootstrap_path = os.path.join(PROJECT_ROOT, "scripts", "bootstrap_rag_corpus.py")
    python_exe = sys.executable
    t0 = time.perf_counter()
    proc = subprocess.run(
        [python_exe, bootstrap_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    dt = time.perf_counter() - t0
    info["bootstrap_returncode"] = proc.returncode
    info["bootstrap_seconds"]    = round(dt, 2)
    # Pull the reported chunk count from the script's TOTAL line.
    chunk_count_reported = None
    for line in (proc.stdout or "").splitlines():
        if "rag_chunks rows in store" in line:
            try:
                chunk_count_reported = int(line.rsplit(":", 1)[-1].strip())
            except ValueError:
                pass
    info["bootstrap_reported_chunks"] = chunk_count_reported

    _check(
        "bootstrap script exits cleanly",
        proc.returncode == 0,
        f"returncode={proc.returncode}, stderr_tail={(proc.stderr or '')[-200:]}",
    )

    pre_count = store.count_chunks()
    print(f"  chunks after bootstrap: {pre_count}")

    # Step 2: ingest live MSFT structured data. Failures here are noted in
    # the report but do not fail the suite — the network may be unstable.
    live_ingests: List[Dict[str, Any]] = []
    if _HAVE_SEC_EXTRACTORS:
        live_jobs = [
            ("extract_risk_factors",  lambda: extract_risk_factors("MSFT")),
            ("extract_mda",           lambda: extract_mda("MSFT")),
            ("get_earnings_releases", lambda: get_earnings_releases("MSFT", max_quarters=2)),
            ("get_supply_chain",      lambda: get_supply_chain("NVDA")),
        ]
        for tool_name, runner in live_jobs:
            entry: Dict[str, Any] = {"tool": tool_name}
            try:
                t1 = time.perf_counter()
                result = runner()
                entry["extract_seconds"] = round(time.perf_counter() - t1, 2)
                entry["extract_success"] = bool(result and result.get("success"))
                if not entry["extract_success"]:
                    entry["error"] = (result or {}).get("error") or "no payload"
                    print(f"  live ingest SKIPPED {tool_name}: {entry['error']}")
                    live_ingests.append(entry)
                    continue
                ing = ingest_extractor_output(tool_name, result)
                entry["docs_ingested"]         = ing.get("docs_ingested", 0)
                entry["total_chunks_inserted"] = ing.get("total_chunks_inserted", 0)
                print(f"  live ingest {tool_name}: {entry['docs_ingested']} docs, "
                      f"{entry['total_chunks_inserted']} chunks")
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  live ingest FAILED {tool_name}: {entry['error']}")
            live_ingests.append(entry)
    else:
        print(f"  live extractors unavailable: {_SEC_IMPORT_ERR}")
    info["live_ingests"] = live_ingests

    # Step 3: corpus-size gate. The spec wants >= 150 chunks. If live
    # ingestion failed, the bootstrap alone might be short — we still
    # check the gate but the live failure shows up in the report.
    post_count = store.count_chunks()
    info["chunks_after_live_ingest"] = post_count
    _check(
        "rag_chunks count >= 150 after additions",
        post_count >= 150,
        f"count={post_count}",
    )

    # Step 4: doc_type spread.
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT doc_type, COUNT(*) AS n FROM rag_chunks GROUP BY doc_type"
        ).fetchall()
        type_breakdown = {r["doc_type"]: r["n"] for r in rows}
        rows2 = conn.execute(
            "SELECT ticker, COUNT(*) AS n FROM rag_chunks GROUP BY ticker"
        ).fetchall()
        ticker_breakdown = {(r["ticker"] or "(none)"): r["n"] for r in rows2}
    finally:
        conn.close()

    info["doc_type_breakdown"] = type_breakdown
    info["ticker_breakdown"]   = ticker_breakdown
    print(f"  doc_type spread: {type_breakdown}")
    print(f"  ticker spread:   {ticker_breakdown}")

    # The four canonical bootstrap types must be present. The live types
    # (10K_*, earnings_release, supply_chain_signals) are best-effort.
    required = {"analogue", "analyst_writeup", "rule"}
    _check(
        "all required bootstrap doc_types present",
        required.issubset(set(type_breakdown.keys())),
        f"missing={required - set(type_breakdown.keys())}",
    )
    # Soft check: at least one of the live types showed up.
    live_types = {"10K_risk_factors", "10K_mda", "earnings_release",
                  "supply_chain_signals", "forward_signal"}
    overlap = live_types & set(type_breakdown.keys())
    _check(
        "at least one live/extractor doc_type ingested",
        len(overlap) >= 1,
        f"observed_live_types={overlap}",
    )

    return info


# ---------------------------------------------------------------------------
# Section 2 — Ground-truth Q&A retrieval
# ---------------------------------------------------------------------------

# Ground-truth pairs. Schema per row:
#   (query, expected_substr_or_None, ticker_or_None, doc_type_or_None, category)
# - expected_substr_or_None: a case-insensitive substring that must appear
#   in the chunk_text of a "correct" result. None => out-of-corpus.
# - ticker_or_None: optional retrieval-time filter AND correctness check.
# - doc_type_or_None: optional retrieval-time filter AND correctness check.
# Filters are applied at retrieval time; correctness is also checked
# against ticker/doc_type so spec compliance is end-to-end.

GROUND_TRUTH: List[Tuple[str, Optional[str], Optional[str], Optional[str], str]] = [
    # ---- Filings / structured ----
    ("Azure cloud revenue growth",                       "azure",        "MSFT", None,              "filings"),
    ("Microsoft Cloud commercial remaining performance", "remaining performance", "MSFT", None,     "filings"),
    ("Microsoft revenue and cloud highlights",           "microsoft cloud", "MSFT", None,           "filings"),
    ("MSFT productivity and business processes revenue", "productivity", "MSFT", None,              "filings"),
    ("forward-looking guidance from Microsoft earnings", "guidance",     "MSFT", "forward_signal",  "filings"),

    # ---- Analyst writeups ----
    ("Intelligent Cloud margin compression",             "margin",       "MSFT", "analyst_writeup", "analyst"),
    ("MSFT IB analyst playbook verdict",                 "msft",         "MSFT", "analyst_writeup", "analyst"),
    ("Phase B3 pilot analyst writeup",                   "phase b3",     "MSFT", "analyst_writeup", "analyst"),
    ("Bull case versus bear case for Microsoft",         "msft",         "MSFT", "analyst_writeup", "analyst"),

    # ---- Analogues ----
    ("1999 dot-com capex peak",                          "dot-com",      None,   "analogue",        "analogues"),
    ("memory cycle DRAM glut",                           "dram",         None,   "analogue",        "analogues"),
    ("2007 2008 housing financial crisis",               "housing",      None,   "analogue",        "analogues"),
    ("AI hyperscaler capex cycle",                       "hyperscaler",  None,   "analogue",        "analogues"),
    ("SPAC reflexivity bust 2021",                       "spac",         None,   "analogue",        "analogues"),
    ("oil shale collapse 2014 2016",                     "shale",        None,   "analogue",        "analogues"),
    ("smartphone mobile cycle 2010",                     "smartphone",   None,   "analogue",        "analogues"),
    ("COVID cloud acceleration",                         "covid",        None,   "analogue",        "analogues"),
    ("subprime auto consumer credit",                    "subprime",     None,   "analogue",        "analogues"),
    ("China infrastructure stimulus",                    "china",        None,   "analogue",        "analogues"),
    ("energy capex drought 2015 2020",                   "capex",        None,   "analogue",        "analogues"),

    # ---- Rules ----
    ("LLM proposes tools execute",                       "llm",          None,   "rule",            "rules"),
    ("SEC data grounding",                               "sec",          None,   "rule",            "rules"),
    ("project structure rules",                          "project",      None,   "rule",            "rules"),
    ("learning roadmap finance",                         "learning",     None,   "rule",            "rules"),

    # ---- Negative / out-of-corpus ----
    ("Bill Ackman pershing square position",             None,           None,   None,              "negative"),
    ("Tesla deliveries Q4 2024",                         None,           None,   None,              "negative"),
    ("Federal Reserve rate cuts in October 2025",        None,           None,   None,              "negative"),
    ("Lithium mining in Argentina",                      None,           None,   None,              "negative"),
    ("Argentine peso devaluation strategy",              None,           None,   None,              "negative"),
    ("recipe for sourdough bread",                       None,           None,   None,              "negative"),

    # ---- Paraphrases (same fact, different wording) ----
    ("microsoft cloud infrastructure expansion plans",   "cloud",        "MSFT", None,              "paraphrase"),
    ("how do hyperscaler capex cycles end",              "capex",        None,   "analogue",        "paraphrase"),
    ("what happens when DRAM memory supply gluts the market", "dram",    None,   "analogue",        "paraphrase"),
    ("the great financial crisis housing bubble",        "housing",      None,   "analogue",        "paraphrase"),
    ("Azure year over year acceleration",                "azure",        "MSFT", None,              "paraphrase"),

    # ---- Cross-document (should pull from multiple docs) ----
    ("capex peak tech cycle",                            "capex",        None,   "analogue",        "cross_doc"),
    ("cloud growth and customer commitments",            "cloud",        "MSFT", None,              "cross_doc"),
    ("multiples expansion and contraction in tech",      None,           None,   "analogue",        "cross_doc"),
]


def _query_matches_truth(
    result: Dict[str, Any],
    expected_substr: Optional[str],
    ticker: Optional[str],
    doc_type: Optional[str],
) -> bool:
    """Decide whether a single retrieval row satisfies the ground-truth.

    For positive cases (expected_substr is not None):
      - chunk_text must contain expected_substr (case-insensitive)
      - if ticker given, result['ticker'] must match
      - if doc_type given, result['doc_type'] must match
    For negative cases this function is never called.
    """
    if expected_substr is None:
        return False
    txt = (result.get("chunk_text") or "").lower()
    if expected_substr.lower() not in txt:
        return False
    if ticker is not None and result.get("ticker") != ticker:
        return False
    if doc_type is not None and result.get("doc_type") != doc_type:
        return False
    return True


def section_2_ground_truth() -> Dict[str, Any]:
    _section("2. Ground-truth Q&A retrieval (P@1, P@5, MRR)")

    per_category: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "total": 0, "p1_hits": 0, "p5_hits": 0, "mrr_sum": 0.0,
        "neg_total": 0, "neg_pass": 0,
    })
    failures: List[Dict[str, Any]] = []
    all_top_scores: List[float] = []

    for q_idx, (query, exp_substr, ticker, doc_type, category) in enumerate(GROUND_TRUTH):
        # Apply ticker/doc_type as filters at retrieval time when set.
        # (Negatives never set filters.)
        res = rag_search(query, ticker=ticker, doc_type=doc_type, top_k=5)
        results = res.get("results", [])
        top_score = results[0]["similarity"] if results else 0.0
        all_top_scores.append(top_score)

        bucket = per_category[category]
        bucket["total"] += 1

        if exp_substr is None:
            # Negative case — top-1 score must be below 0.5.
            bucket["neg_total"] += 1
            passed = top_score < 0.5
            if passed:
                bucket["neg_pass"] += 1
            else:
                failures.append({
                    "category":   category,
                    "query":      query,
                    "issue":      "negative case had high top-1 score",
                    "top_score":  round(top_score, 3),
                    "top_doc_id": results[0]["doc_id"] if results else None,
                })
            continue

        # Positive case — find rank of first matching result in top-5.
        rank: Optional[int] = None
        for i, r in enumerate(results[:5]):
            if _query_matches_truth(r, exp_substr, ticker, doc_type):
                rank = i + 1
                break

        if rank == 1:
            bucket["p1_hits"] += 1
        if rank is not None:
            bucket["p5_hits"] += 1
            bucket["mrr_sum"] += 1.0 / rank
        else:
            failures.append({
                "category":  category,
                "query":     query,
                "issue":     "no top-5 hit",
                "top_doc_ids": [r.get("doc_id") for r in results[:5]],
                "top_score": round(top_score, 3),
                "expected_substr": exp_substr,
                "expected_ticker": ticker,
                "expected_doc_type": doc_type,
            })

    # Aggregate
    overall = {
        "n":           sum(b["total"]   for b in per_category.values()),
        "p1_hits":     sum(b["p1_hits"] for b in per_category.values()),
        "p5_hits":     sum(b["p5_hits"] for b in per_category.values()),
        "mrr_sum":     sum(b["mrr_sum"] for b in per_category.values()),
        "positive_n":  sum(b["total"] - b["neg_total"] for b in per_category.values()),
        "negative_n":  sum(b["neg_total"] for b in per_category.values()),
        "negative_pass": sum(b["neg_pass"] for b in per_category.values()),
    }
    positive_n = max(overall["positive_n"], 1)
    overall["p_at_1"] = overall["p1_hits"] / positive_n
    overall["p_at_5"] = overall["p5_hits"] / positive_n
    overall["mrr"]    = overall["mrr_sum"] / positive_n

    print(f"  ground-truth pairs: {overall['n']} "
          f"(positives={overall['positive_n']}, negatives={overall['negative_n']})")
    print(f"  P@1 = {overall['p_at_1']:.3f}  "
          f"P@5 = {overall['p_at_5']:.3f}  "
          f"MRR = {overall['mrr']:.3f}")
    print(f"  negative-case pass rate = "
          f"{overall['negative_pass']}/{overall['negative_n']}")

    # Per-category readout
    cat_summary: Dict[str, Dict[str, Any]] = {}
    for cat, b in per_category.items():
        pos = b["total"] - b["neg_total"]
        denom = max(pos, 1)
        cat_summary[cat] = {
            "total":   b["total"],
            "positive": pos,
            "p_at_1":  (b["p1_hits"] / denom) if pos else None,
            "p_at_5":  (b["p5_hits"] / denom) if pos else None,
            "mrr":     (b["mrr_sum"] / denom) if pos else None,
            "neg_total": b["neg_total"],
            "neg_pass":  b["neg_pass"],
        }
        if pos:
            print(f"    [{cat:11s}] n={pos:>2}  "
                  f"P@1={cat_summary[cat]['p_at_1']:.2f}  "
                  f"P@5={cat_summary[cat]['p_at_5']:.2f}  "
                  f"MRR={cat_summary[cat]['mrr']:.2f}")
        else:
            print(f"    [{cat:11s}] negatives n={b['neg_total']}  "
                  f"pass={b['neg_pass']}/{b['neg_total']}")

    # Soft checks fed into the gate evaluation
    _check(
        "P@5 >= 0.80 (quality gate)",
        overall["p_at_5"] >= 0.80,
        f"P@5={overall['p_at_5']:.3f}",
    )
    _check(
        "MRR >= 0.65 (quality gate)",
        overall["mrr"] >= 0.65,
        f"MRR={overall['mrr']:.3f}",
    )
    _check(
        "all negative-case top-1 scores < 0.5",
        overall["negative_pass"] == overall["negative_n"],
        f"{overall['negative_pass']}/{overall['negative_n']} passed",
    )

    return {
        "overall":     overall,
        "per_category": cat_summary,
        "failures":    failures,
    }


# ---------------------------------------------------------------------------
# Section 3 — Adversarial battery
# ---------------------------------------------------------------------------

# 5 facts x 3 paraphrases each. Each "fact" must map to a single chunk in
# the corpus (we accept the most common doc_id across the three queries).
PARAPHRASE_FACTS: List[Tuple[str, List[str]]] = [
    ("azure_cloud_growth", [
        "Azure cloud revenue growth",
        "Microsoft Azure year-over-year acceleration",
        "MSFT cloud business expansion rate",
    ]),
    ("dotcom_capex_peak", [
        "1999 dot-com capex peak",
        "telecom build-out at the internet bubble top",
        "Cisco Lucent Nortel peak multiples 2000",
    ]),
    ("dram_memory_bust", [
        "memory cycle DRAM glut",
        "Micron and SK Hynix oversupply",
        "hyperscaler DRAM inventory pull-forward",
    ]),
    ("ai_hyperscaler_capex", [
        "AI hyperscaler capex cycle",
        "ChatGPT triggered cloud capex acceleration",
        "Big 4 CSPs capex going from 130B to 300B",
    ]),
    ("housing_financial_crisis", [
        "2007 2008 housing financial crisis",
        "subprime mortgage collapse and bank failures",
        "Lehman collapse and great financial crisis",
    ]),
]


def section_3_adversarial() -> Dict[str, Any]:
    _section("3. Adversarial battery")

    out: Dict[str, Any] = {}

    # ---- 3a. Paraphrase robustness ----
    para_results: List[Dict[str, Any]] = []
    paraphrase_pass_count = 0
    for fact_id, queries in PARAPHRASE_FACTS:
        top_ids = []
        for q in queries:
            res = rag_search(q, top_k=5)
            top_doc_id = res["results"][0]["doc_id"] if res.get("results") else None
            top_ids.append(top_doc_id)
        # Find the most common doc_id; require it shows up in >= 2/3 queries
        counter = Counter([d for d in top_ids if d])
        most_common, freq = (counter.most_common(1)[0] if counter else (None, 0))
        passed = freq >= 2
        if passed:
            paraphrase_pass_count += 1
        para_results.append({
            "fact_id":     fact_id,
            "top_doc_ids": top_ids,
            "consensus":   most_common,
            "consensus_count": freq,
            "passed":      passed,
        })
        print(f"  paraphrase [{fact_id}] consensus={most_common} ({freq}/3) "
              f"{'PASS' if passed else 'FAIL'}")
    out["paraphrase"] = {
        "facts_total":   len(PARAPHRASE_FACTS),
        "facts_passed":  paraphrase_pass_count,
        "details":       para_results,
    }
    _check(
        "paraphrase robustness: >= 4/5 facts have a 2/3 consensus",
        paraphrase_pass_count >= 4,
        f"{paraphrase_pass_count}/{len(PARAPHRASE_FACTS)} passed",
    )

    # ---- 3b. Ambiguous queries ----
    ambig_results: List[Dict[str, Any]] = []
    for q in ["Azure", "capex", "cloud", "margin"]:
        res = rag_search(q, top_k=5)
        results = res.get("results", [])
        n_results = len(results)
        n_msft = sum(1 for r in results if r.get("ticker") == "MSFT")
        n_distinct_docs = len({r.get("doc_id") for r in results})
        top_score = results[0]["similarity"] if results else 0.0
        ambig_results.append({
            "query":         q,
            "n_results":     n_results,
            "n_msft":        n_msft,
            "n_distinct_docs": n_distinct_docs,
            "top_score":     round(top_score, 3),
        })
        print(f"  ambiguous '{q}': {n_results} hits, "
              f"{n_distinct_docs} distinct docs, top_score={top_score:.3f}")
    out["ambiguous"] = ambig_results
    _check(
        "ambiguous queries return non-empty results without crashing",
        all(r["n_results"] >= 1 for r in ambig_results),
        f"empties={[r['query'] for r in ambig_results if r['n_results'] == 0]}",
    )

    # ---- 3c. Out-of-corpus thorough negatives ----
    negatives = [
        "quantum computing roadmap of Rigetti",
        "Argentine peso devaluation strategy",
        "ancient Greek philosophy and metaphysics",
        "recipe for sourdough bread starter",
        "FIFA World Cup 2022 final score",
        "best hiking trails in New Zealand",
    ]
    neg_results: List[Dict[str, Any]] = []
    for q in negatives:
        res = rag_search(q, top_k=3)
        top = res["results"][0]["similarity"] if res.get("results") else 0.0
        neg_results.append({"query": q, "top_score": round(top, 3),
                            "passed": top < 0.5})
        print(f"  oop negative '{q[:40]:40s}' top_score={top:.3f}  "
              f"{'PASS' if top < 0.5 else 'FAIL'}")
    out["negatives"] = neg_results
    failed_negs = [n for n in neg_results if not n["passed"]]
    _check(
        "no high-confidence false positives on out-of-corpus queries",
        len(failed_negs) == 0,
        f"{len(failed_negs)} negatives scored >= 0.5",
    )

    # ---- 3d. Polarity flip ----
    # Ask the OPPOSITE of a known fact. The related chunk should still
    # surface (the topic is the same) but ideally not as top-1.
    polarity_pairs = [
        ("Azure margins are expanding rapidly", "MSFT"),
        ("hyperscaler capex is collapsing",     None),
    ]
    polarity_results: List[Dict[str, Any]] = []
    for q, ticker in polarity_pairs:
        res = rag_search(q, top_k=5)
        results = res.get("results", [])
        top_score = results[0]["similarity"] if results else 0.0
        # Determine if the topical chunk shows up anywhere in top-5
        # (use a simple keyword match).
        topical_kw = "azure" if "azure" in q.lower() else "capex"
        topical_rank = None
        for i, r in enumerate(results):
            if topical_kw in (r.get("chunk_text") or "").lower():
                topical_rank = i + 1
                break
        polarity_results.append({
            "query":         q,
            "top_score":     round(top_score, 3),
            "topical_rank":  topical_rank,
            "top_doc_id":    results[0]["doc_id"] if results else None,
        })
        print(f"  polarity '{q[:45]}'  topical_rank={topical_rank}  "
              f"top_score={top_score:.3f}")
    out["polarity"] = polarity_results
    # We just document polarity behavior — no hard gate. We do require
    # that the topical chunk shows up in top-5 (since the topic is the
    # same).
    _check(
        "polarity-flip query still surfaces topical chunk in top-5",
        all(p["topical_rank"] is not None for p in polarity_results),
        f"missing in: {[p['query'] for p in polarity_results if p['topical_rank'] is None]}",
    )

    # ---- 3e. Cross-document spread ----
    cross_queries = [
        "tech capex cycle",
        "valuation expansion and multiple compression",
        "cloud growth across hyperscalers",
    ]
    cross_results: List[Dict[str, Any]] = []
    for q in cross_queries:
        res = rag_search(q, top_k=5)
        doc_ids = [r["doc_id"] for r in res.get("results", [])]
        distinct = len(set(doc_ids))
        cross_results.append({
            "query":          q,
            "distinct_docs":  distinct,
            "top_doc_ids":    doc_ids,
            "passed":         distinct >= 2,
        })
        print(f"  cross-doc '{q}' -> {distinct} distinct docs in top-5")
    out["cross_doc"] = cross_results
    _check(
        "cross-document queries hit >= 2 distinct doc_ids in top-5",
        all(c["passed"] for c in cross_results),
        f"failing: {[c['query'] for c in cross_results if not c['passed']]}",
    )

    return out


# ---------------------------------------------------------------------------
# Section 4 — Volume + latency
# ---------------------------------------------------------------------------

def _make_synthetic_text(seed: int) -> str:
    """Deterministic synthetic chunk text — every chunk distinct.

    We mix some financial vocabulary so the embedding model doesn't
    collapse everything to the same neighborhood, but we keep the strings
    unique via the seed so we can later look one up by exact match.
    """
    rng = random.Random(seed)
    nouns = ["margin", "revenue", "capex", "fcf", "backlog", "guidance",
             "inventory", "buyback", "dividend", "leverage", "spread",
             "yield", "premium", "synergy", "tailwind", "headwind"]
    verbs = ["accelerates", "decelerates", "expands", "compresses",
             "stabilizes", "normalizes", "diverges", "converges"]
    sentences: List[str] = []
    for _ in range(5):
        n1 = rng.choice(nouns)
        v  = rng.choice(verbs)
        n2 = rng.choice(nouns)
        tag = "".join(rng.choices(string.ascii_lowercase + string.digits, k=10))
        sentences.append(f"Synthetic chunk {seed}-{tag}: {n1} {v} as {n2} resets.")
    return " ".join(sentences)


def section_4_volume_latency(target_chunks: int = 2000) -> Dict[str, Any]:
    _section("4. Volume + latency")
    out: Dict[str, Any] = {}

    # Generate `target_chunks` synthetic single-chunk docs. We use the
    # ingest_document path so the test exercises the same code an extractor
    # would. Each doc becomes 1 chunk (text is short) -> chunk count == doc count.
    print(f"  ingesting {target_chunks} synthetic chunks under prefix {_VOL_PREFIX} ...")
    n_inserted = 0
    t_start = time.perf_counter()
    BATCH = 100
    sample_lookup: List[Tuple[str, str]] = []  # (doc_id, text)
    for batch_start in range(0, target_chunks, BATCH):
        batch_n = min(BATCH, target_chunks - batch_start)
        for i in range(batch_n):
            seed = batch_start + i
            text = _make_synthetic_text(seed)
            doc_id = f"{_VOL_PREFIX}{seed:05d}"
            try:
                res = ingest_document(
                    text,
                    {"source_tool": "test_rag_heavy_stress",
                     "doc_type":    "stress_synthetic",
                     "ticker":      None},
                    doc_id=doc_id,
                    target_tokens=500, overlap_tokens=0,
                )
                n_inserted += res.get("chunks_inserted", 0)
                # Save a few for the "find exact" test below.
                if seed in (10, 200, 500, 1000, 1500):
                    sample_lookup.append((doc_id, text))
            except Exception as exc:
                print(f"  ingest failed for {doc_id}: {exc}")
        if (batch_start + batch_n) % 500 == 0:
            print(f"    ... ingested {batch_start + batch_n}/{target_chunks}")
    t_ingest = time.perf_counter() - t_start

    throughput = n_inserted / max(t_ingest, 1e-6)
    out["chunks_target"]    = target_chunks
    out["chunks_inserted"]  = n_inserted
    out["ingest_seconds"]   = round(t_ingest, 2)
    out["throughput_cps"]   = round(throughput, 1)
    print(f"  ingestion: {n_inserted} chunks in {t_ingest:.2f}s "
          f"-> {throughput:.1f} chunks/sec")
    _check(
        f"ingested {target_chunks} synthetic chunks",
        n_inserted >= target_chunks * 0.99,
        f"only {n_inserted}/{target_chunks}",
    )

    # Now run 20 distinct queries to measure latency. Mix real corpus
    # queries with a couple of synthetic-text queries so the planner
    # touches both clusters of the vector space.
    print("  measuring query latency over 20 mixed queries ...")
    queries = [
        "Azure cloud growth",                   "DRAM memory bust",
        "1999 dot-com bubble",                  "hyperscaler capex cycle",
        "subprime housing crisis",              "Microsoft cloud commercial",
        "MSFT forward guidance",                "Bull bear case analyst writeup",
        "smartphone mobile cycle",              "energy capex drought",
        "China infrastructure stimulus",        "SPAC reflexivity bust",
        "AI capex acceleration",                "valuation compression tech",
        "oil shale collapse",                   "COVID cloud demand pull-forward",
        "Federal Reserve guidance",             "remaining performance obligation",
        # Two synthetic-leaning queries — short and likely to fall into the noise cluster.
        "synthetic chunk margin expands as revenue resets",
        "stress test synthetic accelerates capex",
    ]
    # Warm-up.
    rag_search("warmup", top_k=3)
    latencies_ms: List[float] = []
    for q in queries:
        t0 = time.perf_counter()
        rag_search(q, top_k=5)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    latencies_ms.sort()
    p50 = latencies_ms[len(latencies_ms) // 2]
    p95 = latencies_ms[max(0, int(len(latencies_ms) * 0.95) - 1)]
    p99 = latencies_ms[max(0, int(len(latencies_ms) * 0.99) - 1)]
    out["latency_p50_ms"] = round(p50, 1)
    out["latency_p95_ms"] = round(p95, 1)
    out["latency_p99_ms"] = round(p99, 1)
    out["latency_n"]      = len(latencies_ms)
    print(f"  latency: p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms  "
          f"(n={len(latencies_ms)})")
    _check(
        "p95 latency < 1000ms",
        p95 < 1000.0,
        f"p95={p95:.1f}ms",
    )

    # Vector-index correctness: query for the exact text of a few sample
    # chunks. Top-1 should be the original doc.
    print("  verifying vector index round-trip for 5 sample chunks ...")
    exact_hits = 0
    for doc_id, text in sample_lookup:
        # Use a long substring of the original text so the query is
        # close to the embedded chunk in vector space.
        q = text[:200]
        res = rag_search(q, top_k=1)
        results = res.get("results", [])
        if results and results[0].get("doc_id") == doc_id:
            exact_hits += 1
        else:
            top_id = results[0]["doc_id"] if results else None
            print(f"  exact-text round-trip failed for {doc_id} (top={top_id})")
    out["exact_round_trip_pass"] = exact_hits
    out["exact_round_trip_total"] = len(sample_lookup)
    _check(
        "vector-index round-trip: 5/5 exact-text queries return the same chunk",
        exact_hits == len(sample_lookup),
        f"{exact_hits}/{len(sample_lookup)} round-trips matched",
    )

    return out


# ---------------------------------------------------------------------------
# Section 5 — Report
# ---------------------------------------------------------------------------

def _render_report(report_path: str) -> None:
    """Write the human-readable engineering report to disk."""
    r = _REPORT
    lines: List[str] = []
    lines.append("# RAG Heavy Stress Test Report")
    lines.append("")
    lines.append(f"- **Started:**   {r['started_at']}")
    lines.append(f"- **Finished:**  {r['finished_at']}")
    lines.append(f"- **Test file:** `testing/test_rag_heavy_stress.py`")
    lines.append(f"- **Harness:**   {_results['pass']} PASS / {_results['fail']} FAIL")
    lines.append("")

    # Quality gates header — surfaced first because that's the headline.
    lines.append("## Quality Gates")
    lines.append("")
    lines.append("| Gate | Threshold | Observed | Status |")
    lines.append("|------|-----------|----------|--------|")
    for gate, info in r["gates"].items():
        status = "PASS" if info["passed"] else "FAIL"
        lines.append(f"| {gate} | {info['threshold']} | {info['observed']} | **{status}** |")
    lines.append("")

    # Section 1
    bs = r.get("bootstrap") or {}
    lines.append("## Section 1 — Bootstrap + corpus expansion")
    lines.append("")
    lines.append(f"- bootstrap script returncode: `{bs.get('bootstrap_returncode')}`")
    lines.append(f"- bootstrap chunk count (reported by script): `{bs.get('bootstrap_reported_chunks')}`")
    lines.append(f"- bootstrap duration: `{bs.get('bootstrap_seconds')}s`")
    lines.append(f"- chunks after live ingestion: `{bs.get('chunks_after_live_ingest')}`")
    lines.append("")
    lines.append("### doc_type breakdown")
    lines.append("")
    lines.append("| doc_type | count |")
    lines.append("|----------|-------|")
    for t, n in sorted((bs.get("doc_type_breakdown") or {}).items()):
        lines.append(f"| {t} | {n} |")
    lines.append("")
    lines.append("### ticker breakdown")
    lines.append("")
    lines.append("| ticker | count |")
    lines.append("|--------|-------|")
    for t, n in sorted((bs.get("ticker_breakdown") or {}).items()):
        lines.append(f"| {t} | {n} |")
    lines.append("")
    lines.append("### Live extractor ingestion attempts")
    lines.append("")
    lines.append("| tool | success | chunks | error |")
    lines.append("|------|---------|--------|-------|")
    for entry in (bs.get("live_ingests") or []):
        lines.append(
            f"| {entry.get('tool')} | "
            f"{entry.get('extract_success')} | "
            f"{entry.get('total_chunks_inserted', 0)} | "
            f"{(entry.get('error') or '').replace(chr(10), ' ')[:80]} |"
        )
    lines.append("")

    # Section 2
    qa = r.get("qa_metrics") or {}
    overall = qa.get("overall") or {}
    lines.append("## Section 2 — Ground-truth Q&A retrieval")
    lines.append("")
    lines.append(f"- ground-truth pairs: **{overall.get('n')}** "
                 f"(positives={overall.get('positive_n')}, "
                 f"negatives={overall.get('negative_n')})")
    lines.append(f"- **P@1** = `{overall.get('p_at_1', 0):.3f}`")
    lines.append(f"- **P@5** = `{overall.get('p_at_5', 0):.3f}`")
    lines.append(f"- **MRR** = `{overall.get('mrr', 0):.3f}`")
    lines.append(f"- negative-case pass rate = "
                 f"`{overall.get('negative_pass')}/{overall.get('negative_n')}`")
    lines.append("")
    lines.append("### Per-category breakdown")
    lines.append("")
    lines.append("| category | positives | P@1 | P@5 | MRR | negatives passed |")
    lines.append("|----------|-----------|-----|-----|-----|------------------|")
    for cat, info in sorted((qa.get("per_category") or {}).items()):
        def fmt(v):
            return f"{v:.2f}" if isinstance(v, float) else "-"
        lines.append(
            f"| {cat} | {info['positive']} | {fmt(info['p_at_1'])} | "
            f"{fmt(info['p_at_5'])} | {fmt(info['mrr'])} | "
            f"{info['neg_pass']}/{info['neg_total']} |"
        )
    lines.append("")

    # Section 3
    adv = r.get("adversarial") or {}
    lines.append("## Section 3 — Adversarial battery")
    lines.append("")
    para = adv.get("paraphrase") or {}
    lines.append(f"- **Paraphrase robustness:** "
                 f"{para.get('facts_passed', 0)}/{para.get('facts_total', 0)} "
                 f"facts retrieved the same doc in >= 2/3 paraphrases")
    lines.append("")
    lines.append("| fact_id | consensus doc | consensus / 3 | passed |")
    lines.append("|---------|---------------|---------------|--------|")
    for d in (para.get("details") or []):
        lines.append(f"| {d['fact_id']} | `{d['consensus']}` | "
                     f"{d['consensus_count']} | {d['passed']} |")
    lines.append("")
    lines.append("### Ambiguous queries")
    lines.append("")
    lines.append("| query | n_results | distinct_docs | top_score |")
    lines.append("|-------|-----------|---------------|-----------|")
    for a in (adv.get("ambiguous") or []):
        lines.append(f"| `{a['query']}` | {a['n_results']} | "
                     f"{a['n_distinct_docs']} | {a['top_score']} |")
    lines.append("")
    lines.append("### Out-of-corpus negatives")
    lines.append("")
    lines.append("| query | top_score | passed (<0.5) |")
    lines.append("|-------|-----------|---------------|")
    for n in (adv.get("negatives") or []):
        lines.append(f"| `{n['query']}` | {n['top_score']} | {n['passed']} |")
    lines.append("")
    lines.append("### Polarity flip")
    lines.append("")
    lines.append("Documented behavior: when we ask the opposite of a known "
                 "fact, the related chunk should still surface in top-5 because "
                 "the topic embeddings are similar. Top-1 score is reported "
                 "for transparency.")
    lines.append("")
    lines.append("| query | top_doc_id | topical_rank | top_score |")
    lines.append("|-------|------------|--------------|-----------|")
    for p in (adv.get("polarity") or []):
        lines.append(f"| `{p['query']}` | `{p['top_doc_id']}` | "
                     f"{p['topical_rank']} | {p['top_score']} |")
    lines.append("")
    lines.append("### Cross-document spread")
    lines.append("")
    lines.append("| query | distinct_docs in top-5 | passed (>=2) |")
    lines.append("|-------|------------------------|--------------|")
    for c in (adv.get("cross_doc") or []):
        lines.append(f"| `{c['query']}` | {c['distinct_docs']} | {c['passed']} |")
    lines.append("")

    # Section 4
    vol = r.get("volume") or {}
    lines.append("## Section 4 — Volume + latency")
    lines.append("")
    lines.append(f"- synthetic chunks ingested: `{vol.get('chunks_inserted')}` "
                 f"/ target `{vol.get('chunks_target')}`")
    lines.append(f"- ingestion duration: `{vol.get('ingest_seconds')}s`")
    lines.append(f"- ingestion throughput: `{vol.get('throughput_cps')}` chunks/sec")
    lines.append(f"- query latency p50: `{vol.get('latency_p50_ms')}ms`")
    lines.append(f"- query latency p95: `{vol.get('latency_p95_ms')}ms`")
    lines.append(f"- query latency p99: `{vol.get('latency_p99_ms')}ms`")
    lines.append(f"- exact-text round-trip: "
                 f"`{vol.get('exact_round_trip_pass')}/"
                 f"{vol.get('exact_round_trip_total')}`")
    lines.append("")

    # Top failures
    lines.append("## Top failures (worst 5 queries)")
    lines.append("")
    if not r.get("top_failures"):
        lines.append("_No failures recorded._")
    else:
        lines.append("| category | query | issue | diagnosis |")
        lines.append("|----------|-------|-------|-----------|")
        for f in r["top_failures"]:
            diag = f.get("diagnosis", "")
            lines.append(f"| {f.get('category')} | `{f.get('query')}` | "
                         f"{f.get('issue')} | {diag} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_Report generated by `testing/test_rag_heavy_stress.py`. "
                 "Failures above represent real retrieval misses on the current "
                 "corpus — investigate before shipping if any quality gate is FAIL._")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _evaluate_gates() -> None:
    """Compute pass/fail for each gate and populate _REPORT['gates']."""
    qa = _REPORT.get("qa_metrics", {}).get("overall", {})
    vol = _REPORT.get("volume", {})
    p_at_5 = qa.get("p_at_5", 0.0)
    mrr    = qa.get("mrr", 0.0)
    neg_pass  = qa.get("negative_pass", 0)
    neg_total = qa.get("negative_n", 0)
    p95 = vol.get("latency_p95_ms", 9999.0)

    _REPORT["gates"] = {
        "P@5 >= 0.80": {
            "threshold": "0.80",
            "observed":  f"{p_at_5:.3f}",
            "passed":    p_at_5 >= 0.80,
        },
        "MRR >= 0.65": {
            "threshold": "0.65",
            "observed":  f"{mrr:.3f}",
            "passed":    mrr >= 0.65,
        },
        "No high-confidence false positives": {
            "threshold": "all neg top-1 < 0.5",
            "observed":  f"{neg_pass}/{neg_total} passed",
            "passed":    (neg_total > 0 and neg_pass == neg_total),
        },
        "p95 query latency < 1000ms": {
            "threshold": "1000ms",
            "observed":  f"{p95}ms",
            "passed":    p95 < 1000.0,
        },
    }


def _select_top_failures() -> None:
    """Pick the 5 worst Q&A failures with rough diagnoses."""
    qa = _REPORT.get("qa_metrics", {}) or {}
    fails = list(qa.get("failures", []))
    if not fails:
        _REPORT["top_failures"] = []
        return
    # Order: prefer no-top-5-hit first (more diagnostic), then negatives
    # that scored too high.
    def sort_key(f):
        if f.get("issue") == "no top-5 hit":
            return (0, -float(f.get("top_score", 0.0)))
        return (1, -float(f.get("top_score", 0.0)))
    fails.sort(key=sort_key)

    top = []
    for f in fails[:5]:
        diag = ""
        if f.get("issue") == "no top-5 hit":
            # Common diagnostic patterns
            if f.get("expected_doc_type") in ("rule", "analogue") and f.get("expected_ticker") is None:
                diag = ("topic embedding likely conflated with adjacent doc_type; "
                        "consider tightening filter or augmenting chunks")
            else:
                diag = ("expected substring not found in top-5; "
                        "check chunker token boundaries and chunk_text content")
        elif f.get("issue") == "negative case had high top-1 score":
            diag = ("query overlaps semantically with a real corpus chunk; "
                    "consider raising negative threshold or rephrasing query")
        top.append({**f, "diagnosis": diag})
    _REPORT["top_failures"] = top


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("\nRAG HEAVY stress test\n")
    init_schema()

    _REPORT["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Belt-and-suspenders cleanup from any prior run, BEFORE we measure
    # corpus stats. (The bootstrap stays untouched.) We also sweep
    # SEARCH_TEST_ which is owned by test_rag_search.py — if a previous
    # run crashed mid-test those rows can pollute our retrieval.
    n_pre = _cleanup_prefix(_VOL_PREFIX)
    n_pre += _cleanup_prefix(_STRESS_PREFIX)
    n_pre += _cleanup_prefix("SEARCH_TEST_")
    if n_pre:
        print(f"  pre-clean removed {n_pre} stale test docs")

    try:
        # Section 1
        try:
            _REPORT["bootstrap"] = section_1_bootstrap()
        except Exception:
            traceback.print_exc()
            _REPORT["bootstrap"] = {"fatal_error": traceback.format_exc()}

        # Section 2
        try:
            _REPORT["qa_metrics"] = section_2_ground_truth()
        except Exception:
            traceback.print_exc()
            _REPORT["qa_metrics"] = {"fatal_error": traceback.format_exc(),
                                      "overall": {}}

        # Section 3
        try:
            _REPORT["adversarial"] = section_3_adversarial()
        except Exception:
            traceback.print_exc()
            _REPORT["adversarial"] = {"fatal_error": traceback.format_exc()}

        # Section 4
        try:
            _REPORT["volume"] = section_4_volume_latency(target_chunks=2000)
        except Exception:
            traceback.print_exc()
            _REPORT["volume"] = {"fatal_error": traceback.format_exc()}

        # Compute gates + top failures
        _evaluate_gates()
        _select_top_failures()

    finally:
        # ALWAYS clean up stress docs and ALWAYS write the report.
        n_post = _cleanup_prefix(_VOL_PREFIX)
        n_post += _cleanup_prefix(_STRESS_PREFIX)
        print(f"\n  post-clean removed {n_post} test docs")
        _REPORT["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        report_path = os.path.join(
            PROJECT_ROOT, "testing", "fixtures", "rag_stress_report.md"
        )
        try:
            _render_report(report_path)
            print(f"  wrote report -> {report_path}")
        except Exception:
            print("  FAILED to write report:")
            traceback.print_exc()

    # Print summary at the end
    print(f"\n=== Summary ===")
    print(f"  PASS: {_results['pass']}")
    print(f"  FAIL: {_results['fail']}")
    if _results["failures"]:
        print("\nFailing harness checks:")
        for name, hint in _results["failures"]:
            print(f"  - {name}: {hint}")
    # Quality gates as a final headline
    print("\nQuality gates:")
    for gate, info in _REPORT.get("gates", {}).items():
        status = "PASS" if info["passed"] else "FAIL"
        print(f"  [{status}] {gate}  (observed={info['observed']})")

    return 0 if _results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
