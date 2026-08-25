"""One person is one member, and several people are never one total.

Both faults are real and were found in the live store.

**A surname is not a person.** `member_holdings("Scott")` matched seven
people -- Rick Scott, Tim Scott, Austin Scott, and four House members whose
FIRST name is Scott -- and summed all 365 holdings into a single
$250m-$730m block presented as though it belonged to one filer. The matched
list disclosed the names, but the number beside it was a merge.

**The same person is written several ways.** The House index gives
"Rudy C. Yakym" on one filing and "Rudy Yakym" on another, "Thomas Suozzi"
and "Thomas R. Suozzi", "Laurel Lee" and "Laurel Mrs Lee". Keyed literally,
each variant became its own member and split that person's history in two.
"""
import pytest

from tools.altdata_server import congress_store as store
from tools.altdata_server import congress_queries as q


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_CONGRESS_DB", str(tmp_path / "c.db"))
    store.init_schema()


# ------------------------------------------------------------ member identity

@pytest.mark.parametrize("a,b", [
    (("Yakym", "Rudy C."), ("Yakym", "Rudy")),
    (("Suozzi", "Thomas"), ("Suozzi", "Thomas R.")),
    (("Lee", "Laurel"), ("Lee", "Laurel Mrs")),
    (("Aderholt", "Robert B."), ("Aderholt", "Robert")),
])
def test_one_person_written_two_ways_is_one_member(a, b):
    """A middle initial or an honorific is not a different human being."""
    assert store.member_id("house", a[0], a[1], "IN") == \
        store.member_id("house", b[0], b[1], "IN")


@pytest.mark.parametrize("a,b", [
    (("Scott", "Rick"), ("Scott", "Tim")),
    (("Smith", "Adam"), ("Smith", "Adrian")),
    (("Lee", "Susie"), ("Lee", "Michael")),
])
def test_two_people_sharing_a_surname_stay_apart(a, b):
    assert store.member_id("senate", a[0], a[1]) != \
        store.member_id("senate", b[0], b[1])


def test_the_chamber_still_separates_otherwise_identical_names():
    assert store.member_id("house", "Scott", "Rick") != \
        store.member_id("senate", "Scott", "Rick")


# --------------------------------------------------------------- disambiguation

def _seed_two_scotts():
    rick = store.member_id("senate", "Scott", "Rick", "FL")
    tim = store.member_id("senate", "Scott", "Tim", "SC")
    austin = store.member_id("house", "Scott", "Austin", "GA")
    desjarlais = store.member_id("house", "DesJarlais", "Scott", "TN")
    people = [(rick, "senate", "Scott", "Rick", "Rick Scott"),
              (tim, "senate", "Scott", "Tim", "Tim Scott"),
              (austin, "house", "Scott", "Austin", "Austin Scott"),
              (desjarlais, "house", "DesJarlais", "Scott", "Scott DesJarlais")]
    for mid, chamber, last, first, full in people:
        store.upsert_member({"member_id": mid, "chamber": chamber, "last": last,
                             "first": first, "full_name": full, "state": "XX",
                             "district": None, "office": None,
                             "first_seen": "2026-01-01", "last_seen": "2026-01-01"})
        store.upsert_filing({"filing_id": f"f:{mid}", "chamber": chamber,
                             "doc_id": mid, "member_id": mid,
                             "filing_type": "annual", "filed_date": "2026-05-15",
                             "year": 2025, "parse_status": "parsed"})
        store.replace_holdings(f"f:{mid}", mid, [
            {"ticker": "AAPL", "asset_name": "Apple Inc", "owner": "self",
             "value_min": 1000, "value_max": 5000, "as_of": "2025-12-31"}])
        store.replace_transactions(f"f:{mid}", mid, [
            {"ticker": "AAPL", "asset_name": "Apple Inc", "owner": "self",
             "transaction_type": "purchase", "transaction_date": "2025-06-01",
             "amount_min": 1001, "amount_max": 15000}])
    return {"rick": rick, "tim": tim, "austin": austin, "desjarlais": desjarlais}


def test_a_surname_does_not_match_someones_first_name(db):
    """Scott DesJarlais is not a Scott for this purpose."""
    _seed_two_scotts()
    names = {m["full_name"] for m in q.member_holdings("Scott")["matched_members"]}

    assert "Scott DesJarlais" not in names, (
        "matched a member whose FIRST name is Scott; a surname query that "
        "sweeps in first names attributes holdings to the wrong people")
    assert {"Rick Scott", "Tim Scott", "Austin Scott"} <= names


def test_totals_are_broken_out_when_several_people_match(db):
    """One number covering several filers is not a fact about any of them."""
    _seed_two_scotts()
    result = q.member_holdings("Scott")

    assert result["ambiguous"] is True
    per = {p["member"]: p for p in result["per_member"]}
    assert set(per) >= {"Rick Scott", "Tim Scott", "Austin Scott"}
    for person in per.values():
        assert person["totals"]["value_min_total"] == 1000
    assert "more than one member" in result["note"]


def test_a_single_match_is_not_flagged_ambiguous(db):
    _seed_two_scotts()
    result = q.member_holdings("Rick Scott")

    assert result["ambiguous"] is False
    assert [m["full_name"] for m in result["matched_members"]] == ["Rick Scott"]
    assert result["holding_count"] == 1


def test_trades_are_broken_out_the_same_way(db):
    _seed_two_scotts()
    result = q.member_activity("Scott")

    assert result["ambiguous"] is True
    assert len(result["per_member"]) >= 3
    for person in result["per_member"]:
        assert person["totals"]["amount_min_total"] == 1001


# ------------------------------------------------------------------ migration

def _split_person(db_unused=None):
    """The same person under two keys, as the live store held them."""
    old = "house:yakym:rudy_c.:IN"          # written before the key normalised
    new = store.member_id("house", "Yakym", "Rudy", "IN")
    assert old != new
    for mid, first, full in ((old, "Rudy C.", "Rudy C. Yakym"),
                             (new, "Rudy", "Rudy Yakym")):
        store.upsert_member({"member_id": mid, "chamber": "house", "last": "Yakym",
                             "first": first, "full_name": full, "state": "IN",
                             "district": "IN02", "office": None,
                             "first_seen": "2025-01-01", "last_seen": "2026-01-01"})
        store.upsert_filing({"filing_id": f"f:{mid}", "chamber": "house",
                             "doc_id": mid, "member_id": mid, "filing_type": "ptr",
                             "filed_date": "2026-01-01", "year": 2026,
                             "parse_status": "parsed"})
        store.replace_transactions(f"f:{mid}", mid, [
            {"ticker": "AAPL", "asset_name": "Apple", "owner": "self",
             "transaction_type": "purchase", "transaction_date": "2025-06-01",
             "amount_min": 1001, "amount_max": 15000}])
    return old, new


def test_a_split_person_is_merged_under_one_key(db):
    old, new = _split_person()
    merged = store.merge_duplicate_members()

    assert merged == 1
    with store.connect() as conn:
        ids = [r[0] for r in conn.execute("SELECT member_id FROM members")]
    assert ids == [new], f"left {ids}"


def test_merging_repoints_every_row_rather_than_dropping_them(db):
    """The whole point is to reunite a fragmented history, not halve it."""
    old, new = _split_person()
    store.merge_duplicate_members()

    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 2
        orphaned = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE member_id = ?", (old,)).fetchone()[0]
        assert orphaned == 0, "rows still point at the retired key"
        assert conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE member_id = ?", (new,)).fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM filings WHERE member_id = ?", (new,)).fetchone()[0] == 2


def test_the_query_sees_one_person_after_merging(db):
    _split_person()
    store.merge_duplicate_members()
    result = q.member_activity("Yakym")

    assert result["ambiguous"] is False
    assert result["transaction_count"] == 2, (
        "the person's history is still split across two member records")


def test_merging_leaves_distinct_people_alone(db):
    _seed_two_scotts()
    before = store.merge_duplicate_members()

    assert before == 0
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM members").fetchone()[0] == 4


def test_merging_is_idempotent(db):
    _split_person()
    assert store.merge_duplicate_members() == 1
    assert store.merge_duplicate_members() == 0
