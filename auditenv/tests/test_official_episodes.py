"""Блок 2 определений жюри: условия зачёта эпизодов D01-D20 (независимо от matcher команды)."""
import json

import pytest

from qa.evidence import validate_evidence_dir
from qa.generated import validate_generated_dir
from tests.official import ALLOC, CUT, DUP, FX, MATRIX, S1, S2, build_all

IDS = [m[0] for m in MATRIX]
EXPECT = {  # (число ошибок каждого типа, сценарий)
    **{f"D0{i}": {} for i in range(1, 6)},
    "D06": {DUP: 1}, "D07": {DUP: 2}, "D08": {DUP: 1},
    "D09": {ALLOC: 1}, "D10": {ALLOC: 1}, "D11": {ALLOC: 1},
    "D12": {FX: 1}, "D13": {FX: 2}, "D14": {CUT: 1}, "D15": {CUT: 2},
    **{f"D{n}": {ALLOC: 1, FX: 1} for n in (16, 17, 18)},
    **{f"D{n}": {DUP: 1, FX: 1} for n in (19, 20)},
}


@pytest.fixture(scope="module")
def eps(tmp_path_factory):
    return build_all(tmp_path_factory.mktemp("D"))


def load(d, name):
    return json.loads((d / name).read_text(encoding="utf-8"))


def test_twenty_episodes_with_the_announced_matrix():
    assert len(MATRIX) == 20 and len(set(m[1] for m in MATRIX)) == 20


@pytest.mark.parametrize("eid", IDS)
def test_condition_1_5_books_consistent_and_formats(eps, eid):
    # книги согласованы до внесения ошибки, форматы §03 (суммы целые, время со смещением, null явные)
    assert validate_generated_dir(eps[eid]) == []


@pytest.mark.parametrize("eid", IDS)
def test_condition_2_3_4_injection_is_real_and_evidence_available(eps, eid):
    # независимые правила аудита (из policy.json) находят ровно заявленное; доказательства достижимы
    assert validate_evidence_dir(eps[eid]) == []


@pytest.mark.parametrize("eid", IDS)
def test_truth_matches_matrix(eps, eid):
    issues = load(eps[eid], "truth.json")["issues"]
    counts = {}
    for i in issues:
        counts[i["type"]] = counts.get(i["type"], 0) + 1
    assert counts == EXPECT[eid]
    if eid <= "D05":
        assert issues == []   # чистый эпизод: эталонных ошибок нет
    for i in issues:
        assert i["acceptable_evidence_sets"] and isinstance(i["expected_amount_minor"], int)
        assert i["currency"] and i["affected_ids"]


@pytest.mark.parametrize("eid", IDS)
def test_scenarios_match_matrix(eps, eid):
    scen = {i["scenario"] for i in load(eps[eid], "truth.json")["issues"] if i["scenario"]}
    want = {S1} if eid in ("D16", "D17", "D18") else {S2} if eid in ("D19", "D20") else set()
    assert scen == want


def test_allocation_has_missing_and_partial(eps):
    variants = {x.get("variant") for e in ("D09", "D10", "D11")
                for x in load(eps[e], "provenance.json")["injections"] if x["type"] == ALLOC}
    assert {"missing", "partial"} <= variants


def test_provenance_documents_before_state_of_each_injection(eps):
    for eid in IDS[5:]:
        prov = load(eps[eid], "provenance.json")
        assert prov["injections"] and all("operation" in x and "issue_id" in x for x in prov["injections"])


def test_repeated_scenario_episodes_differ(eps):
    for group in (("D16", "D17", "D18"), ("D19", "D20")):
        sigs = {json.dumps(load(eps[e], "truth.json"), sort_keys=True) for e in group}
        assert len(sigs) == len(group)


def test_no_unsolvable_data_oracle_agent_solves_every_episode(eps):
    from env.harness import HostEpisode, run_policy_full
    from evaluator import TruthGrader
    from policies.careful import CarefulPolicy
    for eid in IDS:
        g = TruthGrader(eps[eid])
        res = run_policy_full(HostEpisode(eps[eid], g), CarefulPolicy(), seed=1)
        assert g.last.metrics.f1 == 1.0, eid   # доказательств хватает, чтобы найти всё и не выдумать лишнего
