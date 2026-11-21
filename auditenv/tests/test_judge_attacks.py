"""Атаки на судью: попытки получить награду без настоящей работы. Все должны провалиться."""
import copy

import pytest

from evaluator import TruthGrader
from policies.oracle import findings_from_truth, load_correct_findings
import json


@pytest.fixture(scope="module")
def ep(gen):
    return gen["hard"]


@pytest.fixture
def grader(ep):
    return TruthGrader(ep)


@pytest.fixture
def correct(ep):
    return copy.deepcopy(load_correct_findings(ep)["findings"])


def test_correct_answer_is_max(grader, correct):
    assert grader.grade(correct, 0) == 1.0


def test_empty_answer_on_faulty_company_is_minimum(grader):
    assert grader.grade([], 0) == -1.0


def test_empty_answer_on_clean_company_is_max(gen):
    assert TruthGrader(gen["clean"]).grade([], 0) == 1.0


def test_spam_on_clean_company_is_minimum(gen):
    spam = [{"type": "duplicate_invoice", "affected_ids": [f"INV-00000{i}", f"DOC-INV-00000{i}"],
             "evidence_ids": [], "claimed_amount_minor": 1, "currency": "USD", "explanation": "x"} for i in range(1, 6)]
    assert TruthGrader(gen["clean"]).grade(spam, 0) == -1.0


def test_extra_findings_lower_the_reward(grader, correct):
    noisy = correct + [{**correct[0], "affected_ids": ["INV-000001", "DOC-INV-000001"]}]
    assert grader.grade(noisy, 0) < grader.grade(correct, 0)


def test_repeating_a_correct_finding_does_not_help(grader, correct):
    assert grader.grade(correct + [copy.deepcopy(correct[0])] * 5, 0) < 1.0


def test_one_shotgun_finding_with_all_ids_fails(grader, correct):
    ids = sorted({x for f in correct for x in f["affected_ids"]})[:10]
    shot = [{**correct[0], "affected_ids": ids}]
    assert grader.grade(shot, 0) == -1.0


@pytest.mark.parametrize("field,value", [
    ("claimed_amount_minor", lambda v: v + 1), ("claimed_amount_minor", lambda v: v * 2), ("currency", lambda v: "XXX"),
    ("type", lambda v: "period_cutoff" if v != "period_cutoff" else "duplicate_invoice"),
])
def test_wrong_field_loses_that_finding(grader, correct, field, value):
    bad = copy.deepcopy(correct)
    bad[0][field] = value(bad[0][field])
    assert grader.grade(bad, 0) < grader.grade(correct, 0)


def test_missing_evidence_is_not_a_hit(grader, correct):
    bad = copy.deepcopy(correct)
    bad[0]["evidence_ids"] = []
    assert grader.grade(bad, 0) < grader.grade(correct, 0)


def test_partial_affected_ids_is_not_a_hit(grader, correct):
    bad = copy.deepcopy(correct)
    bad[0]["affected_ids"] = [bad[0]["affected_ids"][0]]
    assert grader.grade(bad, 0) < grader.grade(correct, 0)


def test_extra_foreign_id_is_not_a_hit(grader, correct):
    bad = copy.deepcopy(correct)
    bad[0]["affected_ids"] = sorted(set(bad[0]["affected_ids"]) | {"INV-000001"})
    assert grader.grade(bad, 0) < grader.grade(correct, 0)


def test_ids_and_type_are_compared_exactly(grader, correct):
    messy = copy.deepcopy(correct)
    messy[0]["affected_ids"] = [f" {x.lower()} " for x in messy[0]["affected_ids"]]
    assert grader.grade(messy, 0) < 1.0
    messy = copy.deepcopy(correct)
    messy[0]["type"] = " " + messy[0]["type"].upper() + " "
    assert grader.grade(messy, 0) < 1.0


@pytest.mark.parametrize("bad", [
    None, "all", 5, {"findings": "x"}, {"x": []}, [None], [[]], ["text"], [{}],
    [{"type": "t", "affected_ids": "INV", "amount_minor": 1, "currency": "USD", "explanation": ""}],
    [{"type": "t", "affected_ids": ["a"], "amount_minor": "100", "currency": "USD", "explanation": ""}],
    [{"type": "t", "affected_ids": ["a"], "amount_minor": 1.5, "currency": "USD", "explanation": ""}],
    [{"type": "t", "affected_ids": ["a"], "amount_minor": -1, "currency": "USD", "explanation": ""}],
    [{"type": "t", "affected_ids": ["a"], "amount_minor": True, "currency": "USD", "explanation": ""}],
    [{"type": "t", "affected_ids": ["a"] * 11, "amount_minor": 1, "currency": "USD", "explanation": ""}],
    [{"type": "t", "affected_ids": ["a"], "amount_minor": 1, "currency": "USD", "explanation": "x" * 1001}],
    [{"type": "t", "affected_ids": ["a"], "amount_minor": 1, "currency": "USD", "explanation": "", "extra": 1}],
])
def test_structurally_invalid_answers_get_minimum(grader, bad):
    assert grader.grade(bad, 0) == -1.0


def test_more_than_30_findings_is_invalid(grader, correct):
    assert grader.grade([correct[0]] * 31, 0) == -1.0


def test_inspection_penalty_is_exact_and_clipped(grader, correct):
    assert grader.grade(correct, 50) == pytest.approx(0.9)
    assert grader.grade(correct, 100) == pytest.approx(0.8)
    assert grader.grade([], 100) == -1.0  # не уходит ниже -1


def test_reward_improves_with_each_found_error(grader, correct):
    rewards = [grader.grade(correct[:k], 0) for k in range(len(correct) + 1)]
    assert rewards == sorted(rewards) and rewards[0] == -1.0 and rewards[-1] == 1.0


def test_no_free_reward_for_reading_nothing_on_faulty_company(grader, correct):
    # молчание на ошибочной компании хуже любого верного ответа
    assert grader.grade([], 0) < grader.grade(correct[:1], 50)
