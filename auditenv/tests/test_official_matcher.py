"""Блок 3 определений жюри: 20 тестов matcher и награды (№1-20) и открытые примеры M01-M12."""
import copy
import math

import pytest

from evaluator.grader import evaluate

T = "wrong_date_or_fx_rate"
OTHER_TYPES = ["duplicate_invoice", "missing_or_partial_allocation", "period_cutoff"]
KNOWN = frozenset({"R1", "R2", "R3", "D1", "D2", "D3"})


def A(issue_id="ISS-A", ids=("R1",), evidence=(("D1", "D2"),), amount=10000, tol=0, typ=T):
    return {"issue_id": issue_id, "type": typ, "affected_ids": list(ids), "amount_minor": amount,
            "currency": "KZT", "evidence_sets": [list(s) for s in evidence], "tolerance_minor": tol}


def F(ids=("R1",), evidence=("D1", "D2"), amount=10000, cur="KZT", typ=T, text="ok"):
    return {"type": typ, "affected_ids": list(ids), "evidence_ids": list(evidence),
            "claimed_amount_minor": amount, "currency": cur, "explanation": text}


def run(findings, truth, steps=10, known=KNOWN):
    r = evaluate(findings, truth, steps, known_ids=known)
    return r


def check(r, tp, fp, fn, reward):
    assert (r.metrics.tp, r.metrics.fp, r.metrics.fn) == (tp, fp, fn)
    assert math.isclose(r.reward, reward, abs_tol=1e-9)


def test_01_correct():
    r = run([F()], [A()])
    check(r, 1, 0, 0, 0.98)
    assert r.metrics.precision == r.metrics.recall == r.metrics.f1 == 1


def test_02_empty_findings():
    r = run([], [A()])
    check(r, 0, 0, 1, -1)
    assert (r.metrics.precision, r.metrics.recall, r.metrics.f1) == (0, 0, 0)


def test_03_clean_empty():
    r = run([], [])
    check(r, 0, 0, 0, 0.98)
    assert (r.metrics.precision, r.metrics.recall, r.metrics.f1) == (0, 0, 1)


def test_04_clean_false_positive():
    check(run([F()], []), 0, 1, 0, -1)


def test_05_duplicates():
    r = run([F(), F()], [A()])
    check(r, 1, 1, 0, 47 / 150)
    assert math.isclose(r.metrics.f1, 2 / 3)


@pytest.mark.parametrize("n,mod", [
    (6, dict(typ="duplicate_invoice")),
    (7, dict(ids=("R2",))),
    (8, dict(amount=10001)),
    (9, dict(cur="USD")),
    (10, dict(evidence=("D1",))),
    (11, dict(evidence=("D1", "U"))),
])
def test_06_to_11_single_wrong_field(n, mod):
    check(run([F(**mod)], [A()]), 0, 1, 1, -1)


def test_12_shotgun_attack():
    recs = ["R1", "R2", "R3", "R4", "R5"]
    known = KNOWN | {"R4", "R5"}
    types = [T] + OTHER_TYPES
    shots = [F(ids=(r,), typ=t) for r in recs for t in types]
    assert len(shots) == 20
    r = run(shots, [A()], known=known)
    check(r, 1, 19, 0, -871 / 1050)
    assert math.isclose(r.metrics.f1, 2 / 21)


def test_13_one_of_two():
    r = run([F()], [A(), A("ISS-B", ids=("R2",), evidence=(("D2", "D3"),))])
    check(r, 1, 0, 1, 47 / 150)


def test_14_one_plus_false():
    r = run([F(), F(ids=("R3",))], [A(), A("ISS-B", ids=("R2",), evidence=(("D2", "D3"),))])
    check(r, 1, 1, 1, -0.02)


@pytest.mark.parametrize("steps,expected", [(0, 1.0), (10, 0.98), (50, 0.90), (100, 0.80)])
def test_15_to_17_inspection_cost(steps, expected):
    r = run([F()], [A()], steps=steps)
    assert math.isclose(r.reward, expected, abs_tol=1e-9)


def test_18_31_findings_invalid():
    r = run([F()] * 31, [A()])
    assert r.reward == -1 and r.metrics is None and r.schema_errors


def test_19_11_evidence_ids_invalid():
    ev = [f"D{i}" for i in range(11)]
    r = run([F(evidence=ev)], [A()])
    assert r.reward == -1 and r.metrics is None


def test_20_long_explanation_invalid():
    r = run([F(text="x" * 1001)], [A()])
    assert r.reward == -1 and r.metrics is None
    assert run([F(text="x" * 1000)], [A()]).metrics is not None


def test_explanation_counts_unicode_code_points_not_bytes():
    assert run([F(text="я" * 1000)], [A()]).metrics is not None
    assert run([F(text="я" * 1001)], [A()]).metrics is None


# ---------- M01-M12 ----------
AB = lambda: A(ids=("R1", "R2"))


def test_m01_duplicate_id_in_list_is_a_set():
    check(run([F(ids=("R2", "R1", "R1"))], [AB()]), 1, 0, 0, 0.98)


def test_m02_extra_existing_unaffected_id():
    check(run([F(ids=("R1", "R2", "R3"))], [AB()]), 0, 1, 1, -1)


def test_m03_extra_existing_evidence_ok():
    check(run([F(ids=("R1", "R2"), evidence=("D2", "D1", "D3"))], [AB()]), 1, 0, 0, 0.98)


def test_m04_unknown_evidence_poisons_finding():
    check(run([F(ids=("R1", "R2"), evidence=("D1", "D2", "U"))], [AB()]), 0, 1, 1, -1)


def test_m05_eleven_evidence_ids_counted_before_dedup():
    r = run([F(ids=("R1", "R2"), evidence=["D1"] * 10 + ["D2"])], [AB()])
    assert r.reward == -1 and r.metrics is None


@pytest.mark.parametrize("cur", ["kzt", " KZT", "KZT "])
def test_m06_currency_is_exact(cur):
    check(run([F(ids=("R1", "R2"), cur=cur)], [AB()]), 0, 1, 1, -1)


def test_m07_tau_zero():
    check(run([F(ids=("R1", "R2"), amount=10001)], [AB()]), 0, 1, 1, -1)


def test_m08_tau_one_inclusive():
    t = [A(ids=("R1", "R2"), tol=1)]
    check(run([F(ids=("R1", "R2"), amount=10001)], t), 1, 0, 0, 0.98)
    check(run([F(ids=("R1", "R2"), amount=9999)], t), 1, 0, 0, 0.98)
    check(run([F(ids=("R1", "R2"), amount=10002)], t), 0, 1, 1, -1)


@pytest.mark.parametrize("bad", ["10000", True, 10000.0, None])
def test_m09_amount_type(bad):
    r = run([F(ids=("R1", "R2"), amount=bad)], [AB()])
    assert r.reward == -1 and r.metrics is None


def test_m10_max_matching_beats_greedy():
    a = A("ISS-A", ids=("R1",), evidence=(("D1",),))
    b = A("ISS-B", ids=("R1",), evidence=(("D1", "D2"),))
    f0 = F(ids=("R1",), evidence=("D1", "D2"))      # подходит A и B
    f1 = F(ids=("R1",), evidence=("D1",))           # подходит только A
    check(run([f0, f1], [a, b]), 2, 0, 0, 1 - 0.02 * 0.1 - 0.0 if False else 0.98)


def test_m11_permutation_does_not_change_score():
    a = A("ISS-A", ids=("R1",))
    b = A("ISS-B", ids=("R1",))
    f0, f1 = F(), F(text="другой")
    r1, r2 = run([f0, f1], [a, b]), run([f1, f0], [b, a])
    check(r1, 2, 0, 0, 0.98)
    assert (r1.metrics, r1.reward) == (r2.metrics, r2.reward)


def test_m11_tie_break_assigns_smallest_issue_to_smallest_index():
    from evaluator.matcher import match
    a = A("ISS-A", ids=("R1",))
    b = A("ISS-B", ids=("R1",))
    assert match([F(), F()], [b, a], known_ids=KNOWN) == [(0, 1), (1, 0)]  # A (индекс 1 в truth) -> вывод 0


def test_m12_identical_findings_not_merged():
    check(run([F(ids=("R1", "R2")), F(ids=("R1", "R2"))], [AB()]), 1, 1, 0, 47 / 150)


def test_unknown_type_is_fp_not_structure_error():
    r = run([F(typ="brand_new_type")], [A()])
    assert r.metrics is not None
    check(r, 0, 1, 1, -1)


def test_object_wrapper_and_list_are_both_accepted():
    assert run({"findings": [F()]}, [A()]).metrics.tp == 1
    assert run({"findings": [F()], "extra": 1}, [A()]).metrics is None


def test_missing_field_is_structure_error():
    f = F()
    del f["evidence_ids"]
    assert run([f], [A()]).metrics is None


def test_judge_does_not_mutate_inputs():
    fs, ts = [F(), F()], [A()]
    a, b = copy.deepcopy(fs), copy.deepcopy(ts)
    run(fs, ts)
    assert fs == a and ts == b
