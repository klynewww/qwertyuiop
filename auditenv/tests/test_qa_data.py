"""Валидатор равенств и валидатор доказательств: чистые данные проходят, испорченные ловятся."""
import copy
import json

import pytest

from env.loader import load_open_company
from qa.evidence import validate_evidence, validate_evidence_dir
from qa.generated import validate_generated, validate_generated_dir

PROFILES = ["clean", "easy", "medium", "hard"]


@pytest.mark.parametrize("profile", PROFILES)
def test_generated_data_has_no_defects(gen, profile):
    assert validate_generated_dir(gen[profile]) == []


@pytest.mark.parametrize("profile", PROFILES)
def test_truth_is_provable_from_open_data(gen, profile):
    assert validate_evidence_dir(gen[profile]) == []


def company(gen, profile="hard"):
    return copy.deepcopy(load_open_company(gen[profile] / "public"))


def codes(issues):
    return {i.code for i in issues}


# ---------- мутации открытых данных: валидатор равенств обязан поймать ----------
def mut_ledger_debit(c):
    c["tables"]["ledger"][0]["debit_minor"] = str(int(c["tables"]["ledger"][0]["debit_minor"]) + 1)


def mut_invoice_total(c):
    c["tables"]["invoices"][0]["total_minor"] += 100


def mut_line_amount(c):
    c["tables"]["invoices"][0]["line_items"][0]["amount_minor"] += 1


def mut_unknown_invoice_ref(c):
    c["tables"]["payments"][0]["invoice_refs"][0]["invoice_id"] = "INV-999999"


def mut_cash(c):
    c["tables"]["payments"][0]["amount_minor"] += 7


def mut_bad_rate(c):
    c["tables"]["fx"][0]["rate"] = "-1.00000000"


def mut_rate_precision(c):
    c["tables"]["fx"][0]["rate"] = "1.5"


def mut_drop_fx(c):
    c["tables"]["fx"].pop(0)


def mut_dup_invoice_id(c):
    c["tables"]["invoices"][1]["invoice_id"] = c["tables"]["invoices"][0]["invoice_id"]


def mut_count(c):
    c["manifest"]["counts"]["payments"] += 1


def mut_unposted_invoice(c):
    iid = c["tables"]["invoices"][0]["invoice_id"]
    c["tables"]["ledger"] = [r for r in c["tables"]["ledger"] if not (r["invoice_id"] == iid and r["payment_id"] == "null")]


def mut_both_sides(c):
    c["tables"]["ledger"][0]["credit_minor"] = c["tables"]["ledger"][0]["debit_minor"]


def mut_text_amount(c):
    c["tables"]["ledger"][2]["debit_minor"] = "12.5"


def mut_bad_currency(c):
    c["tables"]["ledger"][0]["currency"] = "XXX"


def mut_due_before_issue(c):
    inv = c["tables"]["invoices"][0]
    inv["due_at"] = "2000-01-01T00:00:00+00:00"


@pytest.mark.parametrize("mutate,code", [
    (mut_ledger_debit, "UNBALANCED"), (mut_invoice_total, "TOTAL_MISMATCH"),
    (mut_line_amount, "LINE_MISMATCH"), (mut_unknown_invoice_ref, "UNKNOWN_REF"),
    (mut_cash, "CASH_MISMATCH"), (mut_bad_rate, "BAD_RATE"), (mut_rate_precision, "BAD_RATE"),
    (mut_drop_fx, "FX_GAP"), (mut_dup_invoice_id, "DUPLICATE_ID"), (mut_count, "COUNT_MISMATCH"),
    (mut_unposted_invoice, "INVOICE_NOT_POSTED"), (mut_both_sides, "BAD_SIDES"),
    (mut_text_amount, "BAD_AMOUNT"), (mut_bad_currency, "UNKNOWN_CURRENCY"),
    (mut_due_before_issue, "BAD_DATES"),
])
def test_data_validator_catches_mutation(gen, mutate, code):
    c = company(gen)
    mutate(c)
    assert code in codes(validate_generated(c))


def test_data_validator_ignores_business_errors(gen):
    """Ошибки аудита (дубль, неверный курс...) в данных есть, а равенств они не нарушают."""
    assert validate_generated_dir(gen["hard"]) == []
    truth = json.loads((gen["hard"] / "private" / "truth.json").read_text(encoding="utf-8"))
    assert len(truth["issues"]) >= 8


# ---------- мутации эталона: валидатор доказательств обязан поймать ----------
def truth_of(gen, profile="hard"):
    return json.loads((gen[profile] / "private" / "truth.json").read_text(encoding="utf-8"))


def ev_codes(gen, mutate, profile="hard"):
    t = truth_of(gen, profile)
    mutate(t)
    return codes(validate_evidence(company(gen, profile), t))


def issue_of(t, typ):
    return next(i for i in t["issues"] if i["type"] == typ)


@pytest.mark.parametrize("name,mutate,code", [
    ("unknown affected id", lambda t: t["issues"][0]["affected_ids"].append("INV-999999"), "UNKNOWN_ID"),
    ("amount changed", lambda t: t["issues"][0].update(expected_amount_minor=t["issues"][0]["expected_amount_minor"] + 5), "AMOUNT_NOT_REPRODUCIBLE"),
    ("currency changed", lambda t: t["issues"][0].update(currency="XXX"), "BAD_CURRENCY"),
    ("bad type", lambda t: t["issues"][0].update(type="made_up"), "BAD_TYPE"),
    ("empty evidence", lambda t: t["issues"][0].update(acceptable_evidence_sets=[]), "NO_EVIDENCE"),
    ("unrelated evidence", lambda t: t["issues"][0]["acceptable_evidence_sets"].append(["DOC-INV-000001", "DOC-INV-000002"]), "UNRELATED_EVIDENCE"),
    ("unknown evidence id", lambda t: t["issues"][0]["acceptable_evidence_sets"][0].append("R999999"), "UNKNOWN_ID"),
    ("dropped issue", lambda t: t["issues"].pop(0), "UNEXPECTED_FINDING"),
    ("fake issue", lambda t: t["issues"].append({**t["issues"][0], "issue_id": "ISS-099", "affected_ids": ["INV-000001", "INV-000002"]}), "NOT_PROVABLE"),
    ("repeated issue id", lambda t: t["issues"][1].update(issue_id=t["issues"][0]["issue_id"]), "DUPLICATE_ISSUE"),
    ("zero amount", lambda t: t["issues"][0].update(expected_amount_minor=0), "BAD_AMOUNT"),
    ("float amount", lambda t: t["issues"][0].update(expected_amount_minor=1.5), "BAD_AMOUNT"),
    ("no issues key", lambda t: t.pop("issues"), "NO_ISSUES"),
])
def test_evidence_validator_catches(gen, name, mutate, code):
    assert code in ev_codes(gen, mutate), name


def test_evidence_validator_rejects_wrong_evidence_set_for_each_type(gen):
    """Набор из ID, не доказывающий ошибку, не должен приниматься ни для одного из четырёх типов."""
    for typ in ("duplicate_invoice", "missing_or_partial_allocation", "wrong_date_or_fx_rate", "period_cutoff"):
        t = truth_of(gen)
        iss = issue_of(t, typ)
        iss["acceptable_evidence_sets"] = [[iss["affected_ids"][0]]]  # только сам ID, без доказательств
        assert "EVIDENCE_NOT_ACCEPTED" in codes(validate_evidence(company(gen), t)), typ


def test_clean_company_with_fake_issue_is_flagged(gen):
    t = truth_of(gen, "clean")
    t["issues"].append({"issue_id": "ISS-001", "type": "duplicate_invoice",
                        "affected_ids": ["INV-000001", "INV-000002"], "expected_amount_minor": 100,
                        "currency": "USD", "amount_tolerance_minor": 0,
                        "acceptable_evidence_sets": [["DOC-INV-000001", "DOC-INV-000002"]]})
    assert "NOT_PROVABLE" in ev_codes_from(gen, t, "clean")


def ev_codes_from(gen, t, profile):
    return codes(validate_evidence(company(gen, profile), t))


def test_all_four_types_and_both_scenarios_appear_in_hard(gen):
    t = truth_of(gen)
    assert {i["type"] for i in t["issues"]} == {"duplicate_invoice", "missing_or_partial_allocation",
                                                 "wrong_date_or_fx_rate", "period_cutoff"}
    assert {i["scenario"] for i in t["issues"]} >= {"partial_cross_currency_boundary", "duplicate_with_fx_error"}
