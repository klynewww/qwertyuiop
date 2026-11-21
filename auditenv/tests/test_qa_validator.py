import copy

from env.loader import load_open_company
from qa.validator import validate_company, validate_open_dir
from tests.helpers import MINI

OPEN = MINI / "open"


def codes(company):
    return {i.code for i in validate_company(company)}


def mini():
    return copy.deepcopy(load_open_company(OPEN))


def test_mini_is_valid_even_with_duplicate_invoice():
    # дубль счёта не нарушает равенств: валидатор ловит только арифметику и ссылки
    assert validate_open_dir(OPEN) == []


def test_nonpositive_amount_in_ledger_row():
    c = mini()
    c["tables"]["ledger"][0]["amount_minor"] = "0"
    assert "BAD_AMOUNT" in codes(c)


def test_ledger_amount_differs_from_invoice():
    c = mini()
    c["tables"]["ledger"][0]["amount_minor"] = "100001"
    assert "LEDGER_AMOUNT" in codes(c)


def test_missing_ledger_row_for_invoice():
    c = mini()
    c["tables"]["ledger"] = [r for r in c["tables"]["ledger"] if r["ref_id"] != "INV-0003"]
    assert "LEDGER_LINK" in codes(c)


def test_orphan_ledger_row():
    c = mini()
    c["tables"]["ledger"][0]["ref_id"] = "INV-9999"
    assert {"ORPHAN_LEDGER", "LEDGER_LINK"} <= codes(c)


def test_same_account_and_missing_account():
    c = mini()
    c["tables"]["ledger"][0]["credit_account"] = c["tables"]["ledger"][0]["debit_account"]
    assert "SAME_ACCOUNT" in codes(c)
    c = mini()
    c["tables"]["ledger"][0]["debit_account"] = ""
    assert "MISSING_ACCOUNT" in codes(c)


def test_payment_without_invoice_and_wrong_entity_currency():
    c = mini()
    c["tables"]["payments"][0]["invoice_id"] = "INV-9999"
    assert "PAYMENT_NO_INVOICE" in codes(c)
    c = mini()
    c["tables"]["payments"][0]["entity_id"] = "ENT-777"
    assert "PAYMENT_ENTITY" in codes(c)
    c = mini()
    c["tables"]["payments"][0]["currency"] = "USD"
    assert "PAYMENT_CURRENCY" in codes(c)


def test_duplicate_id_and_unknown_currency_and_date():
    c = mini()
    c["tables"]["invoices"][1]["invoice_id"] = "INV-0001"
    assert "DUPLICATE_ID" in codes(c)
    c = mini()
    c["tables"]["ledger"][0]["currency"] = "GBP"
    assert "UNKNOWN_CURRENCY" in codes(c)
    c = mini()
    c["tables"]["invoices"][0]["date"] = "2026-04-01"
    assert "DATE_OUT_OF_PERIOD" in codes(c)


def test_documents_missing_or_inconsistent():
    c = mini()
    del c["documents"]["DOC-0001"]
    assert "DOC_MISSING" in codes(c)
    c = mini()
    c["documents"]["DOC-0002"] = "Счёт INV-0002 от 2026-03-10. Контрагент ENT-002. Сумма 999.00 EUR."
    assert "DOC_MISMATCH" in codes(c)


def test_bad_fx_rate():
    c = mini()
    c["tables"]["fx"][0]["rate"] = "abc"
    assert "BAD_RATE" in codes(c)
    c["tables"]["fx"][0]["rate"] = "0"
    assert "BAD_RATE" in codes(c)
