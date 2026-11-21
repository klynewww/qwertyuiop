"""Валидатор равенств по ОТКРЫТЫМ файлам компании.

Читает только open/ через env.loader. Проверяет арифметику и согласованность
таблиц, а не «бизнес-ошибки»: дубль счёта сам по себе не нарушает равенств
(и не должен ловиться здесь, иначе validator нельзя применять к компаниям с ошибкой).

ДОПУЩЕНИЕ: колонки и поля берутся из tests/fixtures/mini/open (сверить с контрактом).
"""
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from env.loader import ID_FIELDS, load_open_company


@dataclass(frozen=True)
class Issue:
    code: str
    where: str
    message: str


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _fmt_amount(minor: int, decimals: int) -> str:
    sign = "-" if minor < 0 else ""
    m = abs(minor)
    if decimals == 0:
        return f"{sign}{m}"
    return f"{sign}{m // 10**decimals}.{m % 10**decimals:0{decimals}d}"


def validate_open_dir(open_dir) -> list[Issue]:
    return validate_company(load_open_company(open_dir))


def validate_company(company: dict) -> list[Issue]:
    issues: list[Issue] = []
    add = lambda code, where, msg: issues.append(Issue(code, where, msg))
    manifest, tables, docs = company["manifest"], company["tables"], company["documents"]
    currencies: dict = manifest.get("currencies", {})
    start, end = manifest.get("period_start"), manifest.get("period_end")

    # 1. уникальность id и непустота
    for name, rows in tables.items():
        key = ID_FIELDS[name]
        seen = set()
        for r in rows:
            rid = r.get(key)
            if not rid:
                add("EMPTY_ID", name, f"запись без {key}")
            elif rid in seen:
                add("DUPLICATE_ID", f"{name}:{rid}", "id повторяется в таблице")
            seen.add(rid)

    # 2. суммы, валюты, даты в ledger / invoices / payments
    money_tables = {"ledger": tables["ledger"], "invoices": tables["invoices"],
                    "payments": tables["payments"]}
    for name, rows in money_tables.items():
        key = ID_FIELDS[name]
        for r in rows:
            where = f"{name}:{r.get(key)}"
            a = _int(r.get("amount_minor"))
            if a is None or a <= 0:
                add("BAD_AMOUNT", where, f"amount_minor должен быть целым > 0: {r.get('amount_minor')!r}")
            if r.get("currency") not in currencies:
                add("UNKNOWN_CURRENCY", where, f"валюта вне manifest: {r.get('currency')!r}")
            d = r.get("date")
            if not d or (start and d < start) or (end and d > end):
                add("DATE_OUT_OF_PERIOD", where, f"дата вне периода: {d!r}")

    # 3. дебет = кредит (суммарно по валюте) и счета проводки различны
    debit, credit = defaultdict(int), defaultdict(int)
    for r in tables["ledger"]:
        a = _int(r.get("amount_minor"))
        if not r.get("debit_account") or not r.get("credit_account"):
            add("MISSING_ACCOUNT", f"ledger:{r.get('row_id')}", "нет дебетового или кредитового счёта")
        elif r["debit_account"] == r["credit_account"]:
            add("SAME_ACCOUNT", f"ledger:{r.get('row_id')}", "дебет и кредит на одном счёте")
        if a:
            debit[r.get("currency")] += a
            credit[r.get("currency")] += a
    for cur in set(debit) | set(credit):
        if debit[cur] != credit[cur]:
            add("UNBALANCED", f"ledger:{cur}", f"дебет {debit[cur]} != кредит {credit[cur]}")

    # 4. каждый счёт и платёж имеет ровно одну проводку с тем же ref_id, суммой, валютой, датой
    by_ref = defaultdict(list)
    for r in tables["ledger"]:
        by_ref[r.get("ref_id")].append(r)
    for name in ("invoices", "payments"):
        key = ID_FIELDS[name]
        for r in tables[name]:
            where, rid = f"{name}:{r[key]}", r[key]
            rows = by_ref.get(rid, [])
            if len(rows) != 1:
                add("LEDGER_LINK", where, f"проводок с ref_id={rid}: {len(rows)}, ожидалась 1")
                continue
            l = rows[0]
            if _int(l["amount_minor"]) != _int(r.get("amount_minor")) or l["currency"] != r.get("currency"):
                add("LEDGER_AMOUNT", where, "сумма/валюта не совпадает с проводкой")
            if l["date"] != r.get("date"):
                add("LEDGER_DATE", where, "дата не совпадает с проводкой")
    all_ids = {r[ID_FIELDS[n]] for n in ("invoices", "payments") for r in tables[n]}
    for r in tables["ledger"]:
        if r.get("ref_id") not in all_ids:
            add("ORPHAN_LEDGER", f"ledger:{r.get('row_id')}", f"ref_id {r.get('ref_id')!r} не найден")

    # 5. платежи согласуются со счетами
    inv = {r["invoice_id"]: r for r in tables["invoices"]}
    for p in tables["payments"]:
        where = f"payments:{p['payment_id']}"
        i = inv.get(p.get("invoice_id"))
        if i is None:
            add("PAYMENT_NO_INVOICE", where, f"счёта {p.get('invoice_id')!r} нет")
            continue
        if i["entity_id"] != p.get("entity_id"):
            add("PAYMENT_ENTITY", where, "контрагент платежа не совпадает со счётом")
        if i["currency"] != p.get("currency"):
            add("PAYMENT_CURRENCY", where, "валюта платежа не совпадает со счётом")
        if p["date"] < i["date"]:
            add("PAYMENT_BEFORE_INVOICE", where, "платёж раньше счёта")

    # 6. документы существуют и согласуются с записью
    for name, key in (("invoices", "invoice_id"), ("payments", "payment_id")):
        for r in tables[name]:
            where = f"{name}:{r[key]}"
            doc = docs.get(r.get("doc_id"))
            text = None if doc is None else (doc["text"] if "text" in doc else str(doc))
            if text is None:
                add("DOC_MISSING", where, f"документа {r.get('doc_id')!r} нет")
                continue
            a, cur = _int(r.get("amount_minor")), r.get("currency")
            # в документе счёта есть контрагент, в документе платежа — счёт-основание
            want = [r[key], r.get("date"), r.get("entity_id") if name == "invoices" else r.get("invoice_id")]
            if a is not None and cur in currencies:
                want.append(f"{_fmt_amount(a, currencies[cur])} {cur}")
            for token in want:
                if token and token not in text:
                    add("DOC_MISMATCH", where, f"в документе {r['doc_id']} нет {token!r}")

    # 7. курсы
    for r in tables["fx"]:
        where = f"fx:{r.get('row_id')}"
        try:
            ok = Decimal(r["rate"]) > 0
        except (InvalidOperation, KeyError, TypeError):
            ok = False
        if not ok:
            add("BAD_RATE", where, f"курс не положительное число: {r.get('rate')!r}")
        for col in ("from_currency", "to_currency"):
            if r.get(col) not in currencies:
                add("UNKNOWN_CURRENCY", where, f"{col} вне manifest: {r.get(col)!r}")
    return issues


def is_valid(open_dir) -> bool:
    return not validate_open_dir(open_dir)
