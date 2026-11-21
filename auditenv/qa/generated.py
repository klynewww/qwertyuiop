"""Валидатор равенств для данных генератора A (public/). Читает только открытые файлы.

Проверяет арифметику и ссылки, а не бизнес-ошибки: дубль, неверный курс,
частичное отнесение и сдвиг даты равенств не нарушают (иначе валидатор
нельзя было бы применять к компаниям с внесённой ошибкой).
Возвращает список qa.validator.Issue; пустой список = брака нет.
"""
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from env.layout import open_dir_of
from env.loader import load_open_company, record_id
from qa.validator import Issue


def validate_generated_dir(episode_dir) -> list[Issue]:
    root = Path(episode_dir)
    pub = open_dir_of(root)
    return validate_generated(load_open_company(pub))


def validate_generated(company: dict) -> list[Issue]:
    out: list[Issue] = []
    add = lambda code, where, msg: out.append(Issue(code, where, msg))  # noqa: E731
    m, t = company["manifest"], company["tables"]
    cur = m["currencies"]
    invoices = {i["invoice_id"]: i for i in t["invoices"]}
    payments = {p["payment_id"]: p for p in t["payments"]}

    # 1. счётчики и уникальность id
    counts = m["counts"]
    for key, table in (("ledger_rows", "ledger"), ("invoices", "invoices"),
                       ("payments", "payments"), ("fx_rows", "fx")):
        if counts[key] != len(t[table]):
            add("COUNT_MISMATCH", table, f"в manifest {counts[key]}, в файле {len(t[table])}")
    for name, rows in t.items():
        ids = [record_id(name, r) for r in rows]
        if len(ids) != len(set(ids)):
            add("DUPLICATE_ID", name, "id повторяются")
    doc_ids = [r["doc_id"] for n in ("invoices", "payments") for r in t[n]]
    if len(doc_ids) != len(set(doc_ids)):
        add("DUPLICATE_ID", "doc_id", "doc_id повторяются")

    # 2. счета: суммы позиций и валюты
    for inv in t["invoices"]:
        w = inv["invoice_id"]
        if inv["currency"] not in cur:
            add("UNKNOWN_CURRENCY", w, inv["currency"])
        total = 0
        for it in inv["line_items"]:
            if it["quantity"] * it["unit_price_minor"] != it["amount_minor"]:
                add("LINE_MISMATCH", w, "количество × цена != сумма позиции")
            total += it["amount_minor"]
        if total != inv["total_minor"] or total == 0:
            add("TOTAL_MISMATCH", w, f"позиции дают {total}, в счёте {inv['total_minor']}")
        if datetime.fromisoformat(inv["due_at"]) < datetime.fromisoformat(inv["issued_at"]):
            add("BAD_DATES", w, "срок оплаты раньше выписки")

    # 3. проводки: целые, ровно одна сторона, баланс по валютам и по документам
    per_ccy = defaultdict(lambda: [0, 0])
    per_doc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    inv_rows, pay_rows = defaultdict(list), defaultdict(list)
    for r in t["ledger"]:
        w = r["row_id"]
        try:
            d, c = int(r["debit_minor"]), int(r["credit_minor"])
        except ValueError:
            add("BAD_AMOUNT", w, "сумма не целое число")
            continue
        if (d > 0) == (c > 0) or d < 0 or c < 0:
            add("BAD_SIDES", w, "должна быть заполнена ровно одна сторона")
        if r["currency"] not in cur:
            add("UNKNOWN_CURRENCY", w, r["currency"])
        try:
            datetime.fromisoformat(r["timestamp"])
        except ValueError:
            add("BAD_DATES", w, "дата проводки не разбирается")
        per_ccy[r["currency"]][0] += d
        per_ccy[r["currency"]][1] += c
        doc = r["payment_id"] if r["payment_id"] != "null" else r["invoice_id"]
        per_doc[doc][r["currency"]][0] += d
        per_doc[doc][r["currency"]][1] += c
        if r["payment_id"] != "null":
            pay_rows[r["payment_id"]].append(r)
            if r["payment_id"] not in payments:
                add("UNKNOWN_REF", w, f"нет платежа {r['payment_id']}")
        else:
            inv_rows[r["invoice_id"]].append(r)
        if r["invoice_id"] != "null" and r["invoice_id"] not in invoices:
            add("UNKNOWN_REF", w, f"нет счёта {r['invoice_id']}")
    for ccy, (d, c) in per_ccy.items():
        if d != c:
            add("UNBALANCED", f"ledger:{ccy}", f"дебет {d} != кредит {c}")
    for doc, by_ccy in per_doc.items():
        for ccy, (d, c) in by_ccy.items():
            if d != c:
                add("UNBALANCED", doc, f"в {ccy} дебет {d} != кредит {c}")

    # 4. каждый счёт проведён: выручка и дебиторка на его сумму
    for iid, inv in invoices.items():
        rows = inv_rows.get(iid, [])
        ar = sum(int(r["debit_minor"]) - int(r["credit_minor"]) for r in rows if r["account"] == "1200")
        rev = sum(int(r["credit_minor"]) - int(r["debit_minor"]) for r in rows if r["account"] == "4000")
        if ar != inv["total_minor"] or rev != inv["total_minor"]:
            add("INVOICE_NOT_POSTED", iid, f"проводки {ar}/{rev}, сумма счёта {inv['total_minor']}")

    # 5. платежи: ссылки, валюты, кассовая проводка, распределение = дебиторка + 2100
    for pid, p in payments.items():
        refs = p["invoice_refs"]
        if not refs or any(r["invoice_id"] not in invoices for r in refs):
            add("UNKNOWN_REF", pid, "платёж ссылается на несуществующий счёт")
            continue
        inv_ccy = {invoices[r["invoice_id"]]["currency"] for r in refs}
        if len(inv_ccy) != 1:
            add("MIXED_CURRENCY", pid, "один платёж на счета в разных валютах")
            continue
        (ic,) = inv_ccy
        doc_alloc = sum(r["allocated_minor"] for r in refs)
        if ic == p["currency"] and doc_alloc != p["amount_minor"]:
            add("ALLOC_MISMATCH", pid, f"распределено {doc_alloc}, платёж {p['amount_minor']}")
        rows = pay_rows.get(pid, [])
        cash = sum(int(r["debit_minor"]) for r in rows if r["account"] == "1000")
        if cash != p["amount_minor"]:
            add("CASH_MISMATCH", pid, f"в кассу {cash}, платёж {p['amount_minor']}")
        posted = sum(int(r["credit_minor"]) for r in rows
                     if r["account"] in ("1200", "2100") and r["currency"] == ic)
        if posted != doc_alloc:
            add("PAYMENT_NOT_POSTED", pid, f"проводки закрывают {posted}, документ распределяет {doc_alloc}")

    # 6. курсы: положительные, 8 знаков, обратные пары согласованы, нет дыр по датам
    by_day = defaultdict(dict)
    for r in t["fx"]:
        try:
            rate = Decimal(r["rate"])
        except InvalidOperation:
            add("BAD_RATE", record_id("fx", r), "курс не число")
            continue
        if rate <= 0 or -rate.as_tuple().exponent != 8:
            add("BAD_RATE", record_id("fx", r), f"курс {r['rate']}: должен быть > 0 и с 8 знаками")
        by_day[r["date"]][(r["source_currency"], r["target_currency"])] = rate
    pairs = {(a, b) for a in cur for b in cur if a != b}
    for day, table in by_day.items():
        if set(table) != pairs:
            add("FX_GAP", day, "нет части валютных пар")
            continue
        for (a, b), rate in table.items():
            if abs(rate * table[(b, a)] - 1) > Decimal("0.001"):
                add("FX_INVERSE", day, f"{a}/{b} и {b}/{a} не обратны")
    return out
