"""Правила аудита: чистые функции над ОТКРЫТЫМИ таблицами компании.

Пишутся строго по policy.json (правила компании), а не по коду генератора.
Используются в двух местах:
  * политика «дотошный аудитор» (policies/careful.py) ищет ими ошибки;
  * валидатор доказательств (qa/evidence.py) проверяет, что каждая ошибка
    из truth выводится из открытых данных и что чистые «отвлекающие» случаи
    (подписки, возврат -> повторный счёт, счета у границы периода) не
    выглядят ошибками.

Правила нарушаются четырьмя способами (имена типов как в генераторе A):
  duplicate_invoice, missing_or_partial_allocation, wrong_date_or_fx_rate,
  period_cutoff.
"""
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal

T_DUP = "duplicate_invoice"
T_ALLOC = "missing_or_partial_allocation"
T_FX = "wrong_date_or_fx_rate"
T_CUT = "period_cutoff"
ALL_TYPES = (T_DUP, T_ALLOC, T_FX, T_CUT)

ACC_AR, ACC_REV, ACC_UNAPPL = "1200", "4000", "2100"


def fx_id(day: str, src: str, tgt: str) -> str:
    return f"FX:{day}:{src}:{tgt}"


def convert_minor(amount: int, src: str, tgt: str, rate: Decimal, decimals: dict) -> int:
    """Перевод суммы в минимальных единицах src -> tgt (tgt за 1 src), округление half-even."""
    scale = Decimal(10) ** (decimals[tgt] - decimals[src])
    return int((Decimal(amount) * rate * scale).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def _ledger_int(row, key):
    return int(row[key])


def _none(v):
    return None if v in (None, "null", "") else v


def _items_signature(inv):
    return tuple(sorted((i["description"], i["quantity"], i["unit_price_minor"]) for i in inv["line_items"]))


def _offset(manifest) -> timezone:
    sign = -1 if manifest["timezone"].startswith("-") else 1
    h, m = manifest["timezone"].lstrip("+-").split(":")
    return timezone(sign * timedelta(hours=int(h), minutes=int(m)))


def find_duplicates(invoices) -> list[dict]:
    """Одинаковые клиент, валюта и позиции у двух положительных счетов без кредит-ноты между ними."""
    credit_notes = {}  # id исходного счёта -> моменты кредит-нот на него
    for inv in invoices:
        if inv["total_minor"] < 0:
            for item in inv["line_items"]:
                desc = item["description"]
                if desc.startswith("Credit note for "):
                    ref = desc[len("Credit note for "):].split(":", 1)[0]
                    credit_notes.setdefault(ref, set()).add(inv["issued_at"])
    groups = {}
    for inv in invoices:
        if inv["total_minor"] > 0:
            groups.setdefault((inv["entity_id"], inv["currency"], _items_signature(inv)), []).append(inv)
    found = []
    for group in groups.values():
        group = sorted(group, key=lambda i: (i["issued_at"], i["invoice_id"]))
        first = group[0]
        for later in group[1:]:
            excused = any(first["issued_at"] < t < later["issued_at"]
                          for t in credit_notes.get(first["invoice_id"], ()))
            if excused:
                continue
            ids = sorted([first["invoice_id"], later["invoice_id"]])
            found.append({
                "type": T_DUP, "affected_ids": ids, "amount_minor": later["total_minor"],
                "currency": later["currency"],
                "evidence_sets": [sorted([first["doc_id"], later["doc_id"]])],
                "evidence_pool": set(ids) | {first["doc_id"], later["doc_id"]},
                "explanation": (f"Счета {ids[0]} и {ids[1]}: тот же клиент, те же позиции и сумма, "
                                f"кредит-ноты между ними нет."),
            })
    return found


def find_allocation_gaps(invoices, payments, ledger) -> list[dict]:
    """Платёж по документу распределён на счёт полностью, а в проводках на счёт отнесена меньшая часть."""
    by_id = {i["invoice_id"]: i for i in invoices}
    applied, ar_row, unapplied_row = {}, {}, {}
    for r in ledger:
        pid = _none(r["payment_id"])
        if pid is None:
            continue
        if r["account"] == ACC_AR and _none(r["invoice_id"]):
            key = (pid, r["invoice_id"])
            applied[key] = applied.get(key, 0) + _ledger_int(r, "credit_minor")
            ar_row.setdefault(key, r["row_id"])
        elif r["account"] == ACC_UNAPPL:
            unapplied_row.setdefault(pid, r["row_id"])
    found = []
    for p in payments:
        for ref in p["invoice_refs"]:
            iid = ref["invoice_id"]
            gap = ref["allocated_minor"] - applied.get((p["payment_id"], iid), 0)
            if gap <= 0:
                continue
            ids = sorted([p["payment_id"], iid])
            doc = p["doc_id"]
            sets, pool = [], {*ids, doc}
            if p["payment_id"] in unapplied_row:
                sets.append(sorted([doc, unapplied_row[p["payment_id"]]]))
                pool.add(unapplied_row[p["payment_id"]])
            if (p["payment_id"], iid) in ar_row:
                sets.append(sorted([doc, ar_row[(p["payment_id"], iid)]]))
                pool.add(ar_row[(p["payment_id"], iid)])
            if not sets:
                sets.append([doc])
            found.append({
                "type": T_ALLOC, "affected_ids": ids, "amount_minor": gap,
                "currency": by_id[iid]["currency"], "evidence_sets": sets, "evidence_pool": pool,
                "explanation": (f"Платёж {p['payment_id']} по документу закрывает счёт {iid} на "
                                f"{ref['allocated_minor']}, в проводках на счёт отнесено на {gap} меньше."),
            })
    return found


def needed_fx_ids(invoices, payments) -> list[str]:
    """Какие курсы нужны, чтобы проверить кросс-валютные платежи (дата курса = локальная дата платежа)."""
    by_id = {i["invoice_id"]: i for i in invoices}
    ids = []
    for p in payments:
        for ref in p["invoice_refs"]:
            inv = by_id.get(ref["invoice_id"])
            if inv and inv["currency"] != p["currency"]:
                ids.append(fx_id(p["timestamp"][:10], p["currency"], inv["currency"]))
    return sorted(set(ids))


def find_fx_errors(invoices, payments, fx_rate, decimals, ledger=()) -> list[dict]:
    """Кросс-валютный платёж: зачтённая сумма должна равняться конвертации по курсу даты платежа."""
    by_id = {i["invoice_id"]: i for i in invoices}
    pay_rows = {}
    for r in ledger:
        if _none(r["payment_id"]):
            pay_rows.setdefault(r["payment_id"], []).append(r)
    found = []
    for p in payments:
        if len(p["invoice_refs"]) != 1:
            continue
        ref = p["invoice_refs"][0]
        inv = by_id.get(ref["invoice_id"])
        if inv is None or inv["currency"] == p["currency"]:
            continue
        day = p["timestamp"][:10]
        rate = fx_rate(day, p["currency"], inv["currency"])
        if rate is None:
            continue
        correct = convert_minor(p["amount_minor"], p["currency"], inv["currency"], rate, decimals)
        if correct == ref["allocated_minor"]:
            continue
        ids = sorted([p["payment_id"], inv["invoice_id"]])
        fid = fx_id(day, p["currency"], inv["currency"])
        sets, pool = [sorted([p["doc_id"], fid])], {*ids, p["doc_id"], fid}
        rows = pay_rows.get(p["payment_id"], [])
        cash = next((r for r in rows if r["account"] == "1000"), None)
        ar = next((r for r in rows if r["account"] == ACC_AR and r["invoice_id"] == inv["invoice_id"]), None)
        if cash and ar:
            pool |= {cash["row_id"], ar["row_id"]}
            if not any(r["account"] == ACC_UNAPPL for r in rows):  # платёж отнесён целиком: доказать можно и по проводкам
                sets.append(sorted([cash["row_id"], ar["row_id"], fid]))
        found.append({
            "type": T_FX, "affected_ids": ids,
            "amount_minor": abs(inv["total_minor"] - ref["allocated_minor"]),
            "currency": inv["currency"], "evidence_sets": sets,
            "evidence_pool": pool,
            "explanation": (f"Платёж {p['payment_id']}: по курсу {day} зачесть надо {correct}, "
                            f"зачтено {ref['allocated_minor']}."),
        })
    return found


def find_cutoff_errors(manifest, invoices, ledger) -> list[dict]:
    """Дата счёта и дата его проводки лежат по разные стороны конца отчётного периода."""
    tz = _offset(manifest)
    end = datetime.fromisoformat(manifest["period_end"]).date()

    def side(ts: str) -> bool:
        return datetime.fromisoformat(ts).astimezone(tz).date() <= end

    rows = {}
    for r in ledger:
        if _none(r["payment_id"]) is None and _none(r["invoice_id"]):
            rows.setdefault(r["invoice_id"], []).append(r)
    found = []
    for inv in invoices:
        inv_rows = rows.get(inv["invoice_id"])
        if not inv_rows:
            continue
        if side(inv["issued_at"]) == side(inv_rows[0]["timestamp"]):
            continue
        rev = next(r for r in inv_rows if r["account"] == ACC_REV)
        ar = next(r for r in inv_rows if r["account"] == ACC_AR)
        found.append({
            "type": T_CUT, "affected_ids": [inv["invoice_id"]], "amount_minor": abs(inv["total_minor"]),
            "currency": inv["currency"],
            "evidence_sets": [sorted([inv["doc_id"], rev["row_id"]]), sorted([inv["doc_id"], ar["row_id"]])],
            "evidence_pool": {inv["invoice_id"], inv["doc_id"], rev["row_id"], ar["row_id"]},
            "explanation": (f"Счёт {inv['invoice_id']} датирован {inv['issued_at'][:10]}, а проводка "
                            f"{inv_rows[0]['timestamp'][:10]}: по разные стороны конца периода {end}."),
        })
    return found


def find_issues(manifest, invoices, payments, ledger, fx_rate, types=ALL_TYPES) -> list[dict]:
    """Все найденные нарушения правил. fx_rate(date, src, tgt) -> Decimal или None."""
    found = []
    if T_DUP in types:
        found += find_duplicates(invoices)
    if T_ALLOC in types:
        found += find_allocation_gaps(invoices, payments, ledger)
    if T_FX in types:
        found += find_fx_errors(invoices, payments, fx_rate, manifest["currencies"], ledger)
    if T_CUT in types:
        found += find_cutoff_errors(manifest, invoices, ledger)
    found.sort(key=lambda f: (f["affected_ids"], f["type"]))
    return found


def to_finding(issue: dict) -> dict:
    """Вывод для submit: канонические affected_ids и один полный набор доказательств."""
    ids = set(issue["affected_ids"])
    return {
        "type": issue["type"], "affected_ids": sorted(ids)[:10],
        "evidence_ids": sorted(issue["evidence_sets"][0])[:10],
        "claimed_amount_minor": issue["amount_minor"], "currency": issue["currency"],
        "explanation": issue["explanation"][:1000],
    }
