"""Валидатор доказательств: эталон (truth) проверяется по открытым данным.

Для каждой ошибки из truth проверяем:
  1. структура и ссылки: все ID существуют (счета, платежи, документы, проводки, курсы);
  2. каждый допустимый набор доказательств состоит из ID, относящихся именно к этой ошибке;
  3. ошибка ВЫВОДИТСЯ из открытых данных правилами аудита (policies/audit_rules.py),
     то есть агент в принципе может её доказать: тип, ID, сумма и валюта совпадают;
  4. обратное: правила не находят ничего, чего нет в truth. Значит «отвлекающие»
     случаи (подписки, возврат -> повторный счёт, счета у границы) действительно чистые.
Возвращает список qa.validator.Issue; пустой список = эталон корректен.
"""
import json
from decimal import Decimal
from pathlib import Path

from env.layout import hidden_dir_of, open_dir_of
from env.loader import load_open_company, record_id
from policies import audit_rules as rules
from qa.validator import Issue


def validate_evidence_dir(episode_dir) -> list[Issue]:
    root = Path(episode_dir)
    company = load_open_company(open_dir_of(root))
    truth = json.loads((hidden_dir_of(root) / "truth.json").read_text(encoding="utf-8"))
    return validate_evidence(company, truth)


def _key(i):
    return (i["type"], tuple(sorted(i["affected_ids"])))


def validate_evidence(company: dict, truth: dict) -> list[Issue]:
    out: list[Issue] = []
    add = lambda code, where, msg: out.append(Issue(code, where, msg))  # noqa: E731
    t, manifest = company["tables"], company["manifest"]
    known = set(company["documents"])
    for name, rows in t.items():
        known |= {record_id(name, r) for r in rows}
    rates = {record_id("fx", r): Decimal(r["rate"]) for r in t["fx"]}

    issues = truth.get("issues")
    if not isinstance(issues, list):
        return [Issue("NO_ISSUES", "truth", "в truth нет списка issues")]
    seen_ids = set()
    for iss in issues:
        w = iss.get("issue_id", "?")
        if w in seen_ids:
            add("DUPLICATE_ISSUE", w, "issue_id повторяется")
        seen_ids.add(w)
        if iss.get("type") not in rules.ALL_TYPES:
            add("BAD_TYPE", w, f"неизвестный тип {iss.get('type')!r}")
        amount = iss.get("expected_amount_minor")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            add("BAD_AMOUNT", w, "сумма должна быть целым > 0")
        if iss.get("currency") not in manifest["currencies"]:
            add("BAD_CURRENCY", w, str(iss.get("currency")))
        if iss.get("amount_tolerance_minor", 0) < 0:
            add("BAD_TOLERANCE", w, "допуск < 0")
        affected = iss.get("affected_ids") or []
        if not affected or len(set(affected)) != len(affected):
            add("BAD_AFFECTED", w, "affected_ids пуст или с повторами")
        for x in affected:
            if x not in known:
                add("UNKNOWN_ID", w, f"affected_ids: {x} нет в данных")
        sets = iss.get("acceptable_evidence_sets") or []
        if not sets:
            add("NO_EVIDENCE", w, "нет допустимых наборов доказательств")
        for s in sets:
            if not s or len(set(s)) != len(s) or len(s) > 10:
                add("BAD_EVIDENCE", w, f"набор {s}: пустой, с повторами или длиннее 10")
            for x in s:
                if x not in known:
                    add("UNKNOWN_ID", w, f"доказательство {x} нет в данных")

    def fx_rate(day, src, tgt):
        return rates.get(rules.fx_id(day, src, tgt))

    derived = rules.find_issues(manifest, t["invoices"], t["payments"], t["ledger"], fx_rate)
    derived_by_key = {_key(d): d for d in derived}
    truth_keys = set()
    for iss in issues:
        k = _key(iss)
        truth_keys.add(k)
        d = derived_by_key.get(k)
        w = iss.get("issue_id", "?")
        if d is None:
            add("NOT_PROVABLE", w, "правила аудита не находят эту ошибку в открытых данных")
            continue
        if d["amount_minor"] != iss["expected_amount_minor"] or d["currency"] != iss["currency"]:
            add("AMOUNT_NOT_REPRODUCIBLE", w,
                f"правила дают {d['amount_minor']} {d['currency']}, в truth "
                f"{iss['expected_amount_minor']} {iss['currency']}")
        for s in iss["acceptable_evidence_sets"]:
            stray = set(s) - d["evidence_pool"]
            if stray:
                add("UNRELATED_EVIDENCE", w, f"в наборе {s} посторонние ID: {sorted(stray)}")
        if not any(sorted(s) in iss["acceptable_evidence_sets"] for s in d["evidence_sets"]):
            add("EVIDENCE_NOT_ACCEPTED", w, "набор доказательств из правил не входит в допустимые")
    for k, d in derived_by_key.items():
        if k not in truth_keys:
            add("UNEXPECTED_FINDING", "/".join(k[1]), f"правила видят {k[0]}, а в truth его нет")
    return out
