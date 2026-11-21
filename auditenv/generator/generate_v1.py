"""Генератор компаний для RL-среды финансового аудита (FinTech-кейс). Полная версия роли A.

Конвейер (порядок важен, он требуется в ТЗ: "сначала история, потом ошибки"):
  1. Строим ЧИСТУЮ скрытую историю: счета, платежи, курсы, проводки, отвлекающие случаи.
  2. Вносим ошибки как явные изменения и записываем причинное изменение
     (provenance.json) и эталон (truth.json).
  3. Выгружаем открытые файлы (их увидит агент) и закрытые (только оценщик).

Воспроизводимость: ВСЯ случайность идёт через random.Random(seed). Глобальный
модуль random не используется. Одинаковые (seed, config) -> одинаковые байты.
Деньги: только целые минимальные единицы. Курсы: Decimal, в файлах строки.

Типы ошибок (что именно ломается в данных):
  duplicate_invoice            копия существующего счёта с новым номером, оплаты нет
  missing_or_partial_allocation платёж в документе распределён на счёт полностью, а в
                                проводках на счёт отнесена только часть (или ничего):
                                остаток лежит на счёте 2100 "нераспределённые деньги"
  wrong_date_or_fx_rate        кросс-валютный платёж: сумма, зачтённая на счёт,
                                посчитана по курсу не той даты или по неверному курсу
  period_cutoff                дата документа и дата проводки лежат по разные стороны
                                конца отчётного периода
Взаимодействующие сценарии:
  partial_cross_currency_boundary  платёж у конца периода: неверная дата курса (курс
                                   следующего периода) И частичное отнесение на счёт
  duplicate_with_fx_error          счёт с неверным курсом при оплате И его дубль
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import random
import re
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

GENERATOR_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"

# Валюта -> число знаков после запятой (JPY без копеек: хорошая ловушка).
CURRENCIES = {"USD": 2, "EUR": 2, "KZT": 2, "JPY": 0}
BASE_CURRENCY = "USD"
START_RATES = {"EUR": "1.08000000", "KZT": "0.00200000", "JPY": "0.00670000"}  # USD за 1 ед.
FX_MARGIN_DAYS = 7          # курсы есть и за 7 дней до начала и после конца периода
Q8 = Decimal("0.00000001")

PRICE_RANGE_MAJOR = {
    "USD": (50, 3000), "EUR": (50, 3000),
    "KZT": (20000, 1500000), "JPY": (5000, 400000),
}
DESCRIPTIONS = [
    "Consulting hours", "Software license", "Hardware delivery", "Support plan",
    "Training session", "Cloud hosting", "Design work", "Maintenance visit",
]
TZ_OFFSETS_HOURS = [0, 3, 5, 6]

ACC_CASH, ACC_AR, ACC_FXCLR, ACC_UNAPPL, ACC_REV = "1000", "1200", "1500", "2100", "4000"
ACCOUNTS = {
    "cash": ACC_CASH, "receivable": ACC_AR, "fx_clearing": ACC_FXCLR,
    "unapplied_cash": ACC_UNAPPL, "revenue": ACC_REV,
}
ROUNDING = {"mode": "ROUND_HALF_EVEN", "applies_to": "fx_conversion_of_minor_units"}

T_DUP = "duplicate_invoice"
T_ALLOC = "missing_or_partial_allocation"
T_FX = "wrong_date_or_fx_rate"
T_CUT = "period_cutoff"
ERROR_TYPES = (T_DUP, T_ALLOC, T_FX, T_CUT)
SCEN_FX_PARTIAL = "partial_cross_currency_boundary"
SCEN_DUP_FX = "duplicate_with_fx_error"
SCENARIOS = (SCEN_FX_PARTIAL, SCEN_DUP_FX)
MAX_ISSUES = 12

PAYABLE_KINDS = {"regular", "subscription", "reissue"}
LEDGER_HEADER = ["row_id", "timestamp", "account", "debit_minor", "credit_minor",
                 "currency", "entity_id", "invoice_id", "payment_id"]

DEFAULT_CONFIG = {
    "period_start": "2026-01-01",
    "period_end": "2026-03-31",
    "n_entities": 10,
    "n_invoices": 70,        # обычные счета (без подписок, возвратов, спец-счетов)
    "n_subscribers": 3,      # контрагенты с ежемесячной подпиской (законные повторы сумм)
    "n_returns": 2,          # цепочки "счёт -> кредит-нота -> повторный счёт" (законно)
    "n_boundary_legit": 2,   # законные счета у самой границы периода
    "errors": {},            # {"duplicate_invoice": 2, ...} каждый тип настраивается отдельно
    "scenarios": {},         # {"partial_cross_currency_boundary": 1, ...}
}

# Профили сложности. Размеры укладываются в 200-2000 проводок, 20-200 документов, 0-12 ошибок.
PROFILES = {
    "clean": {"n_invoices": 60},
    "easy": {"n_invoices": 60, "errors": {T_DUP: 1, T_ALLOC: 1}},
    "medium": {"n_invoices": 65,
               "errors": {T_DUP: 1, T_ALLOC: 1, T_FX: 1, T_CUT: 1},
               "scenarios": {SCEN_DUP_FX: 1}},
    "hard": {"n_invoices": 65,
             "errors": {T_DUP: 2, T_ALLOC: 2, T_FX: 2, T_CUT: 2},
             "scenarios": {SCEN_FX_PARTIAL: 1, SCEN_DUP_FX: 1}},
}

# Открытые наборы сидов (закрытые сиды организаторов сюда не попадают никогда).
SPLIT_SEEDS = {"train": list(range(1000, 1010)), "dev": list(range(2000, 2010))}
SPLIT_PROFILES = ["clean", "easy", "medium", "hard", "clean",
                  "easy", "medium", "hard", "medium", "hard"]


# ---------------------------------------------------------------- вспомогательное

def fx_record_id(day: date, src: str, tgt: str) -> str:
    """ID записи курса (в fx.csv нет колонки id, поэтому он вычисляется по полям)."""
    return f"FX:{day.isoformat()}:{src}:{tgt}"


def convert_minor(amount: int, src: str, tgt: str, rate: Decimal) -> int:
    """Перевод суммы в минимальных единицах src -> tgt по курсу (tgt за 1 src)."""
    scale = Decimal(10) ** (CURRENCIES[tgt] - CURRENCIES[src])
    return int((Decimal(amount) * rate * scale).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def make_config(config: dict | None = None) -> dict:
    config = dict(config or {})
    profile = config.pop("profile", None)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if profile is not None:
        if profile not in PROFILES:
            raise ValueError(f"неизвестный профиль: {profile}")
        cfg.update(copy.deepcopy(PROFILES[profile]))
        cfg["profile"] = profile
    cfg.update(copy.deepcopy(config))
    for name, allowed in (("errors", ERROR_TYPES), ("scenarios", SCENARIOS)):
        unknown = set(cfg[name]) - set(allowed)
        if unknown:
            raise ValueError(f"неизвестные ключи в {name}: {sorted(unknown)}")
        if any((not isinstance(v, int)) or v < 0 for v in cfg[name].values()):
            raise ValueError(f"значения в {name} должны быть целыми >= 0")
    n_issues = sum(cfg["errors"].values()) + 2 * sum(cfg["scenarios"].values())
    if n_issues > MAX_ISSUES:
        raise ValueError(f"ошибок {n_issues}, максимум {MAX_ISSUES}")
    if cfg["n_entities"] < 2 or cfg["n_subscribers"] > cfg["n_entities"]:
        raise ValueError("n_entities >= 2 и n_subscribers <= n_entities")
    d0, d1 = date.fromisoformat(cfg["period_start"]), date.fromisoformat(cfg["period_end"])
    if (d1 - d0).days < 45:
        raise ValueError("период должен быть не короче 45 дней")
    return cfg


def _json_bytes(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _jsonl_bytes(rows: list) -> bytes:
    if not rows:
        return b""
    return ("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows)
            + "\n").encode("utf-8")


def _csv_bytes(header: list, rows: list) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _fmt_offset(hours: int) -> str:
    return f"{'+' if hours >= 0 else '-'}{abs(hours):02d}:00"


def _total(spec: dict) -> int:
    return sum(it["amount_minor"] for it in spec["items"])


def _sig(entity: str, items: list) -> tuple:
    return (entity, tuple((i["description"], i["quantity"], i["unit_price_minor"]) for i in items))


# ---------------------------------------------------------------- шаг 1: чистая история

def _random_money(rng: random.Random, currency: str) -> int:
    """Цена в минимальных единицах. Поддерживает валюты с 0 или 2 знаками."""
    lo, hi = PRICE_RANGE_MAJOR[currency]
    major = rng.randint(lo, hi)
    decimals = CURRENCIES[currency]
    if decimals == 0:
        return major
    return major * 10**decimals + rng.choice([0, 0, 0, 50, 99])


def _make_items(rng: random.Random, currency: str) -> list:
    items = []
    for _ in range(rng.randint(1, 3)):
        qty = rng.randint(1, 5)
        price = _random_money(rng, currency)
        items.append({"description": rng.choice(DESCRIPTIONS), "quantity": qty,
                      "unit_price_minor": price, "amount_minor": qty * price})
    return items


def _unique_items(rng, entity, seen):
    """Позиции, которых ещё не было у этого клиента (чтобы случайный счёт не оказался 'дублем')."""
    while True:
        items = _make_items(rng, entity["currency"])
        sig = _sig(entity["id"], items)
        if sig not in seen:
            seen.add(sig)
            return items


def _build_fx(rng, start_d: date, end_d: date) -> dict:
    """Курсы на каждый день для ВСЕХ пар валют: {(date, src, tgt): Decimal}."""
    first = start_d - timedelta(days=FX_MARGIN_DAYS)
    last = end_d + timedelta(days=FX_MARGIN_DAYS)
    usd_per = {c: Decimal(v) for c, v in START_RATES.items()}
    table = {}
    day = first
    while day <= last:
        for cur in sorted(usd_per):
            bp = rng.randint(-30, 30)  # базисные пункты, целые: никакого float
            usd_per[cur] = (usd_per[cur] * Decimal(10000 + bp) / Decimal(10000)).quantize(
                Q8, rounding=ROUND_HALF_EVEN)
        level = {"USD": Decimal(1), **usd_per}
        for src in sorted(CURRENCIES):
            for tgt in sorted(CURRENCIES):
                if src != tgt:
                    table[(day, src, tgt)] = (level[src] / level[tgt]).quantize(
                        Q8, rounding=ROUND_HALF_EVEN)
        day += timedelta(days=1)
    return table


def _make_entities(rng, cfg) -> list:
    entities = []
    currencies = sorted(CURRENCIES)
    for i in range(1, cfg["n_entities"] + 1):
        entities.append({"id": f"ENT-{i:03d}", "currency": rng.choice(currencies),
                         "subscription_price": None})
    for idx in rng.sample(list(range(cfg["n_entities"])), cfg["n_subscribers"]):
        ent = entities[idx]
        ent["subscription_price"] = _random_money(rng, ent["currency"])
    return entities


def _add_spec(specs, kind, entity, currency, issued, items, terms_days, **extra) -> dict:
    spec = {"key": len(specs), "kind": kind, "entity": entity, "currency": currency,
            "issued": issued, "posted": issued, "terms_days": terms_days,
            "items": items, "locked": False}
    spec.update(extra)
    specs.append(spec)
    return spec


def _new_plan(plans, kind, ts, currency, amount, entity, allocs, **extra) -> dict:
    plan = {"key": len(plans), "kind": kind, "timestamp": ts, "currency": currency,
            "amount": amount, "entity": entity, "allocs": allocs, "applied": None,
            "locked": False}
    plan.update(extra)
    plans.append(plan)
    return plan


def _make_regular(rng, cfg, entities, specs, seen, start_dt, end_dt):
    span = int((end_dt - start_dt).total_seconds())
    for _ in range(cfg["n_invoices"]):
        ent = rng.choice(entities)
        items = _unique_items(rng, ent, seen)
        issued = start_dt + timedelta(seconds=rng.randint(0, span))
        _add_spec(specs, "regular", ent["id"], ent["currency"], issued, items,
                  rng.choice([14, 30]))


def _make_subscriptions(entities, specs, start_dt, end_dt):
    """Ежемесячные счета на одну и ту же сумму: ЗАКОННО (в описании разный месяц)."""
    tz = start_dt.tzinfo
    for ent in entities:
        if ent["subscription_price"] is None:
            continue
        y, m = start_dt.year, start_dt.month
        while True:
            first = datetime(y, m, 1, 9, 0, 0, tzinfo=tz)
            if first > end_dt:
                break
            if first >= start_dt:
                price = ent["subscription_price"]
                _add_spec(specs, "subscription", ent["id"], ent["currency"], first,
                          [{"description": f"Monthly subscription {y}-{m:02d}", "quantity": 1,
                            "unit_price_minor": price, "amount_minor": price}], 14)
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _make_returns(rng, cfg, entities, specs, seen, start_dt, end_dt):
    """Счёт -> кредит-нота на всю сумму -> такой же повторный счёт. Выглядит как дубль, но
    законно: между ними есть кредит-нота."""
    span = int((end_dt - start_dt).total_seconds())
    for _ in range(cfg["n_returns"]):
        ent = rng.choice(entities)
        items = _unique_items(rng, ent, seen)
        issued_a = start_dt + timedelta(seconds=rng.randint(0, span // 2))
        a = _add_spec(specs, "returned_original", ent["id"], ent["currency"], issued_a, items, 30)
        cn_items = [{"description": it["description"], "quantity": -it["quantity"],
                     "unit_price_minor": it["unit_price_minor"],
                     "amount_minor": -it["amount_minor"]} for it in items]
        issued_cn = issued_a + timedelta(days=rng.randint(1, 3), seconds=rng.randint(0, 86399))
        _add_spec(specs, "credit_note", ent["id"], ent["currency"], issued_cn, cn_items, 0, ref=a)
        issued_c = issued_cn + timedelta(days=rng.randint(1, 5), seconds=rng.randint(0, 86399))
        _add_spec(specs, "reissue", ent["id"], ent["currency"], issued_c,
                  copy.deepcopy(items), 30)


def _make_boundary_legit(rng, cfg, entities, specs, seen, end_dt):
    """Законные счета в последние минуты периода и в первые минуты следующего."""
    for i in range(cfg["n_boundary_legit"]):
        ent = rng.choice(entities)
        items = _unique_items(rng, ent, seen)
        off = timedelta(seconds=rng.randint(60, 21600))
        if i % 2 == 0:
            _add_spec(specs, "boundary_in", ent["id"], ent["currency"], end_dt - off, items, 14)
        else:
            _add_spec(specs, "boundary_out", ent["id"], ent["currency"], end_dt + off, items, 14)


def _make_cutoff_candidates(rng, cfg, entities, specs, seen, end_dt):
    """Чистая история: истинная дата документа и дата проводки совпадают.
    Ошибку внесём позже: проводку сдвинем через границу периода."""
    for _ in range(cfg["errors"].get(T_CUT, 0)):
        variant = rng.choice(["early", "late"])
        ent = rng.choice(entities)
        items = _unique_items(rng, ent, seen)
        off = timedelta(seconds=rng.randint(3600, 172800))
        # early: документ датирован ПОСЛЕ конца периода; late: документ внутри периода
        issued = end_dt + off if variant == "early" else end_dt - off
        _add_spec(specs, "cutoff_candidate", ent["id"], ent["currency"], issued, items, 14,
                  variant=variant)


def _find_cross(rng, total: int, inv_ccy: str, ts: datetime, fx: dict):
    """Подбор кросс-валютного платежа, который ТОЧНО покрывает счёт.
    Нужна валюта платежа P, где 1 минимальная единица P весит меньше 1 минимальной единицы
    счёта (тогда каждая сумма в валюте счёта достижима). Возвращает (P, сумма_в_P, курс, дата)."""
    day = ts.date()
    options = []
    for pay_ccy in sorted(CURRENCIES):
        if pay_ccy == inv_ccy:
            continue
        rate = fx[(day, pay_ccy, inv_ccy)]
        step = rate * (Decimal(10) ** (CURRENCIES[inv_ccy] - CURRENCIES[pay_ccy]))
        if step < Decimal("0.95"):
            options.append((pay_ccy, rate, step))
    if not options:
        return None
    pay_ccy, rate, step = rng.choice(options)
    a0 = int((Decimal(total) / step).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
    for a in sorted(range(a0 - 3, a0 + 4), key=lambda x: abs(x - a0)):
        if a > 0 and convert_minor(a, pay_ccy, inv_ccy, rate) == total:
            return pay_ccy, a, rate, day
    return None


def _make_fx_candidates(rng, cfg, entities, specs, plans, seen, start_dt, end_dt, fx):
    """Счета с кросс-валютной оплатой, на которых потом сломаем курс.
    Роли: fx_only (ошибка сама по себе), fx_partial (сценарий у границы периода),
    dup_fx (сценарий с дублем)."""
    roles = (["fx_only"] * cfg["errors"].get(T_FX, 0)
             + ["fx_partial"] * cfg["scenarios"].get(SCEN_FX_PARTIAL, 0)
             + ["dup_fx"] * cfg["scenarios"].get(SCEN_DUP_FX, 0))
    span = int((end_dt - start_dt).total_seconds())
    for role in roles:
        boundary = role == "fx_partial"
        for _ in range(500):
            ent = rng.choice(entities)
            items = _make_items(rng, ent["currency"])
            if _sig(ent["id"], items) in seen:
                continue
            if boundary:
                pay_ts = end_dt - timedelta(seconds=rng.randint(60, 30 * 3600))
                issued = pay_ts - timedelta(days=rng.randint(2, 20), seconds=rng.randint(0, 86399))
            else:
                issued = start_dt + timedelta(seconds=rng.randint(0, span - 8 * 86400))
                pay_ts = issued + timedelta(days=rng.randint(2, 6), seconds=rng.randint(0, 86399))
            total = sum(i["amount_minor"] for i in items)
            found = _find_cross(rng, total, ent["currency"], pay_ts, fx)
            if found is None:
                continue
            seen.add(_sig(ent["id"], items))
            spec = _add_spec(specs, "fx_candidate", ent["id"], ent["currency"], issued, items,
                             30, role=role)
            pay_ccy, amount, rate, day = found
            _new_plan(plans, "cross", pay_ts, pay_ccy, amount, ent["id"],
                      [{"spec": spec, "alloc": total}], fx_rate=rate, fx_date=day)
            break
        else:
            raise RuntimeError("не удалось подобрать кросс-валютный платёж")


def _plan_payments(rng, specs, plans, end_dt, fx):
    """Оплаты обычных счетов: полная, кросс-валютная, разбитая на две. Все законные."""
    for spec in list(specs):
        if spec["kind"] not in PAYABLE_KINDS:
            continue
        pattern = rng.randint(1, 100)
        paid = rng.randint(1, 100) <= 85
        lo, hi = (1, 10) if spec["kind"] == "subscription" else (2, 30)
        ts = spec["issued"] + timedelta(days=rng.randint(lo, hi), seconds=rng.randint(0, 86399))
        if not paid or ts > end_dt:
            continue
        total, ccy = _total(spec), spec["currency"]
        if 70 < pattern <= 82:
            found = _find_cross(rng, total, ccy, ts, fx)
            if found is not None:
                pay_ccy, amount, rate, day = found
                _new_plan(plans, "cross", ts, pay_ccy, amount, spec["entity"],
                          [{"spec": spec, "alloc": total}], fx_rate=rate, fx_date=day)
                continue
        if 82 < pattern <= 92 and total >= 2:
            first = max(1, total * rng.randint(30, 70) // 100)
            _new_plan(plans, "split", ts, ccy, first, spec["entity"],
                      [{"spec": spec, "alloc": first}])
            ts2 = ts + timedelta(days=rng.randint(3, 20), seconds=rng.randint(0, 86399))
            if ts2 <= end_dt:  # иначе счёт законно остаётся частично оплаченным
                _new_plan(plans, "split", ts2, ccy, total - first, spec["entity"],
                          [{"spec": spec, "alloc": total - first}])
            continue
        _new_plan(plans, "single", ts, ccy, total, spec["entity"], [{"spec": spec, "alloc": total}])


def _merge_multi(rng, plans):
    """Один платёж закрывает два счёта одного клиента (законно)."""
    groups = {}
    for p in plans:
        if p["kind"] == "single":
            groups.setdefault((p["entity"], p["currency"]), []).append(p)
    removed, added = set(), []
    for gkey in sorted(groups):
        group = sorted(groups[gkey], key=lambda p: (p["timestamp"], p["key"]))
        i = 0
        while i + 1 < len(group):
            if rng.randint(1, 100) <= 30:
                a, b = group[i], group[i + 1]
                added.append({"key": a["key"], "kind": "multi",
                              "timestamp": max(a["timestamp"], b["timestamp"]),
                              "currency": a["currency"], "amount": a["amount"] + b["amount"],
                              "entity": a["entity"], "allocs": a["allocs"] + b["allocs"],
                              "applied": None, "locked": False})
                removed.update([a["key"], b["key"]])
                i += 2
            else:
                i += 1
    plans[:] = [p for p in plans if p["key"] not in removed] + added


# ---------------------------------------------------------------- шаг 2: ошибки

def _plans_by_spec(plans) -> dict:
    out = {}
    for p in plans:
        for a in p["allocs"]:
            out.setdefault(a["spec"]["key"], []).append(p)
    return out


def _make_duplicate(rng, specs, original) -> dict:
    """Копия счёта: те же клиент и позиции, выписана позже, новый номер, не оплачена.
    Вставляется ДО выдачи ID, иначе у дубля был бы самый большой номер (простой трюк)."""
    shift = rng.randint(60, 3 * 86400)
    return _add_spec(specs, "duplicate", original["entity"], original["currency"],
                     original["issued"] + timedelta(seconds=shift),
                     copy.deepcopy(original["items"]), original["terms_days"],
                     shift_seconds=shift, dup_of=original)


def _apply_fx_error(rng, plan, boundary, end_d, fx, scenario) -> dict:
    spec = plan["allocs"][0]["spec"]
    total = _total(spec)
    pay_ccy, inv_ccy, amount = plan["currency"], spec["currency"], plan["amount"]
    min_diff = max(1, total // 1000)  # расхождение должно быть заметным (>= 0.1%)
    variant = rng.choice(["wrong_date", "wrong_rate"])
    used_day = used_rate = wrong_alloc = None
    if variant == "wrong_date":
        if boundary:  # курс следующего периода
            days = [end_d + timedelta(days=k) for k in (1, 2, 3, 4)]
        else:
            deltas = [d for d in range(-7, 8) if abs(d) >= 3]
            rng.shuffle(deltas)
            days = [plan["fx_date"] + timedelta(days=d) for d in deltas]
        for day in days:
            rate = fx[(day, pay_ccy, inv_ccy)]
            alloc = convert_minor(amount, pay_ccy, inv_ccy, rate)
            if abs(alloc - total) >= min_diff:
                used_day, used_rate, wrong_alloc = day, rate, alloc
                break
        if wrong_alloc is None:
            variant = "wrong_rate"
    if variant == "wrong_rate":
        for pct in rng.sample([-5, -4, -3, 3, 4, 5], 6):
            rate = (plan["fx_rate"] * Decimal(100 + pct) / Decimal(100)).quantize(
                Q8, rounding=ROUND_HALF_EVEN)
            alloc = convert_minor(amount, pay_ccy, inv_ccy, rate)
            if abs(alloc - total) >= min_diff:
                used_rate, wrong_alloc = rate, alloc
                break
    if wrong_alloc is None:
        raise RuntimeError("не удалось внести ошибку курса")
    plan["allocs"][0]["alloc"] = wrong_alloc
    plan["locked"] = True
    return {"type": T_FX, "plan": plan, "spec": spec, "variant": variant, "scenario": scenario,
            "recorded_alloc": wrong_alloc, "used_day": used_day, "used_rate": used_rate}


def _apply_alloc_error(rng, plan, force_partial, scenario) -> dict:
    spec = plan["allocs"][0]["spec"]
    doc_alloc = sum(a["alloc"] for a in plan["allocs"])
    variant = "partial" if force_partial else rng.choice(["missing", "partial"])
    if variant == "missing":
        applied = 0
        plan["applied"] = []
    else:
        applied = max(1, doc_alloc * rng.randint(30, 80) // 100)
        if applied >= doc_alloc:
            applied = doc_alloc - 1
        plan["applied"] = [{"spec": spec, "applied": applied}]
    plan["locked"] = True
    return {"type": T_ALLOC, "plan": plan, "spec": spec, "variant": variant,
            "doc_alloc": doc_alloc, "applied": applied, "scenario": scenario}


def _inject(rng, cfg, specs, plans, end_dt, end_d, fx) -> list:
    records = []
    errors = cfg["errors"]
    by_spec = _plans_by_spec(plans)

    # 1) кросс-валютные кандидаты по ролям
    fx_specs = {"fx_only": [], "fx_partial": [], "dup_fx": []}
    for s in specs:
        if s["kind"] == "fx_candidate":
            fx_specs[s["role"]].append(s)
    for s in fx_specs["fx_only"]:
        records.append(_apply_fx_error(rng, by_spec[s["key"]][0], False, end_d, fx, None))
    for s in fx_specs["fx_partial"]:
        plan = by_spec[s["key"]][0]
        records.append(_apply_fx_error(rng, plan, True, end_d, fx, SCEN_FX_PARTIAL))
        records.append(_apply_alloc_error(rng, plan, True, SCEN_FX_PARTIAL))
    for s in fx_specs["dup_fx"]:
        records.append(_apply_fx_error(rng, by_spec[s["key"]][0], False, end_d, fx, SCEN_DUP_FX))
        dup = _make_duplicate(rng, specs, s)
        records.append({"type": T_DUP, "original": s, "duplicate": dup, "scenario": SCEN_DUP_FX})

    # 2) обычные дубли
    n = errors.get(T_DUP, 0)
    cands = [s for s in specs
             if s["kind"] == "regular" and not s["locked"]
             and s["issued"] + timedelta(days=4) <= end_dt
             and len(by_spec.get(s["key"], [])) == 1 and by_spec[s["key"]][0]["kind"] == "single"]
    if n > len(cands):
        raise ValueError(f"нужно {n} дублей, подходящих счетов {len(cands)}")
    for orig in rng.sample(cands, n):
        dup = _make_duplicate(rng, specs, orig)
        orig["locked"] = True
        by_spec[orig["key"]][0]["locked"] = True
        records.append({"type": T_DUP, "original": orig, "duplicate": dup, "scenario": None})

    # 3) пропущенное или частичное распределение платежа
    n = errors.get(T_ALLOC, 0)
    cands = [p for p in plans
             if p["kind"] == "single" and not p["locked"] and not p["allocs"][0]["spec"]["locked"]]
    if n > len(cands):
        raise ValueError(f"нужно {n} ошибок распределения, подходящих платежей {len(cands)}")
    for plan in rng.sample(cands, n):
        records.append(_apply_alloc_error(rng, plan, False, None))

    # 4) граница периода: сдвигаем ПРОВОДКУ через границу, документ остаётся
    for s in [x for x in specs if x["kind"] == "cutoff_candidate"]:
        off = timedelta(seconds=rng.randint(60, 21600))
        s["posted"] = end_dt - off if s["variant"] == "early" else end_dt + off
        records.append({"type": T_CUT, "spec": s, "variant": s["variant"], "scenario": None})
    return records


# ---------------------------------------------------------------- проводки

def _invoice_postings(spec) -> list:
    t = _total(spec)
    if t >= 0:
        return [(ACC_AR, t, 0), (ACC_REV, 0, t)]
    return [(ACC_AR, 0, -t), (ACC_REV, -t, 0)]  # кредит-нота: сторно


def _payment_postings(plan) -> list:
    """Проводки платежа. Дебет == кредит в КАЖДОЙ валюте (кросс-валютные через счёт 1500)."""
    pay_ccy, amount = plan["currency"], plan["amount"]
    allocs = plan["allocs"]
    applied = plan["applied"] if plan["applied"] is not None else [
        {"spec": a["spec"], "applied": a["alloc"]} for a in allocs]
    inv_ccy = allocs[0]["spec"]["currency"]
    doc_total = sum(a["alloc"] for a in allocs)
    rows = [(ACC_CASH, amount, 0, pay_ccy, None)]
    if inv_ccy != pay_ccy:
        rows.append((ACC_FXCLR, 0, amount, pay_ccy, None))
        rows.append((ACC_FXCLR, doc_total, 0, inv_ccy, None))
    for ap in applied:
        rows.append((ACC_AR, 0, ap["applied"], inv_ccy, ap["spec"]["invoice_id"]))
    rest = doc_total - sum(ap["applied"] for ap in applied)
    if rest > 0:
        rows.append((ACC_UNAPPL, 0, rest, inv_ccy, None))
    return rows


# ---------------------------------------------------------------- эталон

def _build_issues(records, rows_by_pay, rows_by_inv):
    items = []
    for rec in records:
        t, scen = rec["type"], rec.get("scenario")
        if t == T_DUP:
            o, d = rec["original"], rec["duplicate"]
            affected = sorted([o["invoice_id"], d["invoice_id"]])
            amount, cur = _total(d), d["currency"]
            # Достаточно двух документов: одинаковые клиент и позиции, разные номера.
            evidence = [sorted(["DOC-" + o["invoice_id"], "DOC-" + d["invoice_id"]])]
            prov = {"operation": "insert_copy_of_invoice", "source_invoice_id": o["invoice_id"],
                    "inserted_invoice_id": d["invoice_id"], "shift_seconds": d["shift_seconds"],
                    "inserted_ledger_row_ids": [r["row_id"] for r in rows_by_inv[d["invoice_id"]]]}
        elif t == T_ALLOC:
            plan, spec = rec["plan"], rec["spec"]
            pid, iid = plan["payment_id"], spec["invoice_id"]
            unapp = [r for r in rows_by_pay[pid] if r["account"] == ACC_UNAPPL]
            ar = [r for r in rows_by_pay[pid] if r["account"] == ACC_AR and r["invoice_id"] == iid]
            affected = sorted([pid, iid])
            amount, cur = rec["doc_alloc"] - rec["applied"], spec["currency"]
            evidence = [sorted(["DOC-" + pid, unapp[0]["row_id"]])]
            if ar:
                evidence.append(sorted(["DOC-" + pid, ar[0]["row_id"]]))
            prov = {"operation": "apply_payment_to_ledger_partially", "variant": rec["variant"],
                    "payment_id": pid, "invoice_id": iid,
                    "before": {"applied_minor": rec["doc_alloc"]},
                    "after": {"applied_minor": rec["applied"], "unapplied_minor": amount}}
        elif t == T_FX:
            plan, spec = rec["plan"], rec["spec"]
            pid, iid = plan["payment_id"], spec["invoice_id"]
            fxid = fx_record_id(plan["fx_date"], plan["currency"], spec["currency"])
            affected = sorted([pid, iid])
            amount, cur = abs(_total(spec) - rec["recorded_alloc"]), spec["currency"]
            evidence = [sorted(["DOC-" + pid, fxid])]
            if plan["applied"] is None:  # без частичного отнесения можно доказать и по проводкам
                cash = [r for r in rows_by_pay[pid] if r["account"] == ACC_CASH][0]
                ar = [r for r in rows_by_pay[pid] if r["account"] == ACC_AR][0]
                evidence.append(sorted([cash["row_id"], ar["row_id"], fxid]))
            prov = {"operation": "convert_payment_with_wrong_fx", "variant": rec["variant"],
                    "payment_id": pid, "invoice_id": iid,
                    "before": {"allocated_minor": _total(spec), "fx_date": plan["fx_date"].isoformat(),
                               "rate": str(plan["fx_rate"])},
                    "after": {"allocated_minor": rec["recorded_alloc"],
                              "fx_date": rec["used_day"].isoformat() if rec["used_day"] else None,
                              "rate": str(rec["used_rate"])}}
        else:  # T_CUT
            s = rec["spec"]
            iid = s["invoice_id"]
            affected = [iid]
            amount, cur = abs(_total(s)), s["currency"]
            rows = rows_by_inv[iid]
            rev = [r for r in rows if r["account"] == ACC_REV][0]
            ar = [r for r in rows if r["account"] == ACC_AR][0]
            evidence = [sorted(["DOC-" + iid, rev["row_id"]]), sorted(["DOC-" + iid, ar["row_id"]])]
            prov = {"operation": "shift_ledger_posting_across_period_boundary",
                    "variant": rec["variant"], "invoice_id": iid,
                    "before": {"posted_at": s["issued"].isoformat()},
                    "after": {"posted_at": s["posted"].isoformat()}}
        issue = {"type": t, "affected_ids": affected, "expected_amount_minor": amount,
                 "currency": cur, "amount_tolerance_minor": 0,
                 "acceptable_evidence_sets": sorted(evidence), "scenario": scen}
        prov.update({"type": t, "scenario": scen})
        items.append((issue, prov))
    items.sort(key=lambda ip: (ip[0]["affected_ids"], ip[0]["type"]))
    issues, injections = [], []
    for n, (issue, prov) in enumerate(items, start=1):
        issue["issue_id"] = prov["issue_id"] = f"ISS-{n:03d}"
        issues.append(issue)
        injections.append(prov)
    return issues, injections


# ---------------------------------------------------------------- сборка модели

def _build_model(seed: int, config: dict | None = None) -> dict:
    cfg = make_config(config)
    rng = random.Random(seed)  # ЕДИНСТВЕННЫЙ источник случайности
    tz_hours = rng.choice(TZ_OFFSETS_HOURS)
    tz = timezone(timedelta(hours=tz_hours))
    start_d, end_d = date.fromisoformat(cfg["period_start"]), date.fromisoformat(cfg["period_end"])
    start_dt = datetime(start_d.year, start_d.month, start_d.day, tzinfo=tz)
    end_dt = datetime(end_d.year, end_d.month, end_d.day, 23, 59, 59, tzinfo=tz)

    fx = _build_fx(rng, start_d, end_d)
    entities = _make_entities(rng, cfg)
    specs, plans, seen = [], [], set()

    # --- чистая история
    _make_regular(rng, cfg, entities, specs, seen, start_dt, end_dt)
    _make_subscriptions(entities, specs, start_dt, end_dt)
    _make_returns(rng, cfg, entities, specs, seen, start_dt, end_dt)
    _make_boundary_legit(rng, cfg, entities, specs, seen, end_dt)
    _make_cutoff_candidates(rng, cfg, entities, specs, seen, end_dt)
    _make_fx_candidates(rng, cfg, entities, specs, plans, seen, start_dt, end_dt, fx)
    _plan_payments(rng, specs, plans, end_dt, fx)
    _merge_multi(rng, plans)

    # --- ошибки
    records = _inject(rng, cfg, specs, plans, end_dt, end_d, fx)
    expected_issues = sum(cfg["errors"].values()) + 2 * sum(cfg["scenarios"].values())
    assert len(records) == expected_issues, (len(records), expected_issues)

    # --- ID выдаём по времени уже ПОСЛЕ внесения ошибок
    specs.sort(key=lambda s: (s["issued"], s["key"]))
    for n, s in enumerate(specs, start=1):
        s["invoice_id"] = f"INV-{n:06d}"
    for s in specs:
        if s["kind"] == "credit_note":
            for it in s["items"]:
                it["description"] = f"Credit note for {s['ref']['invoice_id']}: {it['description']}"
    plans.sort(key=lambda p: (p["timestamp"], p["key"]))
    for n, p in enumerate(plans, start=1):
        p["payment_id"] = f"PAY-{n:06d}"

    invoices = [{
        "doc_id": "DOC-" + s["invoice_id"], "invoice_id": s["invoice_id"],
        "entity_id": s["entity"], "issued_at": s["issued"].isoformat(),
        "due_at": (s["issued"] + timedelta(days=s["terms_days"])).isoformat(),
        "currency": s["currency"], "total_minor": _total(s), "line_items": s["items"],
    } for s in specs]
    payments = [{
        "doc_id": "DOC-" + p["payment_id"], "payment_id": p["payment_id"],
        "timestamp": p["timestamp"].isoformat(), "currency": p["currency"],
        "amount_minor": p["amount"],
        "invoice_refs": sorted(
            [{"invoice_id": a["spec"]["invoice_id"], "allocated_minor": a["alloc"]}
             for a in p["allocs"]], key=lambda r: r["invoice_id"]),
    } for p in plans]

    # --- проводки (двойная запись); у каждой строки своя дата проводки
    raw, seq = [], 0
    for s in specs:
        for acc, debit, credit in _invoice_postings(s):
            raw.append((s["posted"], seq, acc, debit, credit, s["currency"], s["entity"],
                        s["invoice_id"], None))
            seq += 1
    for p in plans:
        for acc, debit, credit, ccy, inv_id in _payment_postings(p):
            raw.append((p["timestamp"], seq, acc, debit, credit, ccy, p["entity"],
                        inv_id, p["payment_id"]))
            seq += 1
    raw.sort(key=lambda r: (r[0], r[1]))
    ledger, rows_by_pay, rows_by_inv = [], {}, {}
    for n, r in enumerate(raw, start=1):
        row = {"row_id": f"R{n:06d}", "timestamp": r[0].isoformat(), "account": r[2],
               "debit_minor": r[3], "credit_minor": r[4], "currency": r[5],
               "entity_id": r[6], "invoice_id": r[7], "payment_id": r[8]}
        ledger.append(row)
        if r[8] is not None:
            rows_by_pay.setdefault(r[8], []).append(row)
        else:
            rows_by_inv.setdefault(r[7], []).append(row)

    fx_rows = [[d.isoformat(), s, t, str(rate)] for (d, s, t), rate in sorted(fx.items())]
    issues, injections = _build_issues(records, rows_by_pay, rows_by_inv)

    episode_id = "EP-" + hashlib.sha256(
        json.dumps({"seed": seed, "config": cfg}, sort_keys=True).encode()
    ).hexdigest()[:12].upper()
    manifest = {
        "schema_version": SCHEMA_VERSION, "episode_id": episode_id,
        "base_currency": BASE_CURRENCY, "currencies": CURRENCIES,
        "period_start": cfg["period_start"], "period_end": cfg["period_end"],
        "timezone": _fmt_offset(tz_hours),
        "counts": {"ledger_rows": len(ledger), "invoices": len(invoices),
                   "payments": len(payments), "fx_rows": len(fx_rows)},
        "rounding": ROUNDING,
    }
    policy = {
        "schema_version": SCHEMA_VERSION,
        "fx_rate_date": {"invoice": "issue_date", "payment": "payment_date"},
        "fx_rate_direction": "units of target_currency per 1 unit of source_currency",
        "fx_record_id": "FX:<date>:<source_currency>:<target_currency>",
        "allocation": ("allocated_minor is in the invoice currency; for a payment in another "
                       "currency it equals convert(amount_minor, payment->invoice currency, "
                       "rate of the payment's local date) with the rounding below"),
        "ledger_application": ("payment documents are applied to invoices in the ledger: account "
                               "1200 credit per invoice; any part not applied sits on account 2100"),
        "period_boundary": ("period_start..period_end inclusive, in the manifest timezone; "
                            "records dated after period_end are the next period"),
        "rounding": ROUNDING, "accounts": ACCOUNTS,
    }
    truth = {"schema_version": SCHEMA_VERSION, "episode_id": episode_id, "issues": issues}
    provenance = {"schema_version": SCHEMA_VERSION, "episode_id": episode_id,
                  "generator_version": GENERATOR_VERSION, "seed": seed, "config": cfg,
                  "injections": injections}
    return {"manifest": manifest, "policy": policy, "ledger": ledger, "invoices": invoices,
            "payments": payments, "fx": fx_rows, "truth": truth, "provenance": provenance}


def _serialize(model: dict) -> dict:
    cell = lambda v: "null" if v is None else str(v)  # noqa: E731 - отсутствующая ссылка = явный null
    return {
        "episode_id": model["manifest"]["episode_id"],
        "public": {
            "manifest.json": _json_bytes(model["manifest"]),
            "ledger.csv": _csv_bytes(LEDGER_HEADER,
                                     [[cell(r[h]) for h in LEDGER_HEADER] for r in model["ledger"]]),
            "invoices.jsonl": _jsonl_bytes(model["invoices"]),
            "payments.jsonl": _jsonl_bytes(model["payments"]),
            "fx.csv": _csv_bytes(["date", "source_currency", "target_currency", "rate"],
                                 model["fx"]),
            "policy.json": _json_bytes(model["policy"]),
        },
        "private": {"truth.json": _json_bytes(model["truth"]),
                    "provenance.json": _json_bytes(model["provenance"])},
    }


def build_episode(seed: int, config: dict | None = None) -> dict:
    """Чистая функция: ничего не пишет на диск, возвращает содержимое файлов в байтах."""
    return _serialize(_build_model(seed, config))


# ---------------------------------------------------------------- безвредные преобразования

_ID_RE = re.compile(r"(INV-\d{6}|PAY-\d{6}|ENT-\d{3}|\bR\d{6}\b)")


def build_paired_episode(seed: int, config: dict | None, permute_seed: int) -> dict:
    """Тот же эпизод, но с ДРУГИМИ ID и порядком записей (парная проверка воспроизводимости).
    Ответ oracle и награда обязаны сохраниться с учётом новых ID."""
    model = _build_model(seed, config)
    rng = random.Random(permute_seed)
    mapping = {}

    def permute(prefix, width, old_ids):
        numbers = list(range(1, len(old_ids) + 1))
        rng.shuffle(numbers)
        for old, num in zip(sorted(old_ids), numbers):
            mapping[old] = f"{prefix}{num:0{width}d}"

    permute("INV-", 6, [i["invoice_id"] for i in model["invoices"]])
    permute("PAY-", 6, [p["payment_id"] for p in model["payments"]])
    permute("R", 6, [r["row_id"] for r in model["ledger"]])
    permute("ENT-", 3, sorted({i["entity_id"] for i in model["invoices"]}))
    old_ep = model["manifest"]["episode_id"]
    new_ep = "EP-" + hashlib.sha256(f"{old_ep}:{permute_seed}".encode()).hexdigest()[:12].upper()

    def walk(obj):
        if isinstance(obj, str):
            if obj == old_ep:
                return new_ep
            return _ID_RE.sub(lambda m: mapping.get(m.group(0), m.group(0)), obj)
        if isinstance(obj, list):
            return [walk(x) for x in obj]
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        return obj

    model = walk(model)
    for issue in model["truth"]["issues"]:
        issue["affected_ids"] = sorted(issue["affected_ids"])
        issue["acceptable_evidence_sets"] = sorted(sorted(s) for s in issue["acceptable_evidence_sets"])
    model["provenance"]["relabel"] = {"permute_seed": permute_seed}
    for part in ("ledger", "invoices", "payments"):
        rng.shuffle(model[part])
    return _serialize(model)


# ---------------------------------------------------------------- запись на диск и CLI

def write_episode(episode: dict, out_dir) -> Path:
    """Открытые файлы -> out_dir/public, закрытые -> out_dir/private."""
    out = Path(out_dir)
    for part in ("public", "private"):
        (out / part).mkdir(parents=True, exist_ok=True)
        for name, data in episode[part].items():
            (out / part / name).write_bytes(data)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Сгенерировать эпизод или открытый набор")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="clean")
    parser.add_argument("--permute-seed", type=int, help="парный эпизод: другие ID и порядок")
    parser.add_argument("--split", choices=sorted(SPLIT_SEEDS),
                        help="сгенерировать все 10 открытых эпизодов train или dev")
    args = parser.parse_args()
    if args.split:
        for seed, profile in zip(SPLIT_SEEDS[args.split], SPLIT_PROFILES):
            ep = build_episode(seed, {"profile": profile})
            write_episode(ep, Path(args.out) / args.split / f"ep_{seed}_{profile}")
            print(f"{args.split} seed={seed} profile={profile} {ep['episode_id']}")
        return
    if args.seed is None:
        parser.error("нужен --seed или --split")
    cfg = {"profile": args.profile}
    ep = (build_paired_episode(args.seed, cfg, args.permute_seed)
          if args.permute_seed is not None else build_episode(args.seed, cfg))
    write_episode(ep, args.out)
    print(f"{ep['episode_id']} -> {args.out}")


if __name__ == "__main__":
    main()
