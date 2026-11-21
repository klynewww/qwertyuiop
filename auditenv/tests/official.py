"""Матрица закрытых эпизодов D01-D20 (по определениям жюри) и безопасные преобразования для пар P01-P20.

Здесь свои открытые сиды 9000+: закрытых сидов организаторов в репозитории нет и быть не может.
"""
import csv
import io
import json
import random
import re
import shutil
from pathlib import Path

import judge_adapter as ja

DUP, ALLOC, FX, CUT = "duplicate_invoice", "missing_or_partial_allocation", "wrong_date_or_fx_rate", "period_cutoff"
S1, S2 = "partial_cross_currency_boundary", "duplicate_with_fx_error"


def _cfg(errors=None, scenarios=None, **extra):
    c = {"n_invoices": 62, "errors": errors or {}, "scenarios": scenarios or {}}
    c.update(extra)
    return c


# (id, seed, config, смысл)
MATRIX = (
    [("D01", 9001, _cfg(), "обычная сверка"),
     ("D02", 9002, _cfg(n_invoices=70), "платёж разбит корректно"),
     ("D03", 9003, _cfg(n_returns=4), "корректный возврат"),
     ("D04", 9004, _cfg(n_subscribers=5), "законные повторяющиеся суммы"),
     ("D05", 9005, _cfg(n_boundary_legit=4), "корректная граница валюты/периода")]
    + [(f"D{6 + i:02d}", 9006 + i, _cfg({DUP: 1 + (i % 2)}), "дубль") for i in range(3)]
    + [(f"D{9 + i:02d}", 9009 + i, _cfg({ALLOC: 1}), "распределение") for i in range(3)]
    + [(f"D{12 + i:02d}", 9012 + i, _cfg({FX: 1 + i}), "курс/дата") for i in range(2)]
    + [(f"D{14 + i:02d}", 9014 + i, _cfg({CUT: 1 + i}), "период") for i in range(2)]
    + [(f"D{16 + i:02d}", 9016 + i, _cfg(scenarios={S1: 1}, n_invoices=62 + i), "сценарий 1") for i in range(3)]
    + [(f"D{19 + i:02d}", 9019 + i, _cfg(scenarios={S2: 1}, n_invoices=62 + i), "сценарий 2") for i in range(2)]
)


def build_all(base: Path) -> dict:
    out = {}
    for eid, seed, cfg, _ in MATRIX:
        d = base / eid
        ja.generate_episode(seed, cfg, str(d))
        out[eid] = d
    return out


# ---------- безопасные преобразования жюри (на копии эпизода) ----------
_ID = re.compile(r"(INV-\d{6}|PAY-\d{6}|ENT-\d{3}|\bR\d{6}\b)")
_OPEN_TABLES = ("ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv")


def _id_map(src: Path, rng: random.Random) -> dict:
    text = "".join((src / n).read_text(encoding="utf-8") for n in
                   ("ledger.csv", "invoices.jsonl", "payments.jsonl"))
    found = {}
    for m in _ID.findall(text):
        key = "R" if m.startswith("R") else m.split("-")[0]
        found.setdefault(key, set()).add(m)
    mapping = {}
    for key, ids in found.items():
        olds = sorted(ids)
        width = len(olds[0]) - (len(key) + (1 if key != "R" else 0))
        nums = list(range(1, len(olds) + 1))
        rng.shuffle(nums)
        pre = key + ("-" if key != "R" else "")
        for old, n in zip(olds, nums):
            mapping[old] = f"{pre}{n:0{width}d}"
    assert len(set(mapping.values())) == len(mapping)
    return mapping


def transform(src: Path, dst: Path, *, shuffle: bool, remap: bool, seed: int) -> dict:
    """Копия эпизода: переставляет строки таблиц и/или меняет ID согласованно во всех файлах
    (включая закрытые truth и provenance). Исходный каталог не трогается. Возвращает отображение."""
    rng = random.Random(seed)
    shutil.copytree(src, dst)
    mapping = _id_map(src, rng) if remap else {}
    sub = (lambda t: _ID.sub(lambda m: mapping.get(m.group(0), m.group(0)), t)) if remap else (lambda t: t)
    for p in dst.iterdir():
        if p.is_file():
            p.write_text(sub(p.read_text(encoding="utf-8")), encoding="utf-8")
    if shuffle:
        for n in _OPEN_TABLES:
            p = dst / n
            lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
            head, body = (lines[:1], lines[1:]) if n.endswith(".csv") else ([], lines)
            rng.shuffle(body)
            p.write_text("".join(head + body), encoding="utf-8")
    return mapping


def unmap_findings(findings, mapping):
    inv = {v: k for k, v in mapping.items()}
    out = json.loads(json.dumps(findings))
    for f in out["findings"]:
        f["affected_ids"] = sorted(inv.get(x, x) for x in f["affected_ids"])
        f["evidence_ids"] = sorted(_ID.sub(lambda m: inv.get(m.group(0), m.group(0)), x) for x in f["evidence_ids"])
    return out
