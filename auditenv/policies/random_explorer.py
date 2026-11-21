"""Случайный исследователь с фиксированным seed.

Делает n_reads случайных чтений (таблицы, записи, документы, которые уже
видел), затем сдаёт один случайный вывод из двух увиденных счетов либо
пустой ответ. Свой random.Random(seed): два эпизода подряд не мешают друг другу.
"""
import random

from .common import finding, invoice_amount, invoice_ids, list_action, submit

TABLES = ("ledger", "invoices", "payments", "fx")


class RandomExplorer:
    def __init__(self, seed=0, n_reads=15, **_):
        self._rng = random.Random(seed)
        self._left = int(n_reads)
        self._invoices = []
        self._doc_ids = []
        self._records = []  # (table, id)

    def act(self, obs):
        self._learn(obs)
        if self._left <= 0:
            return self._answer()
        self._left -= 1
        kind = self._rng.choice(("list", "record", "document"))
        if kind == "record" and self._records:
            table, rid = self._rng.choice(self._records)
            return {"action": "get_record", "table": table, "id": rid}
        if kind == "document" and self._doc_ids:
            return {"action": "get_document", "doc_id": self._rng.choice(self._doc_ids)}
        return list_action(self._rng.choice(TABLES), limit=self._rng.choice((5, 10, 20)))

    def _learn(self, obs):
        for rec in obs.get("records", []) or []:
            table = obs.get("table")
            for key in ("row_id", "invoice_id", "payment_id"):
                if key in rec:
                    self._records.append((table, rec[key]))
                    break
            if table == "invoices":
                self._invoices.append(rec)
            if rec.get("doc_id"):
                self._doc_ids.append(rec["doc_id"])
        if "record" in obs and obs["record"].get("doc_id"):
            self._doc_ids.append(obs["record"]["doc_id"])

    def _answer(self):
        unique = {i["invoice_id"]: i for i in self._invoices}
        ids = sorted(unique)
        if len(ids) < 2:
            return submit([])
        a, b = self._rng.sample(ids, 2)
        inv = unique[a]
        return submit([finding([unique[a], unique[b]], invoice_amount(inv),
                               inv["currency"], "Случайная догадка.")])
