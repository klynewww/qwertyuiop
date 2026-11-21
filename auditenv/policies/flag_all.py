"""Атака «пометить всё»: читает все счета и заявляет дубль для пар
счетов (сначала пары одного контрагента), пока не упрётся в лимит
30 выводов. Проверяет, что награда не даёт выиграть шумом: много
ложных выводов роняют точность."""
from itertools import combinations

from .common import InvoiceLister, finding, invoice_amount, invoice_ids, submit


class FlagAllPolicy:
    def __init__(self, **_):
        self._lister = InvoiceLister()

    def act(self, obs):
        action = self._lister.next_action(obs)
        if action is not None:
            return action
        if self._lister.failed:
            return submit([])
        invoices = sorted(self._lister.invoices, key=lambda i: i["invoice_id"])
        pairs = [(a, b) for a, b in combinations(invoices, 2) if a["currency"] == b["currency"]]
        pairs.sort(key=lambda p: (p[0]["entity_id"] != p[1]["entity_id"], p[0]["invoice_id"], p[1]["invoice_id"]))
        findings = [finding([a, b], invoice_amount(a), a["currency"], "Пометка без проверки.")
                    for a, b in pairs]
        return submit(findings)
