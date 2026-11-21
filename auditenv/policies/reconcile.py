"""Простая сверка: читает все счета и помечает группы одинаковых
(контрагент, сумма, валюта). Не смотрит ни платежи, ни кредит-ноты, ни
документы, поэтому на «отвлекающих» случаях (подписки, возвраты) должна
ошибаться."""
from .common import InvoiceLister, finding, invoice_amount, invoice_ids, submit


class ReconcilePolicy:
    def __init__(self, **_):
        self._lister = InvoiceLister()

    def act(self, obs):
        action = self._lister.next_action(obs)
        if action is not None:
            return action
        if self._lister.failed:
            return submit([])
        groups = {}
        for inv in self._lister.invoices:
            key = (inv["entity_id"], invoice_amount(inv), inv["currency"])
            groups.setdefault(key, []).append(inv)
        findings = []
        for key in sorted(groups):
            group = sorted(groups[key], key=lambda i: i["invoice_id"])
            if len(group) > 1:
                entity, amount, currency = key
                names = ", ".join(i["invoice_id"] for i in group)
                findings.append(finding(
                    group[:5], amount, currency,
                    f"Счета {names} совпадают по контрагенту {entity} и сумме."))
        return submit(findings)
