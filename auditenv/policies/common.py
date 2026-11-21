"""Общее для политик. Политика: объект с методом act(observation) -> action.

Политика создаётся заново на каждый эпизод и получает на вход последнее
наблюдение среды (первое это наблюдение reset).
"""

DUPLICATE_TYPE = "duplicate_invoice"  # точное имя типа из определений жюри
MAX_FINDINGS = 30
MAX_IDS = 10


def list_action(table, cursor=None, limit=50):
    action = {"action": "list_records", "table": table, "limit": limit}
    if cursor:
        action["cursor"] = cursor
    return action


def submit(findings):
    return {"action": "submit", "findings": findings[:MAX_FINDINGS]}


def invoice_amount(inv):
    """Сумма счёта: total_minor (формат A) или amount_minor (мини-эпизод)."""
    return inv["total_minor"] if "total_minor" in inv else inv["amount_minor"]


def invoice_ids(*invoices):
    """Канонические affected_ids: только ID счетов."""
    return {inv["invoice_id"] for inv in invoices}


def doc_ids(*invoices):
    """Доказательства: документы этих счетов (если у счёта нет doc_id, берётся сам счёт)."""
    return {inv.get("doc_id") or inv["invoice_id"] for inv in invoices}


def finding(invoices, amount_minor, currency, text):
    """Вывод «дубликат счёта» по списку счетов: affected = их ID, evidence = их документы."""
    return {
        "type": DUPLICATE_TYPE,
        "affected_ids": sorted(invoice_ids(*invoices))[:MAX_IDS],
        "evidence_ids": sorted(doc_ids(*invoices))[:MAX_IDS],
        "claimed_amount_minor": int(amount_minor),
        "currency": currency,
        "explanation": text[:1000],
    }


class InvoiceLister:
    """Помощник: постранично собирает все счета. feed(obs) -> следующее действие или None, когда всё прочитано."""

    def __init__(self):
        self.invoices = []
        self.cursor = None
        self.failed = False
        self._started = False

    def next_action(self, obs):
        if self._started:
            if "error" in obs:
                self.failed = True
                return None
            if obs.get("table") == "invoices" and "records" in obs:
                self.invoices += obs["records"]
                self.cursor = obs.get("next_cursor")
                if self.cursor is None:
                    return None
        self._started = True
        return list_action("invoices", self.cursor)
