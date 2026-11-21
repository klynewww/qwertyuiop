"""«Дотошный аудитор»: рабочая политика на правилах (policies/audit_rules.py).

Читает счета, платежи и проводки, для кросс-валютных платежей запрашивает
нужные курсы по одному (get_record по ID вида FX:<дата>:<из>:<в>), применяет
правила и сдаёт ответ. Нужна затем, чтобы показать: задачу можно решить
только аккуратной работой с данными, и чтобы получить «потолок» для сравнения
с простыми политиками. Модель с обучением должна приближаться к ней.

CarefulDupOnly ищет только дубли: средняя ступень лестницы сложности.
"""
from decimal import Decimal

from . import audit_rules as rules
from .common import list_action, submit


class _Abort(Exception):
    pass


class CarefulPolicy:
    TYPES = rules.ALL_TYPES

    def __init__(self, **_):
        self._gen = None

    def act(self, obs):
        if self._gen is None:
            self._gen = self._run(obs)
            return next(self._gen)
        try:
            return self._gen.send(obs)
        except StopIteration:
            return submit([])

    # ---- сценарий работы: генератор, который отдаёт действия и получает наблюдения ----
    def _run(self, reset_obs):
        try:
            invoices = yield from self._read_all("invoices")
            payments = yield from self._read_all("payments")
            ledger = []
            if {rules.T_ALLOC, rules.T_CUT} & set(self.TYPES):
                ledger = yield from self._read_all("ledger")
            rates = {}
            if rules.T_FX in self.TYPES:
                for fid in rules.needed_fx_ids(invoices, payments):
                    obs = yield {"action": "get_record", "table": "fx", "id": fid}
                    if "record" in obs:
                        rates[fid] = Decimal(obs["record"]["rate"])
        except _Abort:
            yield submit([])
            return

        def fx_rate(day, src, tgt):
            return rates.get(rules.fx_id(day, src, tgt))

        issues = rules.find_issues(reset_obs["manifest"], invoices, payments, ledger, fx_rate, self.TYPES)
        yield submit([rules.to_finding(i) for i in issues])

    def _read_all(self, table):
        rows, cursor = [], None
        while True:
            obs = yield list_action(table, cursor)
            if "error" in obs:
                raise _Abort
            rows += obs["records"]
            cursor = obs["next_cursor"]
            if cursor is None:
                return rows


class CarefulDupOnly(CarefulPolicy):
    TYPES = (rules.T_DUP,)
