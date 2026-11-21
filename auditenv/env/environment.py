"""Агентская сторона среды: reset / step.

Среда знает только открытые данные. Ни truth, ни оценщика у неё нет:
на submit она лишь запоминает ответ и завершает эпизод, а награду
считает хост (env/harness.py).

Правила (см. docs/env_notes.md):
- каждый вызов step() тратит один шаг, в том числе вызов с ошибкой;
- лимит 100 действий, включая submit; если на 100-м действии не submit,
  эпизод усечён (truncated);
- лимит времени 60 с на эпизод (с момента reset): при превышении
  truncated=True, ответ считается пустым;
- inspection_steps = число всех вызовов step до submit, в том числе ошибочных
  (submit входит в лимит 100, но не в inspection_steps).
"""
import copy
import json
import time

from .calculator import CalcError, calculate
from .loader import load_open_company, record_id

MAX_STEPS = 100
TIME_LIMIT_S = 60.0
PAGE_DEFAULT = 20
PAGE_MAX = 50
MAX_SUBMISSION_CHARS = 1_000_000
READ_ACTIONS = ("list_records", "get_record", "get_document")
KNOWN_ACTIONS = READ_ACTIONS + ("calculator", "submit")


def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


class _Episode:
    """Всё состояние одного эпизода. reset создаёт новый объект целиком."""

    def __init__(self, company: dict, seed, max_steps: int, time_limit_s: float, started: float):
        self.company = company
        self.seed = seed
        self.max_steps = max_steps
        self.time_limit_s = time_limit_s
        self.started = started
        self.steps = 0
        self.inspection_steps = 0
        self.done = False
        self.timed_out = False
        self.submission = None


class AuditEnv:
    def __init__(self, loader=load_open_company, clock=time.monotonic):
        self._loader = loader
        self._clock = clock
        self._ep = None

    # ---------- публичное API ----------
    def reset(self, seed=None, config=None):
        config = dict(config or {})
        open_dir = config.get("open_dir")
        if open_dir is None:
            raise ValueError("config['open_dir'] обязателен")
        max_steps = int(config.get("max_steps", MAX_STEPS))
        time_limit_s = float(config.get("time_limit_s", TIME_LIMIT_S))
        # новый объект: ничего не переносится из прошлого эпизода
        self._ep = _Episode(self._loader(open_dir, typed=True), seed, max_steps, time_limit_s, self._clock())
        company = self._ep.company
        obs = {
            "type": "reset",
            "manifest": copy.deepcopy(company["manifest"]),
            "policy": copy.deepcopy(company["policy"]),
            "tables": {n: len(r) for n, r in company["tables"].items()},
            "steps_left": max_steps,
        }
        return obs, self._info()

    def step(self, action):
        ep = self._require_episode()
        if ep.done:
            raise RuntimeError("Эпизод завершён, вызовите reset()")

        ep.steps += 1

        if self.time_left() <= 0:
            return self._timeout_result()

        name = action.get("action") if isinstance(action, dict) else None
        if name != "submit":
            ep.inspection_steps += 1  # все шаги до submit, включая ошибочные и калькулятор

        obs = self._dispatch(name, action)

        terminated = ep.submission is not None
        truncated = (not terminated) and ep.steps >= ep.max_steps
        ep.done = terminated or truncated
        return obs, 0.0, terminated, truncated, self._info()

    def force_truncate(self, reason: str = "TIMEOUT"):
        """Хост завершает эпизод (агент завис или упал): truncated, ответ пустой."""
        ep = self._require_episode()
        if ep.done:
            raise RuntimeError("Эпизод завершён, вызовите reset()")
        return self._timeout_result(reason)

    def time_left(self) -> float:
        ep = self._require_episode()
        return ep.time_limit_s - (self._clock() - ep.started)

    @property
    def submission(self):
        return None if self._ep is None else self._ep.submission

    @property
    def inspection_steps(self) -> int:
        return 0 if self._ep is None else self._ep.inspection_steps

    @property
    def timed_out(self) -> bool:
        return bool(self._ep and self._ep.timed_out)

    @property
    def elapsed_s(self) -> float:
        ep = self._require_episode()
        return self._clock() - ep.started

    # ---------- внутреннее ----------
    def _require_episode(self) -> _Episode:
        if self._ep is None:
            raise RuntimeError("Сначала вызовите reset()")
        return self._ep

    def _timeout_result(self, reason: str = "TIMEOUT"):
        ep = self._ep
        ep.timed_out = reason == "TIMEOUT"
        ep.submission = None  # по таймауту или сбою ответ пустой
        ep.done = True
        message = "Время эпизода вышло" if reason == "TIMEOUT" else "Сбой агента"
        obs = _error(reason, message)
        return obs, 0.0, False, True, self._info()

    def _info(self) -> dict:
        ep = self._ep
        # только безопасное: номер шага и остаток бюджета
        return {"step": ep.steps, "steps_left": max(0, ep.max_steps - ep.steps)}

    def _dispatch(self, name, action):
        if not isinstance(action, dict) or name is None:
            return _error("INVALID_ACTION", "Действие должно быть объектом с полем action")
        if not isinstance(name, str) or name not in KNOWN_ACTIONS:
            return _error("UNKNOWN_ACTION", "Неизвестное действие")
        if name == "list_records":
            return self._list_records(action)
        if name == "get_record":
            return self._get_record(action)
        if name == "get_document":
            return self._get_document(action)
        if name == "submit":
            return self._submit(action)
        return self._calculator(action)

    def _table(self, action):
        table = action.get("table")
        records = self._ep.company["tables"].get(table) if isinstance(table, str) else None
        return table, records

    def _list_records(self, action):
        table, records = self._table(action)
        if records is None:
            return _error("UNKNOWN_TABLE", "Такой таблицы нет")
        limit = action.get("limit", PAGE_DEFAULT)
        if limit is None:
            limit = PAGE_DEFAULT
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            return _error("INVALID_ARGUMENT", "limit должен быть целым числом больше 0")
        limit = min(limit, PAGE_MAX)  # страница не больше 50 записей
        cursor = action.get("cursor")
        offset = 0
        if cursor is not None:
            if not (isinstance(cursor, str) and cursor.startswith("c") and 2 <= len(cursor) <= 9
                    and cursor[1:].isascii() and cursor[1:].isdecimal()):
                return _error("INVALID_CURSOR", "Некорректный курсор")
            offset = int(cursor[1:])
            if offset > len(records):
                return _error("INVALID_CURSOR", "Некорректный курсор")
        page = records[offset:offset + limit]
        nxt = offset + limit
        return {
            "table": table,
            "records": copy.deepcopy(page),
            "next_cursor": f"c{nxt}" if nxt < len(records) else None,
            "total": len(records),
        }

    def _get_record(self, action):
        table, records = self._table(action)
        if records is None:
            return _error("UNKNOWN_TABLE", "Такой таблицы нет")
        rid = action.get("id")
        for rec in records:
            if record_id(table, rec) == rid:
                return {"table": table, "record": copy.deepcopy(rec)}
        return _error("UNKNOWN_ID", "Записи с таким id нет")

    def _get_document(self, action):
        doc_id = action.get("doc_id")
        doc = self._ep.company["documents"].get(doc_id) if isinstance(doc_id, str) else None
        if doc is None:
            return _error("UNKNOWN_DOCUMENT", "Документа с таким id нет")
        return {"doc_id": doc_id, "document": copy.deepcopy(doc)}

    def _calculator(self, action):
        try:
            return {"result": calculate(action.get("expression"))}
        except CalcError as e:
            return _error(e.code, e.message)

    def _submit(self, action):
        if "findings" not in action:
            return _error("INVALID_ARGUMENT", "Нужно поле findings")
        try:
            # храним чистый JSON-вид: без циклов, NaN и посторонних объектов
            text = json.dumps(action["findings"], allow_nan=False)
            if len(text) > MAX_SUBMISSION_CHARS:
                return _error("INVALID_ARGUMENT", "Ответ слишком большой")
            self._ep.submission = json.loads(text)
        except (TypeError, ValueError, RecursionError):
            return _error("INVALID_ARGUMENT", "Ответ должен быть корректным JSON")
        return {"submitted": True}
