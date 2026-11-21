"""Мост между нашими политиками и протоколом жюри.

Наши политики говорят на внутреннем формате ({"action": ...}, наблюдения с table/records).
Жюри говорит конвертами {"name", "args"} и наблюдениями {"result", "error", "remaining_steps"}.
Shim переводит туда и обратно; решений за политику не принимает.
Политика не импортирует оценщик и не знает путей эпизода.
"""

TABLE_COUNT_KEYS = {"ledger": "ledger_rows", "invoices": "invoices", "payments": "payments", "fx": "fx_rows"}


class EnvelopeShim:
    def __init__(self, policy):
        self.policy = policy
        self._last = None

    def act(self, obs):
        legacy = self._to_legacy(obs)
        action = self.policy.act(legacy)
        self._last = action
        name = action["action"]
        args = {k: v for k, v in action.items() if k != "action"}
        if name == "submit":
            args = {"findings": {"findings": list(args.get("findings", []))}}
        return {"name": name, "args": args}

    def _to_legacy(self, obs):
        if "task" in obs:  # первое наблюдение
            counts = obs["manifest"].get("counts", {})
            return {"type": "reset", "manifest": obs["manifest"], "policy": obs["policy"],
                    "tables": {t: counts.get(k, 0) for t, k in TABLE_COUNT_KEYS.items()},
                    "steps_left": obs["remaining_steps"]}
        if obs.get("error"):
            return {"error": obs["error"]}
        name, res = self._last["action"], obs["result"]
        if name == "list_records":
            return {"table": self._last.get("table"), "records": res["records"],
                    "next_cursor": res["next_cursor"]}
        if name == "get_record":
            return {"table": self._last.get("table"), "record": res}
        if name == "get_document":
            return {"doc_id": self._last.get("doc_id"), "document": res}
        if name == "calculator":
            return {"result": res}
        return {"submitted": True}
