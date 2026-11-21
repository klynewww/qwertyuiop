"""Фаззинг среды мусорными действиями: ни исключений, ни зависаний, ни утечек."""
import json
import random

import pytest

from env.environment import AuditEnv, MAX_STEPS
from tests.helpers import MINI

OPEN = str(MINI / "open")
ACTIONS = ["list_records", "get_record", "get_document", "calculator", "submit",
           "LIST_RECORDS", " submit", "", None, 5, ["submit"], "__import__", "step", "reset"]
TABLES = ["ledger", "invoices", "payments", "fx", "hidden", "truth", "../hidden", "", None, 3, ["fx"], {"a": 1}]
IDS = ["INV-0001", "PAY-0001", "DOC-0001", "L-0001", "inv-0001", "", None, 7, 1.5, [], {}, "x" * 10000,
       "\x00", "' OR 1=1 --", "../hidden/truth.json", "FX:2026-03-01:EUR:USD"]
CURSORS = [None, "c0", "c1", "c5", "c6", "c-1", "c1e3", "", "x", 5, "c" + "9" * 50, "c²", [], {}]
LIMITS = [None, 0, 1, 50, 51, -5, 2**63, 1.5, "10", True, [], float("inf")]
EXPRS = ["1+1", "__import__('os')", "1/0", "(" * 500, "9**9**9", None, 5, "", "a.b", "1e999", "'x'*10**9"]
JUNK = [None, True, 0, -1, 2**70, 1.5, float("nan"), "", "text", "‮\u0000", [], [[]], {}, {"findings": []},
        [{"type": "X"}], {"a": {"b": [1, 2, {"c": None}]}}]


def garbage(rng):
    kind = rng.random()
    if kind < 0.05:
        return rng.choice(JUNK)
    action = {"action": rng.choice(ACTIONS)}
    for key, pool in (("table", TABLES), ("id", IDS), ("doc_id", IDS), ("cursor", CURSORS),
                      ("limit", LIMITS), ("expression", EXPRS), ("findings", JUNK)):
        if rng.random() < 0.5:
            action[key] = rng.choice(pool)
    if rng.random() < 0.1:
        action[rng.choice(["x", "__class__", "", "action "])] = rng.choice(JUNK)
    return action


def run_fuzz(seed, max_steps=MAX_STEPS):
    rng = random.Random(seed)
    env = AuditEnv()
    env.reset(seed=seed, config={"open_dir": OPEN, "max_steps": max_steps})
    prev_step = 0
    for _ in range(max_steps + 5):
        action = garbage(rng)
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except RuntimeError:
            return env  # эпизод закончился: только так среда имеет право «ругаться»
        assert isinstance(obs, dict)
        json.dumps(obs)  # наблюдение всегда чистый JSON
        assert reward == 0.0 and set(info) == {"step", "steps_left"}
        assert info["step"] == prev_step + 1 and 0 <= info["steps_left"] <= max_steps
        prev_step = info["step"]
        if "error" in obs:
            assert set(obs["error"]) == {"code", "message"} and len(obs["error"]["message"]) < 80
            assert "Traceback" not in obs["error"]["message"]
        assert not (terminated and truncated)
    raise AssertionError("эпизод не закончился за лимит шагов")


@pytest.mark.parametrize("seed", range(300))
def test_fuzz_episode(seed):
    env = run_fuzz(seed)
    assert env._ep.steps <= MAX_STEPS


def test_fuzz_never_leaks_truth():
    for seed in range(100):
        env = AuditEnv()
        env.reset(seed=seed, config={"open_dir": OPEN})
        rng = random.Random(seed)
        seen = []
        for _ in range(60):
            try:
                seen.append(env.step(garbage(rng))[0])
            except RuntimeError:
                break
        text = json.dumps(seen)
        assert "TRUTH-ONLY" not in text and "ERR-0001" not in text and "evidence_sets" not in text


def test_fuzz_is_deterministic():
    a, b = run_fuzz(42), run_fuzz(42)
    assert (a._ep.steps, a.submission) == (b._ep.steps, b.submission)


def test_unreadable_names_cannot_reach_hidden():
    env = AuditEnv()
    env.reset(seed=1, config={"open_dir": OPEN})
    for table in ("hidden", "truth", "../hidden", "../hidden/truth.json", "open/../hidden"):
        obs, *_ = env.step({"action": "list_records", "table": table})
        assert obs["error"]["code"] == "UNKNOWN_TABLE"
    for doc in ("../../hidden/truth", "../hidden/truth.json", "truth", "/etc/passwd"):
        obs, *_ = env.step({"action": "get_document", "doc_id": doc})
        assert obs["error"]["code"] == "UNKNOWN_DOCUMENT"
