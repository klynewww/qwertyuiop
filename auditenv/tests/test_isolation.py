"""truth не должен попадать в память агентской стороны."""
import builtins
import json
from pathlib import Path

from env.environment import AuditEnv
from env.harness import HostEpisode
from evaluator import TruthGrader
from tests.helpers import MINI

MARKER = "TRUTH-ONLY-7f3a"


def test_marker_really_is_in_truth():
    assert MARKER in (MINI / "hidden" / "truth.json").read_text(encoding="utf-8")


def test_loader_opens_only_open_dir(monkeypatch):
    opened = []
    real_open = builtins.open

    def spy(file, *a, **kw):
        opened.append(str(file))
        return real_open(file, *a, **kw)

    monkeypatch.setattr(builtins, "open", spy)
    env = AuditEnv()
    env.reset(seed=1, config={"open_dir": str(MINI / "open")})
    assert opened, "шпион ничего не поймал"
    for path in opened:
        assert str(MINI / "hidden") not in path
        assert "truth" not in Path(path).name and "provenance" not in Path(path).name


def test_env_memory_has_no_truth():
    env = AuditEnv()
    env.reset(seed=1, config={"open_dir": str(MINI / "open")})
    env.step({"action": "list_records", "table": "invoices"})
    dumped = json.dumps(vars(env._ep), default=lambda o: repr(vars(o)) if hasattr(o, "__dict__") else repr(o))
    assert MARKER not in dumped
    assert "ERR-0001" not in dumped
    assert "evidence_sets" not in dumped


def test_env_has_no_reference_to_grader():
    host = HostEpisode(MINI, TruthGrader(MINI))
    host.reset(seed=1)
    assert not any(hasattr(host.env, n) for n in ("grader", "truth", "_grader", "_truth"))
    assert "grader" not in vars(host.env._ep)


def test_no_observation_leaks_truth():
    env = AuditEnv()
    obs, _ = env.reset(seed=1, config={"open_dir": str(MINI / "open")})
    seen = [obs]
    for tbl in ("ledger", "invoices", "payments", "fx"):
        seen.append(env.step({"action": "list_records", "table": tbl, "limit": 50})[0])
    text = json.dumps(seen)
    assert MARKER not in text and "ERR-0001" not in text
