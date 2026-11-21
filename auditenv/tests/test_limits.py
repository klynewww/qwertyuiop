"""Лимиты: шаги, время, повторная отправка, неверный ввод."""
import pytest

from env.environment import AuditEnv, TIME_LIMIT_S
from env.harness import HostEpisode, run_policy_full
from evaluator import TruthGrader
from tests.helpers import MINI

OPEN = str(MINI / "open")


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(clock, **cfg):
    env = AuditEnv(clock=clock)
    env.reset(seed=1, config={"open_dir": OPEN, **cfg})
    return env


def test_default_time_limit_is_60():
    assert TIME_LIMIT_S == 60.0


def test_step_within_time_is_fine():
    clock = FakeClock()
    env = make(clock)
    clock.t += 59.9
    obs, _, term, trunc, _ = env.step({"action": "list_records", "table": "fx"})
    assert "records" in obs and not term and not trunc


def test_timeout_truncates_and_drops_answer():
    clock = FakeClock()
    env = make(clock)
    clock.t += 60.5
    obs, _, term, trunc, info = env.step({"action": "submit", "findings": [{"x": 1}]})
    assert obs["error"]["code"] == "TIMEOUT"
    assert trunc and not term and info["step"] == 1
    assert env.submission is None and env.timed_out
    with pytest.raises(RuntimeError):
        env.step({"action": "list_records", "table": "fx"})


def test_timeout_episode_is_graded_as_empty_answer():
    clock = FakeClock()
    host = HostEpisode(MINI, TruthGrader(MINI), env=AuditEnv(clock=clock))
    host.reset(seed=1)

    class Slow:
        def act(self, obs):
            clock.t += 61
            return {"action": "submit", "findings": [{"leaks": "nothing"}]}

    res = run_policy_full(host, Slow(), seed=1)
    assert res.truncated and not res.terminated
    assert res.reward == -1.0  # пустой ответ на компании с ошибкой


def test_custom_time_limit_from_config():
    clock = FakeClock()
    env = make(clock, time_limit_s=5)
    clock.t += 6
    obs, _, _, trunc, _ = env.step({"action": "list_records", "table": "fx"})
    assert trunc and obs["error"]["code"] == "TIMEOUT"


def test_timeout_does_not_count_as_inspection():
    clock = FakeClock()
    env = make(clock)
    clock.t += 61
    env.step({"action": "list_records", "table": "fx"})
    assert env.inspection_steps == 0


def test_repeated_submit_is_rejected():
    env = make(FakeClock())
    env.step({"action": "submit", "findings": []})
    with pytest.raises(RuntimeError):
        env.step({"action": "submit", "findings": [{"a": 1}]})
    assert env.submission == []  # первая отправка не перезаписана


def test_step_after_truncation_is_rejected():
    env = make(FakeClock(), max_steps=1)
    _, _, _, trunc, _ = env.step({"action": "list_records", "table": "fx"})
    assert trunc
    with pytest.raises(RuntimeError):
        env.step({"action": "submit", "findings": []})


@pytest.mark.parametrize("findings", [
    pytest.param(float("nan"), id="nan"), pytest.param({1, 2}, id="set"), pytest.param(object(), id="object"),
    pytest.param(b"x", id="bytes"), pytest.param("x" * 1_100_000, id="huge_string"),  # короткий id: в Windows имя теста попадает в переменную окружения (лимит 32767)
])
def test_submit_rejects_non_json_or_huge(findings):
    env = make(FakeClock())
    obs, _, term, *_ = env.step({"action": "submit", "findings": findings})
    assert obs["error"]["code"] == "INVALID_ARGUMENT" and not term


def test_submit_rejects_cyclic_and_deep_structures():
    env = make(FakeClock())
    cyc = []
    cyc.append(cyc)
    obs, _, term, *_ = env.step({"action": "submit", "findings": cyc})
    assert obs["error"]["code"] == "INVALID_ARGUMENT" and not term
    deep = []
    for _ in range(100_000):
        deep = [deep]
    obs, _, term, *_ = env.step({"action": "submit", "findings": deep})
    assert obs["error"]["code"] == "INVALID_ARGUMENT" and not term


def test_submit_stores_independent_copy():
    env = make(FakeClock())
    findings = [{"type": "X"}]
    env.step({"action": "submit", "findings": findings})
    findings[0]["type"] = "CHANGED"
    assert env.submission == [{"type": "X"}]


@pytest.mark.parametrize("cursor", [pytest.param(c, id=f"cursor{i}") for i, c in enumerate(
    ["c²", "c١", "c" + "9" * 5000, "c 1", "c+1", "c-0", "c1.5", "C1", "c"])])
def test_unicode_and_huge_cursors_are_errors_not_crashes(cursor):
    env = make(FakeClock())
    obs, *_ = env.step({"action": "list_records", "table": "ledger", "cursor": cursor})
    assert obs["error"]["code"] == "INVALID_CURSOR"
