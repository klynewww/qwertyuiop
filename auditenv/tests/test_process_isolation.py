"""Агент в отдельном процессе: не может открыть hidden/, падения и зависания безопасны."""
import json
import os

import pytest

from env.harness import HostEpisode, TrajectoryLogger
from env.isolated import AgentProcess, run_policy_isolated
from evaluator import TruthGrader
from policies import SPECS
from policies.oracle import OraclePolicy, load_correct_findings
from tests.attackers import run_attacks
from tests.helpers import MINI

MARKER = "TRUTH-ONLY-7f3a"
A = "tests.attackers:"


def host(log=None):
    return HostEpisode(MINI, TruthGrader(MINI), log=log)


def test_attacks_would_succeed_without_sandbox(monkeypatch):
    """Контроль теста: в обычном процессе те же приёмы читают hidden/ (значит тест не пустой)."""
    monkeypatch.setenv("AUDIT_SECRET_FOR_TEST", "s3cret")
    leaks, blocked = run_attacks(str(MINI / "hidden"))
    assert {"open_abs", "open_dotdot", "pathlib_read", "listdir", "scandir", "glob",
            "os_open", "subprocess_cat", "os_system", "truth_grader", "env_secret"} <= set(leaks)


def test_agent_process_cannot_open_hidden(monkeypatch):
    monkeypatch.setenv("AUDIT_SECRET_FOR_TEST", "s3cret")
    h = host()
    res = run_policy_isolated(h, A + "HiddenReader", seed=1)
    answer = h.env.submission[0]
    assert answer["leaks"] == [], f"утечки: {answer['leaks']}"
    assert len(answer["blocked"]) >= 12
    assert res.agent_status == "ok" and res.terminated
    assert MARKER not in json.dumps(h.env.submission)


def test_good_policy_runs_end_to_end_in_separate_process():
    res = run_policy_isolated(host(), SPECS["reconcile"], seed=1)
    assert res.terminated and res.agent_status == "ok"
    assert res.reward == pytest.approx(0.998)  # 1 шаг осмотра (один список) -> штраф 0.002


def test_isolated_gives_same_result_as_in_process():
    from env.harness import run_policy_full
    from policies.reconcile import ReconcilePolicy
    a = run_policy_full(host(), ReconcilePolicy(), seed=1)
    b = run_policy_isolated(host(), SPECS["reconcile"], seed=1)
    assert (a.reward, a.steps) == (b.reward, b.steps)


def test_random_policy_is_deterministic_by_seed_across_processes():
    r1 = run_policy_isolated(host(), SPECS["random"], {"seed": 5}, seed=5)
    r2 = run_policy_isolated(host(), SPECS["random"], {"seed": 5}, seed=5)
    assert (r1.reward, r1.steps) == (r2.reward, r2.steps)


def test_hanging_agent_is_cut_by_time_limit():
    res = run_policy_isolated(host(), A + "Hang", seed=1, config={"time_limit_s": 1})
    assert res.truncated and not res.terminated
    assert res.agent_status == "timeout" and res.reward == -1.0
    assert res.elapsed_s < 5


def test_crashing_agent_is_truncated_with_empty_answer():
    h = host()
    res = run_policy_isolated(h, A + "Crash", seed=1)
    assert res.truncated and res.agent_status == "crashed" and res.reward == -1.0
    assert h.env.submission is None


@pytest.mark.parametrize("name,status", [("NotJson", "crashed"), ("HugeLine", "protocol_error")])
def test_protocol_violations(name, status):
    res = run_policy_isolated(host(), A + name, seed=1)
    assert res.agent_status == status and res.truncated and res.reward == -1.0


@pytest.mark.parametrize("spec", [
    "os:system", "policies.oracle", "builtins:eval", "policies.x:Y; import os", "../x:Y",
    "tests.test_env:Foo", "", "policies.:X",
])
def test_bad_policy_spec_is_rejected(spec):
    with pytest.raises(ValueError):
        AgentProcess(spec, {}, deny_paths=[])


def test_missing_policy_module_does_not_hang():
    res = run_policy_isolated(host(), "policies.no_such_module:Nope", seed=1)
    assert res.agent_status == "crashed" and res.truncated


def test_child_gets_no_environment_secrets(monkeypatch):
    # секрет в окружении хоста не должен доезжать до агента (проверка внутри test_agent_process_cannot_open_hidden)
    monkeypatch.setenv("AUDIT_SECRET_FOR_TEST", "s3cret")
    h = host()
    run_policy_isolated(h, A + "HiddenReader", seed=1)
    assert "env_secret:empty" in h.env.submission[0]["blocked"]


def test_isolated_log_has_no_truth(tmp_path):
    path = tmp_path / "t.jsonl"
    run_policy_isolated(host(TrajectoryLogger(path, "iso")), SPECS["reconcile"], seed=1)
    text = path.read_text(encoding="utf-8")
    assert text.count("\n") == 3 and MARKER not in text and "evidence_sets" not in text


def test_two_isolated_episodes_in_a_row_do_not_interfere():
    for _ in range(2):
        assert run_policy_isolated(host(), SPECS["reconcile"], seed=1).terminated
