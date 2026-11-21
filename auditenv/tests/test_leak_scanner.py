"""Сканер утечек находит специально внесённую утечку и не ругается на чистую среду."""
import json

import pytest

from env.environment import AuditEnv
from env.harness import HostEpisode, TrajectoryLogger, run_policy_full
from env.isolated import run_policy_isolated
from env.loader import load_open_company
from policies import SPECS
from policies.careful import CarefulPolicy
from qa.leaks import plant_canary, scan_observations, scan_text, scan_trajectories, secret_tokens


@pytest.fixture
def canary_ep(gen, tmp_path):
    ep = tmp_path / "ep"
    canary = plant_canary(gen["medium"], ep)
    return ep, canary


def test_secret_tokens_include_canary_and_exclude_public_ids(canary_ep):
    ep, canary = canary_ep
    tokens = secret_tokens(ep)
    assert canary in tokens
    assert "acceptable_evidence_sets" in tokens and "ISS-001" in tokens
    assert "duplicate_invoice" not in tokens  # имена четырёх типов публичны
    assert "INV-000001" not in tokens and "DOC-INV-000001" not in tokens  # есть в открытых данных


def run_clean(ep, log=None):
    host = HostEpisode(ep, __import__("evaluator").TruthGrader(ep), log=log)
    obs_log = []

    class Spy(CarefulPolicy):
        def act(self, obs):
            obs_log.append(obs)
            return super().act(obs)

    run_policy_full(host, Spy(), seed=1)
    return obs_log


def test_clean_env_has_no_leaks_in_observations(canary_ep):
    ep, canary = canary_ep
    assert scan_observations(run_clean(ep), secret_tokens(ep)) == []


def test_clean_isolated_run_has_no_leaks_in_trajectories(canary_ep, tmp_path):
    ep, _ = canary_ep
    path = tmp_path / "t.jsonl"
    host = HostEpisode(ep, __import__("evaluator").TruthGrader(ep), log=TrajectoryLogger(path, "x"))
    run_policy_isolated(host, SPECS["careful"], seed=1)
    assert scan_trajectories(path, secret_tokens(ep)) == []


def test_actions_with_error_type_names_are_not_flagged(canary_ep, tmp_path):
    ep, _ = canary_ep
    path = tmp_path / "t.jsonl"
    host = HostEpisode(ep, __import__("evaluator").TruthGrader(ep), log=TrajectoryLogger(path, "x"))
    run_policy_full(host, CarefulPolicy(), seed=1)
    assert "missing_or_partial_allocation" in path.read_text(encoding="utf-8") or "duplicate_invoice" in path.read_text(encoding="utf-8")
    assert scan_trajectories(path, secret_tokens(ep)) == []


# ---------- специально внесённые утечки ----------
def leaky_loader_factory(canary):
    def loader(open_dir, typed=False):
        company = load_open_company(open_dir, typed=typed)
        company["manifest"]["debug_note"] = canary  # утечка через manifest
        return company
    return loader


def test_leak_through_reset_observation_is_found(canary_ep):
    ep, canary = canary_ep
    env = AuditEnv(loader=leaky_loader_factory(canary))
    obs, _ = env.reset(seed=1, config={"open_dir": str(ep / "public")})
    found = scan_observations([obs], secret_tokens(ep))
    assert [t for t, _ in found] == [canary]


def test_leak_through_error_message_is_found(canary_ep):
    ep, canary = canary_ep

    class LeakyErrors(AuditEnv):
        def _list_records(self, action):
            if action.get("table") == "nope":
                return {"error": {"code": "UNKNOWN_TABLE", "message": f"нет таблицы, подсказка {canary}"}}
            return super()._list_records(action)

    env = LeakyErrors()
    env.reset(seed=1, config={"open_dir": str(ep / "public")})
    obs, *_ = env.step({"action": "list_records", "table": "nope"})
    assert scan_observations([obs], secret_tokens(ep))


def test_leak_through_info_is_found(canary_ep, tmp_path):
    ep, canary = canary_ep
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"episode_id": "x", "step": 1, "action": None, "observation": {},
                                "info": {"step": 1, "hint": canary}}) + "\n", encoding="utf-8")
    assert [t for t, _ in scan_trajectories(path, secret_tokens(ep))] == [canary]


def test_leak_through_trajectory_observation_is_found(canary_ep, tmp_path):
    ep, canary = canary_ep
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"episode_id": "x", "step": 2, "action": {"action": "get_document"},
                                "observation": {"text": f"заметка {canary} конец"}}) + "\n", encoding="utf-8")
    assert scan_trajectories(path, secret_tokens(ep))


def test_leak_of_truth_structure_is_found_without_canary(gen):
    tokens = secret_tokens(gen["hard"])
    text = json.dumps({"observation": {"x": "здесь ключ acceptable_evidence_sets и ISS-003"}})
    assert {t for t, _ in scan_text(text, tokens)} >= {"acceptable_evidence_sets", "ISS-003"}


def test_leak_of_error_type_name_in_observation_is_found(gen):
    tokens = secret_tokens(gen["hard"])
    assert not scan_text("в наблюдении сказано: period_cutoff", tokens)  # тип публичен
    assert scan_text("в наблюдении сказано: partial_cross_currency_boundary", tokens)  # сценарий секрет


def test_scanner_does_not_flag_ordinary_observation(gen):
    env = AuditEnv()
    env.reset(seed=1, config={"open_dir": str(gen["hard"] / "public")})
    obs = [env.step({"action": "list_records", "table": t, "limit": 50})[0]
           for t in ("invoices", "payments", "ledger")]
    assert scan_observations(obs, secret_tokens(gen["hard"])) == []


def test_cli_exit_codes(canary_ep, tmp_path):
    from qa.leaks import main
    ep, canary = canary_ep
    good, bad = tmp_path / "good.jsonl", tmp_path / "bad.jsonl"
    good.write_text(json.dumps({"observation": {"ok": 1}}) + "\n", encoding="utf-8")
    bad.write_text(json.dumps({"observation": {"leak": canary}}) + "\n", encoding="utf-8")
    assert main(["--episode", str(ep), "--trajectories", str(good)]) == 0
    assert main(["--episode", str(ep), "--trajectories", str(bad)]) == 1
