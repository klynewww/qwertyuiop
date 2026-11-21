"""Среда и oracle на данных настоящего генератора A (public/ + private/)."""
import json

import pytest

from env.compat import make_grader
from env.environment import AuditEnv
from env.harness import HostEpisode, TrajectoryLogger, run_policy_full
from env.isolated import run_policy_isolated
from generator.generate_v1 import build_episode, write_episode
from policies import SPECS
from policies.oracle import OraclePolicy, load_correct_findings

PROFILES = ["clean", "easy", "medium", "hard"]
HIDDEN_WORDS = ["acceptable_evidence_sets", "expected_amount_minor", "ISS-0", "injections",
                "shift_seconds", "generator_version", "provenance"]


@pytest.fixture(scope="module")
def episodes(tmp_path_factory):
    root = tmp_path_factory.mktemp("gen")
    out = {}
    for i, profile in enumerate(PROFILES):
        out[profile] = write_episode(build_episode(3000 + i, {"profile": profile}), root / f"ep_{3000 + i}_{profile}")
    return out


@pytest.mark.parametrize("profile", PROFILES)
def test_oracle_gets_max_reward_on_generated(episodes, profile):
    ep = episodes[profile]
    host = HostEpisode(ep, make_grader(ep))
    res = run_policy_full(host, OraclePolicy(load_correct_findings(ep)), seed=1)
    assert res.reward == 1.0 and res.steps == 1


@pytest.mark.parametrize("profile", ["easy", "hard"])
def test_empty_answer_on_faulty_generated_is_minimum(episodes, profile):
    ep = episodes[profile]

    class Empty:
        def act(self, obs):
            return {"action": "submit", "findings": []}

    assert run_policy_full(HostEpisode(ep, make_grader(ep)), Empty(), seed=1).reward == -1.0


def test_env_reads_generated_company(episodes):
    ep = episodes["medium"]
    env = AuditEnv()
    obs, _ = env.reset(seed=1, config={"open_dir": str(ep / "public")})
    assert obs["tables"]["invoices"] == obs["manifest"]["counts"]["invoices"]
    page, *_ = env.step({"action": "list_records", "table": "invoices", "limit": 50})
    first = page["records"][0]
    doc, *_ = env.step({"action": "get_document", "doc_id": first["doc_id"]})
    assert doc["document"] == first
    rec, *_ = env.step({"action": "get_record", "table": "invoices", "id": first["invoice_id"]})
    assert rec["record"] == first


def test_fx_records_are_addressable_by_computed_id(episodes):
    env = AuditEnv()
    env.reset(seed=1, config={"open_dir": str(episodes["easy"] / "public")})
    page, *_ = env.step({"action": "list_records", "table": "fx", "limit": 1})
    r = page["records"][0]
    fx_id = f"FX:{r['date']}:{r['source_currency']}:{r['target_currency']}"
    got, *_ = env.step({"action": "get_record", "table": "fx", "id": fx_id})
    assert got["record"] == r


def test_listing_is_complete_and_stable(episodes):
    env = AuditEnv()
    obs, _ = env.reset(seed=1, config={"open_dir": str(episodes["hard"] / "public")})
    seen, cursor = [], None
    while True:
        act = {"action": "list_records", "table": "ledger", "limit": 50}
        if cursor:
            act["cursor"] = cursor
        page, *_ = env.step(act)
        seen += [r["row_id"] for r in page["records"]]
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert len(seen) == obs["tables"]["ledger"] == len(set(seen)) and seen == sorted(seen)


@pytest.mark.parametrize("policy", ["reconcile", "flag_all", "random"])
@pytest.mark.parametrize("isolated", [False, True])
def test_policies_complete_on_generated(episodes, policy, isolated):
    ep = episodes["hard"]
    host = HostEpisode(ep, make_grader(ep))
    if isolated:
        res = run_policy_isolated(host, SPECS[policy], {"seed": 1}, seed=1)
    else:
        mod, cls = SPECS[policy].split(":")
        res = run_policy_full(host, getattr(__import__(mod, fromlist=[cls]), cls)(seed=1), seed=1)
    assert res.terminated and res.agent_status == "ok" and -1.0 <= res.reward <= 1.0


def test_trajectories_do_not_contain_truth(episodes, tmp_path):
    ep = episodes["hard"]
    path = tmp_path / "t.jsonl"
    host = HostEpisode(ep, make_grader(ep), log=TrajectoryLogger(path, "g"))
    run_policy_isolated(host, SPECS["reconcile"], seed=1)
    text = path.read_text(encoding="utf-8")
    for word in HIDDEN_WORDS:
        assert word not in text, word
    # и ни один ответ из truth не процитирован целиком
    truth = json.loads((ep / "private" / "truth.json").read_text(encoding="utf-8"))
    for issue in truth["issues"]:
        assert issue["issue_id"] not in text


def test_agent_process_cannot_read_generated_private(episodes):
    from tests.attackers import HiddenReader  # noqa: F401  (проверяем импорт)
    ep = episodes["hard"]
    host = HostEpisode(ep, make_grader(ep))
    run_policy_isolated(host, "tests.attackers:HiddenReader", seed=1)
    answer = host.env.submission[0]
    assert answer["leaks"] == []
