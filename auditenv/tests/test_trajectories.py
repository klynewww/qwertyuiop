import json

from env.harness import HostEpisode, TrajectoryLogger, run_policy_full
from evaluator import TruthGrader
from policies.oracle import OraclePolicy, load_correct_findings
from policies.reconcile import ReconcilePolicy
from tests.helpers import MINI

FIELDS = {"episode_id", "step", "action", "observation", "reward", "terminated", "truncated"}
HIDDEN_SNIPPETS = ["TRUTH-ONLY-7f3a", "ERR-0001", "evidence_sets", "provenance"]


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run(tmp_path, policy):
    path = tmp_path / "trajectories.jsonl"
    host = HostEpisode(MINI, TruthGrader(MINI), log=TrajectoryLogger(path, "ep-1"))
    res = run_policy_full(host, policy, seed=1)
    return res, rows(path)


def test_log_written_with_required_fields(tmp_path):
    res, rs = run(tmp_path, ReconcilePolicy())
    assert [r["step"] for r in rs] == [0, 1, 2]
    assert all(set(r) == FIELDS and r["episode_id"] == "ep-1" for r in rs)
    assert rs[0]["action"] is None and rs[1]["action"]["action"] == "list_records"
    assert rs[-1]["terminated"] is True and rs[-1]["reward"] == res.reward
    assert all(r["reward"] == 0.0 for r in rs[:-1])


def test_log_has_no_truth(tmp_path):
    _, rs = run(tmp_path, ReconcilePolicy())
    text = json.dumps(rs, ensure_ascii=False)
    for snippet in HIDDEN_SNIPPETS:
        assert snippet not in text


def test_log_appends_across_episodes(tmp_path):
    path = tmp_path / "t.jsonl"
    for i in range(3):
        host = HostEpisode(MINI, TruthGrader(MINI), log=TrajectoryLogger(path, f"ep-{i}"))
        run_policy_full(host, OraclePolicy(load_correct_findings(MINI)), seed=i)
    assert {r["episode_id"] for r in rows(path)} == {"ep-0", "ep-1", "ep-2"}


def test_log_survives_weird_actions(tmp_path):
    path = tmp_path / "t.jsonl"
    host = HostEpisode(MINI, TruthGrader(MINI), log=TrajectoryLogger(path, "w"))
    host.reset(seed=1)
    host.step({"action": "list_records", "table": {1, 2}})  # set не JSON: пишется через repr
    host.step({"action": float("nan")})
    assert len(rows(path)) == 3


def test_truncated_episode_logged(tmp_path):
    path = tmp_path / "t.jsonl"
    host = HostEpisode(MINI, TruthGrader(MINI), log=TrajectoryLogger(path, "t"))
    host.reset(seed=1, config={"max_steps": 2})
    host.step({"action": "list_records", "table": "fx"})
    host.step({"action": "list_records", "table": "fx"})
    last = rows(path)[-1]
    assert last["truncated"] is True and last["terminated"] is False and last["reward"] == -1.0
