"""Парные проверки: тот же эпизод с другими ID и порядком записей -> награда не меняется."""
import pytest

from env.harness import HostEpisode, run_policy_full
from evaluator import TruthGrader
from generator.generate_v1 import build_episode, build_paired_episode, write_episode
from policies.careful import CarefulPolicy
from policies.oracle import OraclePolicy, load_correct_findings
from policies.reconcile import ReconcilePolicy
from qa.evidence import validate_evidence_dir
from qa.generated import validate_generated_dir

PROFILES = ["clean", "easy", "medium", "hard"]
PAIRS = [(6000 + i, PROFILES[i % 4], 100 + i) for i in range(20)]  # 20 пар


def play(ep, policy):
    grader = TruthGrader(ep)
    res = run_policy_full(HostEpisode(ep, grader), policy, seed=1)
    return res, grader.last.metrics


@pytest.fixture(scope="module")
def pairs(tmp_path_factory):
    root = tmp_path_factory.mktemp("pairs")
    out = []
    for seed, profile, permute in PAIRS:
        cfg = {"profile": profile}
        a = write_episode(build_episode(seed, cfg), root / f"a_{seed}")
        b = write_episode(build_paired_episode(seed, cfg, permute), root / f"b_{seed}")
        out.append((seed, profile, a, b))
    return out


def test_there_are_20_pairs(pairs):
    assert len(pairs) == 20


def test_pair_really_differs_in_ids_and_order(pairs):
    different = 0
    for _, profile, a, b in pairs:
        if (a / "public" / "invoices.jsonl").read_bytes() != (b / "public" / "invoices.jsonl").read_bytes():
            different += 1
    assert different == 20


def test_both_members_of_a_pair_are_valid(pairs):
    for _, _, a, b in pairs:
        for ep in (a, b):
            assert validate_generated_dir(ep) == []
            assert validate_evidence_dir(ep) == []


def test_truth_content_is_preserved(pairs):
    import json
    for _, _, a, b in pairs:
        ta = json.loads((a / "private" / "truth.json").read_text(encoding="utf-8"))["issues"]
        tb = json.loads((b / "private" / "truth.json").read_text(encoding="utf-8"))["issues"]
        sig = lambda t: sorted((i["type"], i["expected_amount_minor"], i["currency"], i["scenario"] or "") for i in t)  # noqa: E731
        assert sig(ta) == sig(tb)


def test_oracle_reward_unchanged_in_pairs(pairs):
    for _, _, a, b in pairs:
        ra, _ = play(a, OraclePolicy(load_correct_findings(a)))
        rb, _ = play(b, OraclePolicy(load_correct_findings(b)))
        assert ra.reward == rb.reward == 1.0


def test_careful_reward_unchanged_in_pairs(pairs):
    for seed, _, a, b in pairs:
        (ra, ma), (rb, mb) = play(a, CarefulPolicy()), play(b, CarefulPolicy())
        assert (ma.tp, ma.fp, ma.fn) == (mb.tp, mb.fp, mb.fn), seed
        assert ra.reward == pytest.approx(rb.reward, abs=0.02)  # шагов может быть чуть больше из-за другого порядка курсов


def test_reconcile_metrics_unchanged_in_pairs(pairs):
    for seed, _, a, b in pairs:
        (_, ma), (_, mb) = play(a, ReconcilePolicy()), play(b, ReconcilePolicy())
        assert (ma.tp, ma.fp, ma.fn) == (mb.tp, mb.fp, mb.fn), seed


def test_empty_answer_reward_unchanged_in_pairs(pairs):
    class Empty:
        def act(self, obs):
            return {"action": "submit", "findings": []}

    for _, _, a, b in pairs:
        assert play(a, Empty())[0].reward == play(b, Empty())[0].reward
