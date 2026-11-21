"""Воспроизводимость: один seed -> одни и те же байты, независимо от процесса, порядка и PYTHONHASHSEED."""
import hashlib
import json
import os
import subprocess
import sys

import pytest

from env.harness import HostEpisode, TrajectoryLogger, run_policy_full
from evaluator import TruthGrader
from generator.generate_v1 import build_episode, write_episode
from policies.careful import CarefulPolicy
from policies.random_explorer import RandomExplorer


def digest(ep: dict) -> str:
    h = hashlib.sha256()
    for part in ("public", "private"):
        for name in sorted(ep[part]):
            h.update(name.encode() + b"\0" + ep[part][name])
    return h.hexdigest()


@pytest.mark.parametrize("profile", ["clean", "easy", "medium", "hard"])
def test_same_seed_same_bytes(profile):
    assert digest(build_episode(77, {"profile": profile})) == digest(build_episode(77, {"profile": profile}))


def test_different_seeds_differ():
    assert len({digest(build_episode(s, {"profile": "medium"})) for s in range(5)}) == 5


def test_profile_changes_data():
    assert digest(build_episode(5, {"profile": "easy"})) != digest(build_episode(5, {"profile": "hard"}))


def test_two_episodes_in_one_process_do_not_interfere():
    a1 = digest(build_episode(1, {"profile": "hard"}))
    b1 = digest(build_episode(2, {"profile": "easy"}))
    b2 = digest(build_episode(2, {"profile": "easy"}))
    a2 = digest(build_episode(1, {"profile": "hard"}))
    assert (a1, b1) == (a2, b2)


def test_global_random_is_not_used():
    import random
    random.seed(1)
    a = digest(build_episode(9, {"profile": "hard"}))
    random.seed(999)
    random.random()
    assert a == digest(build_episode(9, {"profile": "hard"}))


@pytest.mark.parametrize("hashseed", ["0", "1", "424242"])
def test_independent_of_pythonhashseed(hashseed):
    code = ("from tests.test_reproducibility import digest; from generator.generate_v1 import build_episode;"
            "print(digest(build_episode(31, {'profile': 'hard'})))")
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         cwd=os.path.dirname(os.path.dirname(__file__)), check=True).stdout.strip()
    assert out == digest(build_episode(31, {"profile": "hard"}))


def test_written_files_match_in_memory_bytes(tmp_path):
    ep = build_episode(3, {"profile": "medium"})
    root = write_episode(ep, tmp_path / "e")
    for part in ("public", "private"):
        for name, data in ep[part].items():
            assert (root / part / name).read_bytes() == data


def test_config_round_trip_in_provenance():
    ep = build_episode(8, {"profile": "hard"})
    prov = json.loads(ep["private"]["provenance.json"])
    assert prov["seed"] == 8 and prov["config"]["profile"] == "hard"
    again = build_episode(prov["seed"], prov["config"])
    assert digest(again) == digest(ep)  # по provenance эпизод воссоздаётся точно


def run_logged(gen_ep, policy, path):
    host = HostEpisode(gen_ep, TruthGrader(gen_ep), log=TrajectoryLogger(path, "ep"))
    run_policy_full(host, policy, seed=4)
    return path.read_bytes()


def test_trajectories_are_byte_identical_for_same_seed(gen, tmp_path):
    a = run_logged(gen["hard"], CarefulPolicy(), tmp_path / "a.jsonl")
    b = run_logged(gen["hard"], CarefulPolicy(), tmp_path / "b.jsonl")
    assert a == b
    c = run_logged(gen["hard"], RandomExplorer(seed=4), tmp_path / "c.jsonl")
    d = run_logged(gen["hard"], RandomExplorer(seed=4), tmp_path / "d.jsonl")
    assert c == d
    e = run_logged(gen["hard"], RandomExplorer(seed=5), tmp_path / "e.jsonl")
    assert c != e
