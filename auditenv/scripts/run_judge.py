"""Гонит организаторские политики через judge_adapter на эпизодах D01-D20 (как это делает жюри).

  python -m scripts.run_judge [--out runs/judge]

Политики: random (seed=1), reconcile, flag_all, oracle (правильный submit из truth на стороне жюри),
плюс наш careful. Пишет runs/judge/results.json и trajectories.jsonl (поля §09: episode_id, step,
action, observation, reward, terminated, truncated; в observation только то, что видит политика).
"""
import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path

import judge_adapter as ja
from policies.careful import CarefulPolicy
from policies.flag_all import FlagAllPolicy
from policies.judge_shim import EnvelopeShim
from policies.random_explorer import RandomExplorer
from policies.reconcile import ReconcilePolicy
from tests.official import MATRIX, build_all

POLICIES = {
    "random": lambda: EnvelopeShim(RandomExplorer(seed=1)),
    "reconcile": lambda: EnvelopeShim(ReconcilePolicy()),
    "flag_all": lambda: EnvelopeShim(FlagAllPolicy()),
    "careful": lambda: EnvelopeShim(CarefulPolicy()),
    "oracle": None,
}


def run_one(name, episode_dir, log=None, eid=""):
    env = ja.make_env()
    obs, _ = env.reset(1, {"mode": "closed", "episode_dir": str(episode_dir), "team_config": {}})
    policy = POLICIES[name]() if POLICIES[name] else None

    def write(step, action, o, r, t, tr):
        if log:
            log.write(json.dumps({"episode_id": f"{eid}:{name}", "step": step, "action": action, "observation": o,
                                  "reward": r, "terminated": t, "truncated": tr}, ensure_ascii=False) + "\n")

    write(0, None, obs, 0, False, False)
    step, done = 0, False
    while not done:
        action = policy.act(obs) if policy else ja.oracle_submit_action(episode_dir)
        obs, reward, term, trunc, info = env.step(action)
        step += 1
        write(step, action, obs, reward, term, trunc)
        done = term or trunc
    return ja.score_episode(str(episode_dir), env.submission if env.submission is not None else {"findings": []},
                            env.inspection_steps) | {"steps": step, "inspection_steps": env.inspection_steps}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/judge")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix="auditenv_D_"))
    eps = build_all(base)
    rows = {n: {} for n in POLICIES}
    with open(out / "trajectories.jsonl", "w", encoding="utf-8") as log:
        for n in POLICIES:
            for eid, *_ in MATRIX:
                rows[n][eid] = run_one(n, eps[eid], log, eid)
    mean = lambda n, k: sum(r[k] for r in rows[n].values()) / len(rows[n])  # noqa: E731
    summary = {n: {"mean_return": round(mean(n, "return"), 4), "mean_f1": round(mean(n, "f1"), 4),
                   "tp": sum(r["tp"] for r in rows[n].values()), "fp": sum(r["fp"] for r in rows[n].values()),
                   "fn": sum(r["fn"] for r in rows[n].values()),
                   "mean_inspection_steps": round(mean(n, "inspection_steps"), 1)} for n in POLICIES}
    from generator.generate_v1 import GENERATOR_VERSION
    try:
        import pytest
        pytest_v = pytest.__version__
    except ImportError:
        pytest_v = None
    meta = {"python": platform.python_version(), "platform": platform.platform(), "pytest": pytest_v,
            "generator_version": GENERATOR_VERSION, "judge_protocol": "genesis-fintech-checks-2026-11-20",
            "matrix": [{"id": m[0], "seed": m[1], "config": m[2]} for m in MATRIX],
            "mode": "closed", "policies": list(POLICIES)}
    (out / "results.json").write_text(json.dumps({"meta": meta, "summary": summary, "episodes": rows},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'policy':10} {'return':>8} {'F1':>6} {'TP':>4} {'FP':>5} {'FN':>4} {'insp':>6}")
    for n, s in summary.items():
        print(f"{n:10} {s['mean_return']:8.3f} {s['mean_f1']:6.3f} {s['tp']:4d} {s['fp']:5d} {s['fn']:4d} {s['mean_inspection_steps']:6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
