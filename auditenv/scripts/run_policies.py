"""Прогон политик на эпизодах: таблица «политика × профиль» и trajectories.jsonl.

  python -m scripts.run_policies                      # мини-эпизод, 20 прогонов
  python -m scripts.run_policies --episodes-dir DIR   # папки эпизодов (open/ + hidden/)
  python -m scripts.run_policies --isolated           # агент в отдельном процессе

Код возврата 1, если oracle не получил максимум (цепочка сломана).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from env.harness import HostEpisode, TrajectoryLogger, run_policy_full
from env.isolated import run_policy_isolated
from evaluator import TruthGrader
from evaluator.matcher import match
from evaluator.schema import unwrap, validate_findings
from env.layout import is_episode_dir, open_dir_of
from policies import SPECS
from policies.oracle import OraclePolicy, load_correct_findings

ROOT = Path(__file__).resolve().parent.parent
MINI = ROOT / "tests" / "fixtures" / "mini"
ORDER = ["oracle", "careful", "careful_dup", "reconcile", "random", "flag_all"]
TYPES = ["duplicate_invoice", "missing_or_partial_allocation", "wrong_date_or_fx_rate", "period_cutoff"]


def find_episodes(episodes_dir, n):
    if episodes_dir is None:
        return [MINI] * n
    root = Path(episodes_dir)
    if is_episode_dir(root):
        dirs = [root]
    else:  # папка с эпизодами или с наборами train/dev внутри (на два уровня вглубь)
        dirs = sorted(q for p in [root, *root.iterdir()] if p.is_dir()
                      for q in p.iterdir() if q.is_dir() and is_episode_dir(q))
    if not dirs:
        raise SystemExit(f"В {episodes_dir} нет эпизодов (нужны public/+private/ или open/+hidden/)")
    return [dirs[i % len(dirs)] for i in range(n)]


def profile_of(ep_dir):
    tail = Path(ep_dir).name.rsplit("_", 1)[-1]  # ep_2003_hard -> hard
    if tail in ("clean", "easy", "medium", "hard"):
        return tail
    try:
        manifest = json.loads((open_dir_of(ep_dir) / "manifest.json").read_text(encoding="utf-8"))
        return str(manifest.get("profile", "default"))
    except (OSError, ValueError):
        return "default"


def run_one(policy, ep_dir, seed, isolated, log, config):
    grader = TruthGrader(ep_dir)
    host = HostEpisode(ep_dir, grader, log=log)
    if policy == "oracle":
        res = run_policy_full(host, OraclePolicy(load_correct_findings(ep_dir)), seed, config)
    elif isolated:
        res = run_policy_isolated(host, SPECS[policy], {"seed": seed}, seed, config)
    else:
        mod, cls = SPECS[policy].split(":")
        policy_obj = getattr(__import__(mod, fromlist=[cls]), cls)(seed=seed)
        res = run_policy_full(host, policy_obj, seed, config)
    m = grader.last.metrics if grader.last else None
    # сколько ошибок каждого типа найдено (для таблицы «политика × тип ошибки»)
    found, total = {t: 0 for t in TYPES}, {t: 0 for t in TYPES}
    for e in grader._errors:
        total[e["type"].lower()] += 1
    sub = host.env.submission
    if sub is not None and not validate_findings(sub):
        for _, j in match(unwrap(sub), grader._errors, grader.tolerance_minor):
            found[grader._errors[j]["type"].lower()] += 1
    return {"found": found, "total": total, "fp": m.fp if m else 0,
            "n_findings": (len(unwrap(sub)) if sub is not None and unwrap(sub) is not None else 0),
            "n_truth": len(grader._errors),

        "precision": m.precision if m else 0.0, "recall": m.recall if m else 0.0,
        "f1": m.f1 if m else 0.0, "reward": res.reward, "steps": res.steps,
        "elapsed_s": res.elapsed_s, "truncated": res.truncated, "agent_status": res.agent_status,
    }


def summarize(rows):
    keys = ("precision", "recall", "f1", "reward", "steps", "elapsed_s")
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-dir", default=str(ROOT / "data"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--isolated", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--policies", nargs="*", default=ORDER)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    traj = out / "trajectories.jsonl"
    traj.write_text("", encoding="utf-8")  # новый прогон = новый журнал

    episodes = find_episodes(args.episodes_dir, args.episodes)
    results = defaultdict(list)
    all_rows = []
    for policy in args.policies:
        for i, ep_dir in enumerate(episodes):
            profile = profile_of(ep_dir)
            log = TrajectoryLogger(traj, f"{policy}-{Path(ep_dir).name}-s{i:02d}")
            row = run_one(policy, ep_dir, i, args.isolated, log, {})
            row.update(policy=policy, profile=profile, episode=i)
            results[(policy, profile)].append(row)
            all_rows.append(row)

    lines = ["| политика | профиль | n | P | R | F1 | награда | шаги | время, с |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (policy, profile), rows in sorted(results.items(), key=lambda kv: (ORDER.index(kv[0][0]) if kv[0][0] in ORDER else 99, kv[0][1])):
        s = summarize(rows)
        lines.append(f"| {policy} | {profile} | {len(rows)} | {s['precision']:.2f} | {s['recall']:.2f} | "
                     f"{s['f1']:.2f} | {s['reward']:.3f} | {s['steps']:.1f} | {s['elapsed_s']:.3f} |")
    # вторая таблица: какие типы ошибок политика находит (найдено/всего по всем эпизодам)
    lines2 = ["| политика | " + " | ".join(TYPES) + " | ложных выводов |", "|---|" + "---|" * (len(TYPES) + 1)]
    for policy in [p for p in ORDER if p in args.policies]:
        rows = [r for r in all_rows if r["policy"] == policy]
        cells = [f"{sum(r['found'][t] for r in rows)}/{sum(r['total'][t] for r in rows)}" for t in TYPES]
        lines2.append(f"| {policy} | " + " | ".join(cells) + f" | {sum(r['fp'] for r in rows)} |")
    table = "\n".join(lines) + "\n\n" + "\n".join(lines2)
    print(table)
    (out / "results.md").write_text(table + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=1), encoding="utf-8")

    bad = [r for r in all_rows if r["agent_status"] != "ok"]
    if bad:
        print(f"ВНИМАНИЕ: {len(bad)} эпизодов с проблемой агента", file=sys.stderr)
    if "oracle" in args.policies:
        oracle = [r for r in all_rows if r["policy"] == "oracle"]
        if any(r["reward"] < 1.0 for r in oracle):
            print("ОШИБКА: oracle не получил максимум", file=sys.stderr)
            return 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
