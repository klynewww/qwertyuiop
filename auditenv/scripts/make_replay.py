"""Собирает docs/replay.html: самодостаточную страницу «Лента аудита».

Прогоняет политики на одном открытом (dev) эпизоде, снимает траектории и
таблицы по 20 эпизодам и вшивает всё в шаблон. Эталон на странице показан
только после сдачи ответа и только потому, что эпизод открытый.

  python -m scripts.make_replay [--fragment PATH]   # fragment: версия без <html>-обёртки
"""
import argparse
import json
import sys
from pathlib import Path

from env.harness import HostEpisode, run_policy_full
from evaluator import TruthGrader
from evaluator.matcher import match
from evaluator.schema import unwrap, validate_findings
from policies import SPECS
from policies.oracle import OraclePolicy, load_correct_findings
from scripts.run_policies import ORDER, TYPES, find_episodes, run_one

ROOT = Path(__file__).resolve().parent.parent
EPISODE = ROOT / "data" / "dev" / "ep_2003_hard"

LABELS = {
    "oracle": ("Oracle", "Знает ответ заранее. Потолок: 1.0 без единого шага осмотра."),
    "careful": ("Дотошный аудитор", "Читает счета, платежи, проводки, сверяет курсы по правилам из policy.json. Решает задачу целиком."),
    "careful_dup": ("Только дубли", "Тот же аудитор, но проверяет только дубли счетов. Находит один тип из четырёх."),
    "reconcile": ("Простая сверка", "Ищет счета с одинаковыми клиентом и суммой. Ловится на подписках и возвратах, остальные типы не видит."),
    "random": ("Случайный исследователь", "15 случайных чтений, затем догадка из двух счетов. Нужен как нижняя планка."),
    "flag_all": ("Пометить всё", "Заявляет дубль для пар счетов, пока не упрётся в лимит 30 выводов. Атака шумом: награда её не прощает."),
}


def short(action) -> str:
    if action is None:
        return "—"
    if action.get("action") == "list_records":
        return f"list_records {action.get('table')}" + (f" c={action['cursor']}" if action.get("cursor") else "")
    if action.get("action") == "get_record":
        return f"get_record {action.get('table')} {str(action.get('id'))[:30]}"
    if action.get("action") == "submit":
        return f"submit ({len(action.get('findings') or [])} выводов)"
    return str(action.get("action"))


def obs_text(action, obs) -> str:
    if "error" in obs:
        return f"ошибка {obs['error']['code']}: {obs['error']['message']}"
    if obs.get("submitted"):
        return "ответ принят, эпизод завершён"
    if "records" in obs:
        recs = obs["records"]
        head = f"{obs['table']}: {len(recs)} из {obs['total']}, next_cursor={obs['next_cursor']}\n"
        return head + "\n".join(json.dumps(r, ensure_ascii=False)[:150] for r in recs[:3]) + ("\n…" if len(recs) > 3 else "")
    if "record" in obs:
        return json.dumps(obs["record"], ensure_ascii=False, indent=1)[:600]
    if "document" in obs:
        d = obs["document"]
        return (d["text"] if "text" in d else json.dumps(d, ensure_ascii=False, indent=1))[:600]
    return json.dumps(obs, ensure_ascii=False)[:400]


class Collector:
    def __init__(self):
        self.rows = []

    def write(self, step, action, observation, reward, terminated, truncated):
        self.rows.append((step, action, observation))


def play(policy_name, ep):
    grader = TruthGrader(ep)
    col = Collector()
    host = HostEpisode(ep, grader, log=col)
    if policy_name == "oracle":
        policy = OraclePolicy(load_correct_findings(ep))
    else:
        mod, cls = SPECS[policy_name].split(":")
        policy = getattr(__import__(mod, fromlist=[cls]), cls)(seed=2003)
    res = run_policy_full(host, policy, seed=2003)
    steps = []
    for step, action, obs in col.rows[1:]:
        steps.append({"kind": action.get("action", "?"), "short": short(action),
                      "action": json.dumps(action, ensure_ascii=False, indent=1)[:900],
                      "obs": obs_text(action, obs), "read": action.get("action") in ("list_records", "get_record", "get_document")})
    sub = host.env.submission
    items = unwrap(sub) if sub is not None and not validate_findings(sub) else []
    pairs = match(items, grader._errors, grader.tolerance_minor, grader.known_ids)
    hit_i = {i for i, _ in pairs}
    findings = [{"type": f["type"], "ids": f["affected_ids"], "amount": f["claimed_amount_minor"], "currency": f["currency"],
                 "hit": i in hit_i} for i, f in enumerate(items)]
    # сначала верные, потом ложные: так видно суть
    findings.sort(key=lambda f: (not f["hit"], f["type"]))
    m = grader.last.metrics
    label, blurb = LABELS[policy_name]
    return {"label": label, "blurb": blurb, "steps": steps, "findings": findings, "matched": sorted(j for _, j in pairs),
            "reward": res.reward, "tp": m.tp, "fp": m.fp, "fn": m.fn, "f1": m.f1}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", help="путь для версии без <html>-обёртки (для публикации как Artifact)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "replay.html"))
    args = ap.parse_args(argv)

    ep = EPISODE
    from env.environment import AuditEnv
    env = AuditEnv()
    reset, _ = env.reset(seed=1, config={"open_dir": str(ep / "public")})
    truth_doc = json.loads((ep / "private" / "truth.json").read_text(encoding="utf-8"))
    truth = [{"type": i["type"], "ids": i["affected_ids"], "amount": i["expected_amount_minor"], "currency": i["currency"]}
             for i in truth_doc["issues"]]
    runs = {p: play(p, ep) for p in ORDER}

    episodes = find_episodes(ROOT / "data", 20)
    res = {p: [run_one(p, e, i, False, None, {}) for i, e in enumerate(episodes)] for p in ORDER}
    mean = lambda p, k: sum(r[k] for r in res[p]) / len(res[p])  # noqa: E731
    ladder = [{"label": LABELS[p][0], "blurb": LABELS[p][1], "reward": mean(p, "reward"), "f1": mean(p, "f1"),
               "steps": mean(p, "steps")} for p in ORDER]
    rows = [{"label": LABELS[p][0], "fp": sum(r["fp"] for r in res[p]),
             "cells": {t: [sum(r["found"][t] for r in res[p]), sum(r["total"][t] for r in res[p])] for t in TYPES}} for p in ORDER]
    manifest = json.loads((ep / "public" / "manifest.json").read_text(encoding="utf-8"))
    data = {
        "episode": {"name": ep.name, "profile": ep.name.rsplit("_", 1)[-1], "n_issues": len(truth), "counts": manifest["counts"]},
        "reset": (f"компания {manifest['episode_id']}, период {manifest['period_start']} — {manifest['period_end']}, "
                  f"часовой пояс {manifest['timezone']}\nтаблицы: " + ", ".join(f"{k} {v}" for k, v in reset["tables"].items())
                  + "\nвалюты: " + ", ".join(f"{k} ({v} зн.)" for k, v in manifest["currencies"].items())),
        "order": ORDER, "runs": runs, "truth": truth, "ladder": ladder, "types": {"columns": TYPES, "rows": rows},
    }
    template = (ROOT / "scripts" / "replay_template.html").read_text(encoding="utf-8")
    body = template.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
    if args.fragment:
        Path(args.fragment).write_text(body, encoding="utf-8")
    head = ('<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
    Path(args.out).write_text(head + body.replace("<title>", "<title>", 1) + "\n</html>\n", encoding="utf-8")
    print(f"готово: {args.out}")


if __name__ == "__main__":
    sys.exit(main())
