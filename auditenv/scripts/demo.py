"""Демо на одном открытом эпизоде: три сцены. Пишет demo/transcript.txt и печатает его.

  python -m scripts.demo

Сцена 1: дотошный аудитор решает задачу и получает почти максимум.
Сцена 2: «пометить всё»: шум, который награда не прощает.
Сцена 3: агент-взломщик пытается прочитать эталон из своего процесса: все 14 приёмов заблокированы.
"""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from env.harness import HostEpisode, run_policy_full
from env.isolated import run_policy_isolated
from evaluator import TruthGrader
from policies.careful import CarefulPolicy
from policies.flag_all import FlagAllPolicy

ROOT = Path(__file__).resolve().parent.parent
EP = ROOT / "data" / "dev" / "ep_2003_hard"


def scene(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


class Tape:
    def __init__(self):
        self.n = 0

    def write(self, step, action, observation, reward, terminated, truncated):
        if action is None:
            return
        a = action["action"]
        what = {"list_records": lambda: f"list_records {action['table']}",
                "get_record": lambda: f"get_record {action['table']} {action['id']}",
                "submit": lambda: f"submit ({len(action['findings'])} выводов)"}.get(a, lambda: a)()
        if step <= 6 or a == "submit" or step % 5 == 0:
            print(f"  шаг {step:>3}  {what}")


def play(policy):
    g = TruthGrader(EP)
    host = HostEpisode(EP, g, log=Tape())
    res = run_policy_full(host, policy, seed=1)
    m = g.last.metrics
    print(f"  итог: TP={m.tp} FP={m.fp} FN={m.fn}  F1={m.f1:.2f}  осмотр={res.inspection_steps} шагов  награда={res.reward:+.3f}")


def main():
    buf = io.StringIO()
    out = []
    for title, fn in (
        ("Сцена 1. Дотошный аудитор: читает книги, сверяет курсы по policy.json", lambda: play(CarefulPolicy())),
        ("Сцена 2. «Пометить всё»: 30 выводов подряд", lambda: play(FlagAllPolicy())),
        ("Сцена 3. Взломщик в отдельном процессе пытается прочитать эталон", attack),
    ):
        with redirect_stdout(buf):
            scene(title)
            fn()
    text = buf.getvalue()
    Path(ROOT / "demo").mkdir(exist_ok=True)
    (ROOT / "demo" / "transcript.txt").write_text(text, encoding="utf-8")
    print(text)


def attack():
    host = HostEpisode(EP, TruthGrader(EP))
    res = run_policy_isolated(host, "tests.attackers:HiddenReader", seed=1)
    report = host.env.submission[0]
    print(f"  попыток прочитать эталон/каталог эпизода: {len(report['leaks']) + len(report['blocked'])}")
    print(f"  утекло: {len(report['leaks'])}    заблокировано: {len(report['blocked'])}")
    print(f"  статус агента: {res.agent_status}")


if __name__ == "__main__":
    sys.exit(main())
