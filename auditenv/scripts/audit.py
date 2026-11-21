"""make audit: самооценка проекта по пяти критериям жюри (баллы из 100).

  25  контракт и изоляция
  25  данные и эталон
  25  награда
  15  воспроизводимость
  10  инженерия

Каждый критерий состоит из автоматических проверок равного веса; баллы =
вес критерия × доля пройденных проверок. Это НАША самооценка по честным
измеримым проверкам, а не оценка жюри. Код возврата 1, если хоть одна проверка красная.
"""
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.environment import AuditEnv  # noqa: E402
from env.harness import HostEpisode, TrajectoryLogger, run_policy_full  # noqa: E402
from env.isolated import run_policy_isolated  # noqa: E402
from evaluator import TruthGrader  # noqa: E402
from generator import generate_v1 as gen  # noqa: E402
from policies import SPECS  # noqa: E402
from policies.careful import CarefulPolicy  # noqa: E402
from policies.oracle import OraclePolicy, load_correct_findings  # noqa: E402
from policies.random_explorer import RandomExplorer  # noqa: E402
from qa import leaks  # noqa: E402
from qa.evidence import validate_evidence_dir  # noqa: E402
from qa.generated import validate_generated_dir  # noqa: E402
from scripts.run_policies import find_episodes, run_one  # noqa: E402

GROUPS = [("Контракт и изоляция", 25), ("Данные и эталон", 25), ("Награда", 25),
          ("Воспроизводимость", 15), ("Инженерия", 10)]
CHECKS = []  # (группа, название, функция)


def check(group, name):
    def deco(fn):
        CHECKS.append((group, name, fn))
        return fn
    return deco


_cache = {}


def episodes():
    if "eps" not in _cache:
        _cache["eps"] = find_episodes(ROOT / "data", 20)
    return _cache["eps"]


def results():
    """Прогон всех политик на 20 эпизодах (один раз)."""
    if "res" not in _cache:
        out = {}
        for pol in ("oracle", "careful", "careful_dup", "reconcile", "random", "flag_all"):
            out[pol] = [run_one(pol, ep, i, False, None, {}) for i, ep in enumerate(episodes())]
        _cache["res"] = out
    return _cache["res"]


def mean(pol, key="reward"):
    rows = results()[pol]
    return sum(r[key] for r in rows) / len(rows)


def profile(ep):
    return Path(ep).name.rsplit("_", 1)[-1]


# ======================================================= контракт и изоляция
G1 = "Контракт и изоляция"


@check(G1, "20 эпизодов: все открытые файлы на месте, заголовки и поля по контракту")
def _():
    need_inv = {"doc_id", "invoice_id", "entity_id", "issued_at", "due_at", "currency", "total_minor", "line_items"}
    need_pay = {"doc_id", "payment_id", "timestamp", "currency", "amount_minor", "invoice_refs"}
    for ep in episodes():
        pub = ep / "public"
        names = {p.name for p in pub.iterdir()}
        assert names == {"manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "policy.json"}, names
        assert (pub / "ledger.csv").read_text(encoding="utf-8").splitlines()[0].split(",") == gen.LEDGER_HEADER
        assert (pub / "fx.csv").read_text(encoding="utf-8").splitlines()[0] == "date,source_currency,target_currency,rate"
        for line in (pub / "invoices.jsonl").read_text(encoding="utf-8").splitlines():
            assert set(json.loads(line)) == need_inv
        for line in (pub / "payments.jsonl").read_text(encoding="utf-8").splitlines():
            assert set(json.loads(line)) == need_pay
        assert not (ep / "private" / "truth.json").read_text(encoding="utf-8").startswith("{}")
    return f"{len(episodes())} эпизодов"


@check(G1, "среда не падает от мусорных действий (50 фаззинг-эпизодов), ошибки без технических деталей")
def _():
    from tests.test_fuzz import run_fuzz
    for seed in range(50):
        run_fuzz(seed)
    return "50 эпизодов"


@check(G1, "лимиты: 100 шагов, 60 с, повторная отправка отклоняется")
def _():
    mini = ROOT / "tests" / "fixtures" / "mini" / "open"
    env = AuditEnv()
    env.reset(seed=1, config={"open_dir": str(mini)})
    for i in range(100):
        _, _, term, trunc, _ = env.step({"action": "list_records", "table": "fx"})
    assert trunc and not term
    t = [0.0]
    env = AuditEnv(clock=lambda: t[0])
    env.reset(seed=1, config={"open_dir": str(mini)})
    t[0] = 61
    obs, _, _, trunc, _ = env.step({"action": "submit", "findings": []})
    assert trunc and obs["error"]["code"] == "TIMEOUT" and env.submission is None
    env = AuditEnv()
    env.reset(seed=1, config={"open_dir": str(mini)})
    env.step({"action": "submit", "findings": []})
    try:
        env.step({"action": "submit", "findings": [{}]})
        raise AssertionError("повторная отправка принята")
    except RuntimeError:
        pass


@check(G1, "агент в отдельном процессе не может прочитать private/ (13 приёмов взлома)")
def _():
    host = HostEpisode(episodes()[3], TruthGrader(episodes()[3]))
    run_policy_isolated(host, "tests.attackers:HiddenReader", seed=1)
    answer = host.env.submission[0]
    assert answer["leaks"] == [], answer["leaks"]
    return f"заблокировано {len(answer['blocked'])}"


@check(G1, "сканер утечек: в траекториях 20 эпизодов нет ни одного секрета из private/")
def _():
    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, ep in enumerate(episodes()):
            path = Path(tmp) / f"t{i}.jsonl"
            host = HostEpisode(ep, TruthGrader(ep), log=TrajectoryLogger(path, f"e{i}"))
            run_policy_isolated(host, SPECS["careful"], seed=i)
            tokens = leaks.secret_tokens(ep)
            total += len(tokens)
            assert leaks.scan_trajectories(path, tokens) == [], ep.name
    return f"секретов проверено: {total}"


@check(G1, "сканер находит специально внесённую утечку (канарейка в observation)")
def _():
    from env.loader import load_open_company
    with tempfile.TemporaryDirectory() as tmp:
        ep = Path(tmp) / "ep"
        canary = leaks.plant_canary(episodes()[2], ep)

        def leaky(open_dir, typed=False):
            c = load_open_company(open_dir, typed=typed)
            c["manifest"]["debug"] = canary
            return c

        env = AuditEnv(loader=leaky)
        obs, _ = env.reset(seed=1, config={"open_dir": str(ep / "public")})
        found = leaks.scan_observations([obs], leaks.secret_tokens(ep))
        assert [t for t, _ in found] == [canary]


@check(G1, "калькулятор: 20 вредных выражений дают ограниченную ошибку, код не выполняется")
def _():
    from env.calculator import CalcError, calculate
    bad = ["__import__('os').system('id')", "open('private/truth.json').read()", "eval('1')", "().__class__",
           "x", "abs(1)", "2**9999999", "9**9**9", "1e999", "'a'*10**9", "[1]", "lambda: 1", "1;2", "a=1",
           "(" * 400, "1/0", "True", "None", "0x10", "1_000"]
    for expr in bad:
        try:
            calculate(expr)
            raise AssertionError(f"принято: {expr!r}")
        except CalcError as e:
            assert e.code == "CALC_ERROR" and len(e.message) < 60


@check(G1, "info и наблюдения содержат только безопасные поля (step, steps_left)")
def _():
    env = AuditEnv()
    env.reset(seed=1, config={"open_dir": str(episodes()[1] / "public")})
    *_, info = env.step({"action": "list_records", "table": "ledger"})
    assert set(info) == {"step", "steps_left"}


# ======================================================= данные и эталон
G2 = "Данные и эталон"


@check(G2, "валидатор равенств: 0 дефектов в 20 эпизодах (дебет=кредит, счета, платежи, курсы)")
def _():
    for ep in episodes():
        bad = validate_generated_dir(ep)
        assert not bad, (ep.name, bad[:2])


@check(G2, "валидатор доказательств: каждая ошибка выводится из открытых данных, лишних нет")
def _():
    n = 0
    for ep in episodes():
        bad = validate_evidence_dir(ep)
        assert not bad, (ep.name, bad[:2])
        n += len(json.loads((ep / "private" / "truth.json").read_text(encoding="utf-8"))["issues"])
    return f"ошибок проверено: {n}"


@check(G2, "число ошибок совпадает с профилем (clean 0, easy 2, medium 6, hard 12)")
def _():
    expect = {"clean": 0, "easy": 2, "medium": 6, "hard": 12}
    for ep in episodes():
        n = len(json.loads((ep / "private" / "truth.json").read_text(encoding="utf-8"))["issues"])
        assert n == expect[profile(ep)], (ep.name, n)


@check(G2, "набор покрывает все 4 типа ошибок и оба взаимодействующих сценария")
def _():
    types, scen = set(), set()
    for ep in episodes():
        for i in json.loads((ep / "private" / "truth.json").read_text(encoding="utf-8"))["issues"]:
            types.add(i["type"])
            scen.add(i["scenario"])
    assert types == set(gen.ERROR_TYPES), types
    assert set(gen.SCENARIOS) <= scen, scen


@check(G2, "в данных есть отвлекающие законные случаи (подписки, кредит-ноты, счета у границы)")
def _():
    kinds = {"subscription": 0, "credit": 0}
    for ep in episodes():
        for line in (ep / "public" / "invoices.jsonl").read_text(encoding="utf-8").splitlines():
            inv = json.loads(line)
            d = inv["line_items"][0]["description"]
            kinds["subscription"] += d.startswith("Monthly subscription")
            kinds["credit"] += d.startswith("Credit note for")
    assert kinds["subscription"] > 50 and kinds["credit"] > 5, kinds
    return str(kinds)


@check(G2, "валюты с разным числом знаков (JPY без копеек) и 4 валюты в данных")
def _():
    m = json.loads((episodes()[0] / "public" / "manifest.json").read_text(encoding="utf-8"))
    assert m["currencies"] == {"USD": 2, "EUR": 2, "KZT": 2, "JPY": 0}


@check(G2, "эталон решаем: «дотошный аудитор» по правилам даёт F1 = 1 на всех 20 эпизодах")
def _():
    for r, ep in zip(results()["careful"], episodes()):
        assert r["f1"] == 1.0 and r["fp"] == 0, (ep.name, r["f1"])


@check(G2, "валидаторы ловят порчу: 15 мутаций данных и 13 мутаций эталона")
def _():
    import copy
    import tests.test_qa_data as T
    from env.loader import load_open_company
    from qa.evidence import validate_evidence
    from qa.generated import validate_generated
    ep = episodes()[3]
    muts = [(T.mut_ledger_debit, "UNBALANCED"), (T.mut_invoice_total, "TOTAL_MISMATCH"),
            (T.mut_line_amount, "LINE_MISMATCH"), (T.mut_unknown_invoice_ref, "UNKNOWN_REF"),
            (T.mut_cash, "CASH_MISMATCH"), (T.mut_bad_rate, "BAD_RATE"), (T.mut_drop_fx, "FX_GAP"),
            (T.mut_dup_invoice_id, "DUPLICATE_ID"), (T.mut_count, "COUNT_MISMATCH"),
            (T.mut_unposted_invoice, "INVOICE_NOT_POSTED"), (T.mut_both_sides, "BAD_SIDES"),
            (T.mut_text_amount, "BAD_AMOUNT"), (T.mut_bad_currency, "UNKNOWN_CURRENCY"),
            (T.mut_due_before_issue, "BAD_DATES"), (T.mut_rate_precision, "BAD_RATE")]
    for mut, code in muts:
        c = copy.deepcopy(load_open_company(ep / "public"))
        mut(c)
        assert code in {i.code for i in validate_generated(c)}, mut.__name__
    truth = json.loads((ep / "private" / "truth.json").read_text(encoding="utf-8"))
    base = load_open_company(ep / "public")
    tmuts = [lambda t: t["issues"][0]["affected_ids"].append("INV-999999"),
             lambda t: t["issues"][0].update(expected_amount_minor=t["issues"][0]["expected_amount_minor"] + 5),
             lambda t: t["issues"][0].update(type="made_up"), lambda t: t["issues"][0].update(acceptable_evidence_sets=[]),
             lambda t: t["issues"].pop(0), lambda t: t.pop("issues"),
             lambda t: t["issues"][0]["acceptable_evidence_sets"].append(["DOC-INV-000001", "DOC-INV-000002"]),
             lambda t: t["issues"][0].update(currency="XXX"), lambda t: t["issues"][0].update(expected_amount_minor=0),
             lambda t: t["issues"][1].update(issue_id=t["issues"][0]["issue_id"]),
             lambda t: t["issues"][0]["acceptable_evidence_sets"][0].append("R999999"),
             lambda t: t["issues"].append({**t["issues"][0], "issue_id": "ISS-099", "affected_ids": ["INV-000001", "INV-000002"]}),
             lambda t: t["issues"][0].update(expected_amount_minor=1.5)]
    for mut in tmuts:
        t = copy.deepcopy(truth)
        mut(t)
        assert validate_evidence(copy.deepcopy(base), t), "мутация эталона не поймана"


# ======================================================= награда
G3 = "Награда"


@check(G3, "oracle получает максимум (1.0) на всех 20 эпизодах, включая чистые")
def _():
    assert all(r["reward"] == 1.0 for r in results()["oracle"])


@check(G3, "пустой ответ: −1 на компаниях с ошибкой, +1 на чистых")
def _():
    class Empty:
        def act(self, obs):
            return {"action": "submit", "findings": []}

    for ep in episodes():
        r = run_policy_full(HostEpisode(ep, TruthGrader(ep)), Empty(), seed=1).reward
        assert r == (1.0 if profile(ep) == "clean" else -1.0), (ep.name, r)


@check(G3, "«пометить всё» проваливается: средняя награда ≤ −0.9 и ни одного эпизода выше −0.5")
def _():
    rows = results()["flag_all"]
    assert mean("flag_all") <= -0.9 and max(r["reward"] for r in rows) < -0.5
    return f"средняя {mean('flag_all'):.3f}"


@check(G3, "простая сверка проваливается: средняя награда < 0 и она ошибается на чистых компаниях")
def _():
    clean = [r for r, ep in zip(results()["reconcile"], episodes()) if profile(ep) == "clean"]
    assert mean("reconcile") < 0 and all(r["reward"] < 0 for r in clean)
    return f"средняя {mean('reconcile'):.3f}"


@check(G3, "случайный исследователь проваливается: средняя награда < −0.5")
def _():
    assert mean("random") < -0.5


@check(G3, "лестница сложности: oracle > careful > careful_dup > reconcile > random ≥ flag_all")
def _():
    m = {p: mean(p) for p in results()}
    assert m["oracle"] > m["careful"] > m["careful_dup"] > m["reconcile"] > m["random"] >= m["flag_all"], m
    return " > ".join(f"{p} {m[p]:.2f}" for p in ("oracle", "careful", "careful_dup", "reconcile", "random", "flag_all"))


@check(G3, "формула награды: clip(2·F1 − 1 − 0.002·шаги, −1, 1) считается точно")
def _():
    from evaluator.metrics import reward
    assert reward(1.0, 0) == 1.0 and abs(reward(1.0, 50) - 0.9) < 1e-12
    assert abs(reward(0.5, 10) - (-0.02)) < 1e-12 and reward(0.0, 0) == -1.0 and reward(0.0, 500) == -1.0
    assert reward(1.0, 1000) == -1.0 or reward(1.0, 1000) >= -1.0


@check(G3, "структурно неверные ответы (10 видов) получают −1, а не «пустой ответ»")
def _():
    g = TruthGrader(episodes()[0])  # чистая компания: пустой ответ дал бы +1
    bad = [None, "x", 5, {"findings": "x"}, [None], [{}], [{"type": "t"}],
           [{"type": "t", "affected_ids": ["a"], "amount_minor": "1", "currency": "USD", "explanation": ""}],
           [{"type": "t", "affected_ids": ["a"], "amount_minor": 1, "currency": "USD", "explanation": "", "x": 1}],
           [{"type": "t", "affected_ids": [], "amount_minor": 1, "currency": "USD", "explanation": ""}]]
    for b in bad:
        assert g.grade(b, 0) == -1.0, b


@check(G3, "награда растёт с каждой найденной ошибкой и падает от ложных выводов")
def _():
    ep = episodes()[3]
    g = TruthGrader(ep)
    correct = load_correct_findings(ep)["findings"]
    rs = [g.grade(correct[:k], 0) for k in range(len(correct) + 1)]
    assert rs == sorted(rs) and rs[-1] == 1.0
    assert g.grade(correct + [{**correct[0], "affected_ids": ["INV-000001", "DOC-INV-000001"]}], 0) < 1.0


# ======================================================= воспроизводимость
G4 = "Воспроизводимость"


def digest(ep):
    h = hashlib.sha256()
    for part in ("public", "private"):
        for name in sorted(ep[part]):
            h.update(name.encode() + b"\0" + ep[part][name])
    return h.hexdigest()


@check(G4, "data/ совпадает с генератором побайтно: 20 эпизодов пересозданы по seed и профилю")
def _():
    for seed, profile_name in [(s, p) for split in ("train", "dev") for s, p in zip(gen.SPLIT_SEEDS[split], gen.SPLIT_PROFILES)]:
        ep = gen.build_episode(seed, {"profile": profile_name})
        d = ROOT / "data" / ("train" if seed < 2000 else "dev") / f"ep_{seed}_{profile_name}"
        for part, files in (("public", ep["public"]), ("private", ep["private"])):
            for name, data in files.items():
                assert (d / part / name).read_bytes() == data, (d.name, name)


@check(G4, "независимость от PYTHONHASHSEED и процесса (3 значения)")
def _():
    code = ("from scripts.audit import digest; from generator.generate_v1 import build_episode;"
            "print(digest(build_episode(31, {'profile': 'hard'})))")
    outs = set()
    for hs in ("0", "7", "99999"):
        outs.add(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                                cwd=ROOT, env={**os.environ, "PYTHONHASHSEED": hs}).stdout.strip())
    assert len(outs) == 1 and next(iter(outs)) == digest(gen.build_episode(31, {"profile": "hard"}))


@check(G4, "20 парных эпизодов (другие ID и порядок): награда oracle и метрики careful не меняются")
def _():
    profiles = ["clean", "easy", "medium", "hard"]
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(20):
            seed, prof = 6000 + i, profiles[i % 4]
            a = gen.write_episode(gen.build_episode(seed, {"profile": prof}), Path(tmp) / f"a{i}")
            b = gen.write_episode(gen.build_paired_episode(seed, {"profile": prof}, 100 + i), Path(tmp) / f"b{i}")
            outs = []
            for ep in (a, b):
                g = TruthGrader(ep)
                ro = run_policy_full(HostEpisode(ep, g), OraclePolicy(load_correct_findings(ep)), seed=1).reward
                g2 = TruthGrader(ep)
                run_policy_full(HostEpisode(ep, g2), CarefulPolicy(), seed=1)
                m = g2.last.metrics
                outs.append((ro, m.tp, m.fp, m.fn))
            assert outs[0] == outs[1], (i, outs)


@check(G4, "траектории побайтно одинаковы при повторном запуске с тем же seed (careful, random)")
def _():
    ep = episodes()[3]
    with tempfile.TemporaryDirectory() as tmp:
        for name, mk in (("careful", CarefulPolicy), ("random", lambda: RandomExplorer(seed=4))):
            blobs = []
            for k in range(2):
                p = Path(tmp) / f"{name}{k}.jsonl"
                run_policy_full(HostEpisode(ep, TruthGrader(ep), log=TrajectoryLogger(p, "e")), mk(), seed=4)
                blobs.append(p.read_bytes())
            assert blobs[0] == blobs[1], name


@check(G4, "режим «агент в процессе» и режим «отдельный процесс» дают одинаковый результат")
def _():
    for ep in episodes()[:6]:
        a = run_policy_full(HostEpisode(ep, TruthGrader(ep)), CarefulPolicy(), seed=1)
        b = run_policy_isolated(HostEpisode(ep, TruthGrader(ep)), SPECS["careful"], seed=1)
        assert (a.reward, a.steps) == (b.reward, b.steps), ep.name


@check(G4, "эпизод воссоздаётся по provenance.json (seed + config)")
def _():
    for ep in episodes()[:6]:
        prov = json.loads((ep / "private" / "provenance.json").read_text(encoding="utf-8"))
        again = gen.build_episode(prov["seed"], prov["config"])
        assert again["private"]["truth.json"] == (ep / "private" / "truth.json").read_bytes()


# ======================================================= инженерия
G5 = "Инженерия"


@check(G5, "pytest: все тесты зелёные")
def _():
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=ROOT,
                       capture_output=True, text=True)
    last = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-200:]
    assert p.returncode == 0, last
    return last


@check(G5, "лицензия MIT, зависимости зафиксированы, Dockerfile на Python 3.11 без root")
def _():
    assert "MIT License" in (ROOT / "LICENSE").read_text(encoding="utf-8")
    reqs = [ln for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]
    assert reqs and all("==" in r for r in reqs), reqs
    d = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "python:3.11" in d and "USER " in d and "make" in d


@check(G5, "весь код разбирается как Python 3.11 (без возможностей новее)")
def _():
    n = 0
    for p in ROOT.rglob("*.py"):
        if "runs" in p.parts:
            continue
        ast.parse(p.read_text(encoding="utf-8"), feature_version=(3, 11))
        n += 1
    return f"файлов: {n}"


@check(G5, "в среде и политиках нет eval/exec/compile/pickle, в генераторе нет глобального random")
def _():
    banned = {"eval", "exec", "compile", "__import__"}
    for folder in ("env", "policies", "evaluator"):
        for p in (ROOT / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in banned:
                    raise AssertionError(f"{p.name}: вызов {node.func.id}")
                if isinstance(node, ast.Import) and any(a.name == "pickle" for a in node.names):
                    raise AssertionError(f"{p.name}: pickle")
    tree = ast.parse((ROOT / "generator" / "generate_v1.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "random" and node.attr != "Random"):
            raise AssertionError(f"глобальный random.{node.attr}")


@check(G5, "документация на месте: README, design, failure_analysis, contract-заметки")
def _():
    for f in ("README.md", "docs/design.md", "docs/failure_analysis.md", "docs/env_notes.md"):
        assert len((ROOT / f).read_text(encoding="utf-8")) > 500, f


@check(G5, "make check, make audit, make docker-check определены; Dockerfile копирует все пакеты")
def _():
    mk = (ROOT / "Makefile").read_text(encoding="utf-8")
    for t in ("check:", "audit:", "docker-check:"):
        assert t in mk, t
    d = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    for pkg in ("env/", "policies/", "evaluator/", "qa/", "generator/", "scripts/", "tests/", "data/"):
        assert f"COPY {pkg}" in d, pkg


@check(G5, "нет «зашитых» ответов: политики и среда не читают private/ и truth")
def _():
    for folder in ("policies", "env"):
        for p in (ROOT / folder).rglob("*.py"):
            if p.name in ("oracle.py", "layout.py", "agent_runner.py", "isolated.py", "harness.py"):
                continue  # хостовые файлы знают про private/ по определению
            tree = ast.parse(p.read_text(encoding="utf-8"))
            docs = {id(n.body[0].value) for n in ast.walk(tree)
                    if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                    and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
            for node in ast.walk(tree):  # docstring и комментарии не в счёт, смотрим строки в коде
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                    assert "truth.json" not in node.value and "private" not in node.value, p.name



def _pytest_file(name):
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"tests/{name}"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout.strip().splitlines() or ["?"])[-1]
    return (r.stdout.strip().splitlines() or [""])[-1]


@check(G1, "блок 1 жюри: C01-C20 через judge_adapter (плоский episode_dir, закрытый режим)")
def _():
    return _pytest_file("test_official_contract.py")


@check(G1, "политики не импортируют оценщика и не читают каталог эпизода")
def _():
    for p in (ROOT / "policies").rglob("*.py"):
        if p.name == "oracle.py":
            continue  # oracle работает на стороне жюри
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""] if isinstance(n, ast.ImportFrom) else []
            assert not any(m.split(".")[0] in ("evaluator", "judge_adapter") for m in mods), p.name


@check(G2, "блок 2 жюри: условия зачёта D01-D20 (чистые, дубли, распределение, курс, период, 2 сценария)")
def _():
    return _pytest_file("test_official_episodes.py")


@check(G3, "блок 3 жюри: matcher и награда, тесты 1-20 и примеры M01-M12 (47/150, -871/1050 и др.)")
def _():
    return _pytest_file("test_official_matcher.py")


@check(G4, "блок 4 жюри: пары P01-P20 на копиях episode_dir (порядок 5, ID 5, оба 10)")
def _():
    return _pytest_file("test_official_pairs.py")


# =============================================================== запуск
def main() -> int:
    t0 = time.time()
    per = {g: [] for g, _ in GROUPS}
    for group, name, fn in CHECKS:
        t = time.time()
        try:
            note = fn()
            per[group].append((True, name, note or "", time.time() - t))
        except Exception as e:  # noqa: BLE001
            msg = (str(e) or type(e).__name__)[:160].replace("\n", " ")
            per[group].append((False, name, msg, time.time() - t))
    total, failed = 0.0, 0
    print("=" * 78)
    print("  make audit: самооценка по критериям жюри (это НЕ оценка жюри)")
    print("=" * 78)
    for group, weight in GROUPS:
        rows = per[group]
        ok = sum(1 for r in rows if r[0])
        pts = weight * ok / len(rows)
        total += pts
        failed += len(rows) - ok
        print(f"\n{group}: {pts:.1f} / {weight}   ({ok}/{len(rows)} проверок)")
        for passed, name, note, dt in rows:
            print(f"  {'✓' if passed else '✗'} {name}" + (f"\n      → {note}" if note else ""))
    print("\n" + "-" * 78)
    print(f"ИТОГО: {total:.1f} / 100   (проверок красных: {failed}, время {time.time() - t0:.1f} с)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
