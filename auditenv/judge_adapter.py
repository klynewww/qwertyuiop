"""Адаптер жюри (Genesis FinTech): единые точки входа generate_episode / make_env / score_episode.

Протокол подключения описан в README («Подключение жюри»). Коротко:
  * generate_episode(seed, config, output_dir): плоский каталог эпизода, открытые файлы
    (manifest, ledger, invoices, payments, fx, policy) и закрытые (truth, provenance) лежат рядом;
  * make_env(): объект с reset(seed, config) и step({"name", "args"});
  * score_episode(episode_dir, findings, inspection_steps): закрытый вызов оценщика.

Граница доступа: среда читает ТОЛЬКО открытые файлы из белого списка (env/loader.py).
В закрытом режиме truth не открывается вообще, награда в ответе политике всегда 0,
истинная награда считается отдельно (score_episode). В режиме train truth лежит на
стороне хоста (во временном каталоге генерации) и нужен лишь для терминальной награды.
"""
import json
import shutil
import tempfile
from pathlib import Path

from env.environment import AuditEnv, MAX_STEPS
from env.layout import open_dir_of
from evaluator import TruthGrader
from evaluator.schema import validate_findings
from generator.generate_v1 import build_episode, write_episode

TABLES = ("ledger", "invoices", "payments", "fx")
TASK = ("Вы аудитор. Прочитайте книги компании инструментами list_records, get_record, get_document, "
        "calculator, найдите ошибки (duplicate_invoice, missing_or_partial_allocation, "
        "wrong_date_or_fx_rate, period_cutoff) и отправьте их одним submit в формате findings. "
        "Правила компании в policy. Лимит: 100 действий, включая submit.")
MAX_ERROR_TEXT = 200


# ---------------------------------------------------------------- генерация

def generate_episode(seed: int, config: dict, output_dir: str) -> None:
    """Плоский эпизод: открытые и закрытые файлы в одном каталоге (дальше его делит жюри)."""
    episode = build_episode(int(seed), dict(config or {}))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for part in ("public", "private"):
        for name, data in episode[part].items():
            (out / name).write_bytes(data)


# ---------------------------------------------------------------- оценщик

def score_episode(episode_dir: str, findings, inspection_steps: int) -> dict:
    """Закрытый вызов оценщика. Невалидная структура: return=-1, остальные поля None."""
    return TruthGrader(episode_dir).score(findings, int(inspection_steps))


# ---------------------------------------------------------------- среда

def _bounded(text) -> str:
    return str(text)[:MAX_ERROR_TEXT]


class JudgeEnv:
    def __init__(self, clock=None):
        self._inner = AuditEnv(clock=clock) if clock else AuditEnv()
        self._mode = None
        self._grader = None      # только в train: терминальная награда
        self._tmp = None
        self._done = False
        self._last_flags = (False, False)
        self._tables = {}
        self._submission_valid = None

    # ------- публичное
    def reset(self, seed=None, config=None):
        self._cleanup()
        cfg = dict(config or {})
        mode = cfg.get("mode", "train")
        team = dict(cfg.get("team_config") or {})
        inner_cfg = {k: team[k] for k in ("max_steps", "time_limit_s") if k in team}
        episode_dir = cfg.get("episode_dir")
        self._grader = None
        if mode == "closed":
            if episode_dir is None:
                raise ValueError("в закрытом режиме нужен episode_dir")
            open_dir = open_dir_of(episode_dir)   # truth не открывается: загрузчик читает только белый список
        elif episode_dir is not None:
            open_dir = open_dir_of(episode_dir)
            self._grader = TruthGrader(episode_dir)
        else:
            self._tmp = tempfile.mkdtemp(prefix="auditenv_train_")
            gen = {k: v for k, v in team.items() if k not in ("max_steps", "time_limit_s")}
            generate_episode(0 if seed is None else seed, gen, self._tmp)
            open_dir = open_dir_of(self._tmp)
            self._grader = TruthGrader(self._tmp)
        self._mode = mode
        inner_cfg["open_dir"] = str(open_dir)
        obs, info = self._inner.reset(seed, inner_cfg)
        self._done = False
        self._last_flags = (False, False)
        manifest = obs["manifest"]
        counts = dict(manifest.get("counts", {}))
        counts["documents"] = len(self._inner._ep.company["documents"])
        manifest = {**manifest, "counts": counts}
        first = {"task": TASK, "manifest": manifest, "policy": obs["policy"],
                 "remaining_steps": obs["steps_left"]}
        return first, {"step": 0, "mode": mode}

    def step(self, action):
        if self._inner._ep is None:
            raise RuntimeError("Сначала вызовите reset()")
        if self._done:
            return self._after_done()
        name, args = self._unwrap(action)
        findings_bad = None
        if name == "submit":
            findings = args.get("findings") if isinstance(args, dict) else None
            errs = validate_findings(findings) if "findings" in args else ["нет поля findings"]
            findings_bad = errs or None
        internal = {**args, "action": name} if isinstance(name, str) else name
        raw, _, terminated, truncated, info = self._inner.step(internal)
        result, error = self._translate(name, raw)
        if name == "submit" and error is None:
            self._submission_valid = findings_bad is None
            if findings_bad:
                result = None
                error = {"code": "INVALID_STRUCTURE", "message": _bounded(findings_bad[0])}
            else:
                result = {"accepted": True}
        self._done = terminated or truncated
        self._last_flags = (terminated, truncated)
        reward = 0.0
        if self._mode == "train" and self._done and self._grader is not None:
            findings = self._inner.submission if terminated else {"findings": []}
            reward = self._grader.grade(findings, self._inner.inspection_steps)
        obs = {"result": result, "error": error, "remaining_steps": info["steps_left"]}
        return obs, reward, terminated, truncated, {"step": info["step"]}

    # ------- хост-сторона (политика этих полей не видит)
    @property
    def submission(self):
        return self._inner.submission

    @property
    def inspection_steps(self) -> int:
        return self._inner.inspection_steps

    @property
    def elapsed_s(self) -> float:
        return self._inner.elapsed_s

    def close(self):
        self._cleanup()

    # ------- внутреннее
    @staticmethod
    def _unwrap(action):
        if (isinstance(action, dict) and isinstance(action.get("name"), str)
                and isinstance(action.get("args", {}), dict)):
            return action["name"], dict(action.get("args", {}))
        return None, {}   # плохая оболочка: шаг тратится, ошибка INVALID_ACTION

    def _translate(self, name, raw):
        if "error" in raw:
            e = raw["error"]
            return None, {"code": e["code"], "message": _bounded(e["message"])}
        if name == "list_records":
            return {"records": raw["records"], "next_cursor": raw["next_cursor"]}, None
        if name == "get_record":
            return raw["record"], None
        if name == "get_document":
            return raw["document"], None
        if name == "calculator":
            return raw["result"], None
        return raw, None  # submit: разбирается выше

    def _after_done(self):
        """Шаг после завершения: ничего не меняет и не падает."""
        t, tr = self._last_flags
        obs = {"result": None, "error": {"code": "EPISODE_DONE", "message": "Эпизод завершён"},
               "remaining_steps": 0}
        return obs, 0.0, t, tr, {"step": self._inner._ep.steps}

    def _cleanup(self):
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None


def make_env(clock=None):
    return JudgeEnv(clock=clock)


# ---------------------------------------------------------------- oracle (сторона жюри)

def oracle_submit_action(episode_dir: str) -> dict:
    """Правильный submit из truth. Вызывается только жюри/тестами, не политикой."""
    from policies.oracle import findings_from_truth
    from env.layout import hidden_dir_of
    with open(hidden_dir_of(episode_dir) / "truth.json", encoding="utf-8") as f:
        truth = json.load(f)
    return {"name": "submit", "args": {"findings": {"findings": findings_from_truth(truth)}}}
