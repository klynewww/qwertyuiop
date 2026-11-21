"""Политики-«взломщики» для тестов изоляции. Только для tests/, не для прогонов."""
import glob
import os
import subprocess
import sys
import time
from pathlib import Path


def run_attacks(hidden: str):
    """Пытается прочитать hidden/ разными способами. Возвращает (утечки, заблокировано)."""
    leaks, blocked = [], []
    hidden_p = Path(hidden)
    episode = hidden_p.parent

    def attempt(name, fn):
        try:
            result = fn()
            if result:
                leaks.append(name)
            else:
                blocked.append(name + ":empty")
        except PermissionError:
            blocked.append(name)
        except Exception as e:  # любая другая неудача тоже не утечка
            blocked.append(f"{name}:{type(e).__name__}")

    truth = hidden_p / "truth.json"
    attempt("open_abs", lambda: open(truth, encoding="utf-8").read())
    attempt("open_dotdot", lambda: open(episode / "open" / ".." / "hidden" / "truth.json", encoding="utf-8").read())
    attempt("pathlib_read", lambda: truth.read_text(encoding="utf-8"))
    attempt("listdir", lambda: os.listdir(hidden))
    attempt("scandir", lambda: list(os.scandir(hidden)))
    attempt("glob", lambda: glob.glob(str(hidden_p / "*")))
    attempt("os_open", lambda: os.read(os.open(str(truth), os.O_RDONLY), 100))

    def via_symlink():
        link = Path(os.path.join(os.getcwd(), "lnk_attack"))
        os.symlink(hidden, link)
        return open(link / "truth.json").read()

    attempt("symlink", via_symlink)
    attempt("chdir_relative", lambda: (os.chdir(hidden), open("truth.json").read())[1])
    attempt("subprocess_cat", lambda: subprocess.run([sys.executable, "-c", f"print(open({str(truth)!r}).read())"],
                                                     capture_output=True).stdout)
    attempt("os_system", lambda: os.system(f'{"type" if os.name == "nt" else "cat"} "{truth}"') == 0)
    attempt("truth_grader", lambda: __import__("evaluator").TruthGrader(str(episode)))
    attempt("ctypes", lambda: __import__("ctypes").CDLL(None))
    attempt("env_secret", lambda: os.environ.get("AUDIT_SECRET_FOR_TEST"))
    return leaks, blocked


class HiddenReader:
    def __init__(self, **_):
        pass

    def act(self, obs):
        hidden = next(a for a in sys.argv if a.endswith(("hidden", "private")))
        leaks, blocked = run_attacks(hidden)
        return {"action": "submit", "findings": [{"leaks": leaks, "blocked": blocked}]}


class Hang:
    def __init__(self, **_):
        pass

    def act(self, obs):
        time.sleep(3600)


class Crash:
    def __init__(self, **_):
        pass

    def act(self, obs):
        raise RuntimeError("boom")


class NotJson:
    def __init__(self, **_):
        pass

    def act(self, obs):
        return {"action": {1, 2, 3}}  # set не сериализуется


class HugeLine:
    def __init__(self, **_):
        pass

    def act(self, obs):
        return {"action": "submit", "findings": "x" * 2_000_000}
