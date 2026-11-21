"""Режим «агент в отдельном процессе».

Хост (этот процесс) держит среду, оценщик и truth. Политика работает в
дочернем процессе и общается с хостом только строками JSON (наблюдение
туда, действие обратно). У дочернего процесса нет ни ссылки на среду, ни
на оценщика, а папка hidden/ закрыта песочницей (см. agent_runner.py).
Если агент завис, упал или нарушил протокол, эпизод усекается и ответ
считается пустым.
"""
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from .harness import EpisodeResult, HostEpisode

ROOT = Path(__file__).resolve().parent.parent
RUNNER = Path(__file__).resolve().parent / "agent_runner.py"
MAX_LINE = 1_000_000
STARTUP_TIMEOUT_S = 10.0
_SPEC = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$")
ALLOWED_MODULE_PREFIXES = ("policies.", "tests.attackers")


class AgentFailure(Exception):
    status = "crashed"


class AgentTimeout(AgentFailure):
    status = "timeout"


class AgentProtocolError(AgentFailure):
    status = "protocol_error"


class AgentProcess:
    def __init__(self, spec: str, kwargs: dict | None, deny_paths, prefixes=ALLOWED_MODULE_PREFIXES):
        if not _SPEC.match(spec) or not spec.startswith(tuple(prefixes)):
            raise ValueError(f"Недопустимая политика: {spec!r}")
        self.spec = spec
        self.kwargs = kwargs or {}
        self.deny = [str(Path(p).resolve()) for p in deny_paths]
        self._proc = None
        self._lines: queue.Queue = queue.Queue()
        self._stderr = tempfile.TemporaryFile()

    def start(self, timeout: float = STARTUP_TIMEOUT_S):
        env = {k: os.environ[k] for k in ("SYSTEMROOT", "PATH") if k in os.environ}
        cmd = [sys.executable, "-E", "-s", "-P", "-X", "utf8", str(RUNNER),
               str(ROOT), self.spec, json.dumps(self.kwargs), *self.deny]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            cwd=tempfile.gettempdir(), env=env, text=True, encoding="utf-8", bufsize=1,
        )
        threading.Thread(target=self._read_loop, daemon=True).start()
        msg = self._receive(timeout)
        if msg.get("ready") is not True:
            raise AgentProtocolError("агент не сообщил о готовности")

    def act(self, observation, timeout: float):
        try:
            self._proc.stdin.write(json.dumps(observation, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            raise AgentFailure("агент недоступен") from None
        msg = self._receive(timeout)
        if "error" in msg:
            raise AgentFailure("политика упала")
        if "action" not in msg:
            raise AgentProtocolError("в ответе агента нет action")
        return msg["action"]

    def close(self):
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except OSError:
                pass
            for stream in (proc.stdin, proc.stdout):
                try:
                    stream.close()
                except Exception:
                    pass
            proc.wait()
        self._stderr.close()

    # ---------- внутреннее ----------
    def _read_loop(self):
        out = self._proc.stdout
        try:
            while True:
                line = out.readline(MAX_LINE + 1)
                if not line:
                    self._lines.put(("eof", None))
                    return
                if len(line) > MAX_LINE:
                    self._lines.put(("too_long", None))
                    return
                self._lines.put(("line", line))
        except Exception:
            self._lines.put(("eof", None))

    def _receive(self, timeout: float) -> dict:
        try:
            kind, line = self._lines.get(timeout=max(timeout, 0.0))
        except queue.Empty:
            raise AgentTimeout("агент не ответил вовремя") from None
        if kind == "eof":
            raise AgentFailure("агент завершился")
        if kind == "too_long":
            raise AgentProtocolError("слишком длинная строка")
        try:
            msg = json.loads(line)
        except ValueError:
            raise AgentProtocolError("ответ агента не JSON") from None
        if not isinstance(msg, dict):
            raise AgentProtocolError("ответ агента не объект")
        return msg


def run_policy_isolated(host: HostEpisode, spec: str, kwargs: dict | None = None,
                        seed=None, config=None) -> EpisodeResult:
    """Один эпизод с политикой в отдельном процессе. Не бросает исключений из-за агента."""
    agent = AgentProcess(spec, kwargs, deny_paths=[host.hidden_dir])
    status = "ok"
    try:
        obs, info = host.reset(seed, config)
        try:
            agent.start()
        except AgentFailure as e:
            status = e.status
            obs, reward, terminated, truncated, info = host.expire("AGENT_FAILURE")
            return _result(host, reward, info, terminated, truncated, status)
        while True:
            try:
                action = agent.act(obs, timeout=host.env.time_left())
            except AgentFailure as e:
                status = e.status
                reason = "TIMEOUT" if isinstance(e, AgentTimeout) else "AGENT_FAILURE"
                obs, reward, terminated, truncated, info = host.expire(reason)
                break
            obs, reward, terminated, truncated, info = host.step(action)
            if terminated or truncated:
                if host.env.timed_out:
                    status = "timeout"
                break
        return _result(host, reward, info, terminated, truncated, status)
    finally:
        agent.close()


def _result(host, reward, info, terminated, truncated, status) -> EpisodeResult:
    return EpisodeResult(reward, info["step"], terminated, truncated,
                         host.env.elapsed_s, host.env.inspection_steps, status)
