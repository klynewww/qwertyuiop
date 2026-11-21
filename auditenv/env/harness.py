"""Хост-сторона: владеет оценщиком, считает награду, пишет журнал.

Агент видит только AuditEnv. Оценщик (с truth) живёт здесь и в среду
не передаётся. Режим «агент в отдельном процессе» см. env/isolated.py.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .environment import AuditEnv
from .layout import hidden_dir_of, open_dir_of


class Grader(Protocol):
    def grade(self, findings, inspection_steps: int) -> float: ...


class TrajectoryLogger:
    """Пишет по строке JSON на каждый шаг: trajectories.jsonl.

    Поля строки: episode_id, step, action, observation, reward,
    terminated, truncated. В наблюдениях только то, что вернула среда,
    то есть открытые данные.
    """

    def __init__(self, path, episode_id: str):
        self.path = Path(path)
        self.episode_id = episode_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, step, action, observation, reward, terminated, truncated):
        row = {
            "episode_id": self.episode_id,
            "step": step,
            "action": action,
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=repr) + "\n")


class HostEpisode:
    def __init__(self, episode_dir, grader: Grader, env: AuditEnv | None = None,
                 log: TrajectoryLogger | None = None):
        self.episode_dir = Path(episode_dir)
        self.open_dir = open_dir_of(self.episode_dir)
        self.hidden_dir = hidden_dir_of(self.episode_dir)
        self.grader = grader
        self.env = env or AuditEnv()
        self.log = log

    def reset(self, seed=None, config=None):
        cfg = dict(config or {})
        cfg["open_dir"] = str(self.open_dir)
        obs, info = self.env.reset(seed, cfg)
        if self.log:
            self.log.write(0, None, obs, 0.0, False, False)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        return self._finish(action, obs, terminated, truncated, info)

    def expire(self, reason: str = "TIMEOUT"):
        """Агент завис или упал: эпизод усечён, ответ пустой."""
        obs, _, terminated, truncated, info = self.env.force_truncate(reason)
        return self._finish(None, obs, terminated, truncated, info)

    def _finish(self, action, obs, terminated, truncated, info):
        reward = 0.0
        if terminated:
            reward = self.grader.grade(self.env.submission, self.env.inspection_steps)
        elif truncated:
            # не успел сдать (шаги или время): считается пустой ответ
            reward = self.grader.grade({"findings": []}, self.env.inspection_steps)
        if self.log:
            self.log.write(info["step"], action, obs, reward, terminated, truncated)
        return obs, reward, terminated, truncated, info


@dataclass(frozen=True)
class EpisodeResult:
    reward: float
    steps: int
    terminated: bool
    truncated: bool
    elapsed_s: float
    inspection_steps: int
    agent_status: str = "ok"  # ok | timeout | crashed | protocol_error


def run_policy(host: HostEpisode, policy, seed=None, config=None):
    """Гоняет политику в этом же процессе, возвращает (награда, число шагов)."""
    r = run_policy_full(host, policy, seed, config)
    return r.reward, r.steps


def run_policy_full(host: HostEpisode, policy, seed=None, config=None) -> EpisodeResult:
    obs, info = host.reset(seed, config)
    while True:
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = host.step(action)
        if terminated or truncated:
            return EpisodeResult(reward, info["step"], terminated, truncated,
                                 host.env.elapsed_s, host.env.inspection_steps)
