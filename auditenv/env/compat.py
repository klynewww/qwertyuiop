"""Оставлено для совместимости: судья C теперь сам понимает формат генератора A."""
from evaluator import TruthGrader


def make_grader(episode_dir):
    return TruthGrader(episode_dir)
