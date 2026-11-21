"""Метрики P, R, F1 и награда.

Граничные случаи (ДОПУЩЕНИЕ, разобрано на бумаге):
  нет ошибок и нет выводов      -> P=R=0, F1=1 (специальное правило жюри: верно промолчали)
  нет ошибок, выводы есть       -> P=0, R=0, F1=0 (ложная тревога)
  ошибки есть, выводов нет      -> P=0, R=0, F1=0 (пропуск всего)
  TP=0 в остальных случаях      -> F1=0
"""
from dataclasses import dataclass

STEP_PENALTY = 0.002


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def compute_metrics(n_findings: int, n_truth: int, tp: int) -> Metrics:
    if not (0 <= tp <= min(n_findings, n_truth)):
        raise ValueError("tp должно быть в пределах [0, min(выводы, ошибки)]")
    fp, fn = n_findings - tp, n_truth - tp
    if n_findings == 0 and n_truth == 0:
        return Metrics(0, 0, 0, 0.0, 0.0, 1.0)  # P=R=0, F1=1 по специальному правилу
    precision = tp / n_findings if n_findings else 0.0
    recall = tp / n_truth if n_truth else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return Metrics(tp, fp, fn, precision, recall, f1)


def reward(f1: float, inspection_steps: int) -> float:
    """clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)."""
    if inspection_steps < 0:
        raise ValueError("inspection_steps не может быть отрицательным")
    return max(-1.0, min(1.0, 2 * f1 - 1 - STEP_PENALTY * inspection_steps))
