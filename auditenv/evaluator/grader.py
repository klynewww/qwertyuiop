"""Судья: собирает схему, матчер, метрики и награду. Живёт на стороне хоста."""
import json
from dataclasses import dataclass, field
from pathlib import Path

from env.layout import hidden_dir_of, open_dir_of
from env.loader import load_open_company, record_id

from .matcher import match
from .metrics import Metrics, compute_metrics, reward
from .schema import unwrap, validate_findings


@dataclass(frozen=True)
class EvalResult:
    reward: float
    metrics: Metrics | None
    schema_errors: list[str] = field(default_factory=list)


# Структурно неверный ответ получает минимальную награду -1, а не трактуется как пустой
# (иначе на чистой компании мусор давал бы максимум).
INVALID_REWARD = -1.0


def evaluate(findings, truth_errors: list[dict], inspection_steps: int,
             tolerance_minor=None, known_ids=None) -> EvalResult:
    errs = validate_findings(findings)
    if errs:
        return EvalResult(INVALID_REWARD, None, errs)
    items = unwrap(findings)
    pairs = match(items, truth_errors, tolerance_minor, known_ids)
    m = compute_metrics(len(items), len(truth_errors), len(pairs))
    return EvalResult(reward(m.f1, inspection_steps), m)


def errors_from_truth(truth: dict) -> list[dict]:
    """truth.json (§03) -> список ошибок судьи."""
    return [{
        "issue_id": i.get("issue_id", ""),
        "type": i["type"],
        "affected_ids": i["affected_ids"],
        "amount_minor": i["expected_amount_minor"],
        "currency": i["currency"],
        "evidence_sets": i["acceptable_evidence_sets"],
        "tolerance_minor": i.get("amount_tolerance_minor", 0),
    } for i in truth["issues"]]


def known_ids_of(open_dir) -> frozenset:
    """Все ID, которые агент может получить инструментами: записи таблиц и документы."""
    company = load_open_company(open_dir)
    ids = set(company["documents"])
    for name, rows in company["tables"].items():
        ids |= {record_id(name, r) for r in rows}
    return frozenset(ids)


class TruthGrader:
    """Реализует Grader.grade(findings, inspection_steps) -> float из env/harness.py."""

    def __init__(self, episode_dir, tolerance_minor: int | None = None, allowed_types=None):
        root = Path(episode_dir)
        with open(hidden_dir_of(root) / "truth.json", encoding="utf-8") as f:
            self._errors = errors_from_truth(json.load(f))
        self.known_ids = known_ids_of(open_dir_of(root))
        self.tolerance_minor = tolerance_minor  # None: допуск каждой ошибки берётся из truth (по умолчанию 0)
        self.last: EvalResult | None = None

    def grade(self, findings, inspection_steps: int) -> float:
        self.last = evaluate(findings, self._errors, inspection_steps, self.tolerance_minor, self.known_ids)
        return self.last.reward

    def score(self, findings, inspection_steps: int) -> dict:
        """Словарь для judge_adapter.score_episode (метрики null при невалидной структуре)."""
        self.grade(findings, inspection_steps)
        m = self.last.metrics
        if m is None:
            return {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None,
                    "f1": None, "return": -1.0}
        return {"tp": m.tp, "fp": m.fp, "fn": m.fn, "precision": m.precision, "recall": m.recall,
                "f1": m.f1, "return": self.last.reward}
