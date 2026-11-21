"""Судья: структура ответа, точное сравнение, матчер, метрики, награда."""
from .grader import TruthGrader, evaluate
from .schema import validate_findings

__all__ = ["TruthGrader", "evaluate", "validate_findings"]
