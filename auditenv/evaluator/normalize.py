"""Представление вывода и ошибки для сравнения.

Соглашения жюри: ID, type и currency сравниваются ТОЧНО (регистр и пробелы важны,
синонимов и конвертации нет). affected_ids и evidence_ids это множества точных ID:
порядок и повторы не меняют множество. Вызывать только на структурно верных выводах.
"""


def finding_view(f: dict) -> dict:
    return {
        "type": f["type"],
        "affected": frozenset(f["affected_ids"]),
        "evidence": frozenset(f["evidence_ids"]),
        "amount": f["claimed_amount_minor"],
        "currency": f["currency"],
    }


def truth_view(e: dict) -> dict:
    sets = [frozenset(s) for s in e.get("evidence_sets", [])]
    return {
        "issue_id": e.get("issue_id", ""),
        "type": e["type"],
        "affected": frozenset(e["affected_ids"]),
        "amount": int(e["amount_minor"]),
        "currency": e["currency"],
        "evidence_sets": sets,
        "tolerance": int(e.get("tolerance_minor", 0)),
    }
