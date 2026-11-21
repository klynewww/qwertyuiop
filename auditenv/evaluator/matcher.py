"""Матчер вывод <-> ошибка (определения проверок жюри).

Вывод это TP для ошибки, если выполнены ВСЕ условия:
  1. type совпадает точно;
  2. множество affected_ids точно равно каноническому множеству ошибки
     (лишний, пусть и существующий, ID не допускается);
  3. |claimed_amount_minor − expected| <= τ (по умолчанию τ = 0, берётся из truth);
  4. currency совпадает точно;
  5. evidence_ids целиком содержат хотя бы один допустимый набор доказательств, и
     каждый указанный ID существует и доступен через инструменты (иначе вывод не
     сопоставляется и даёт FP). Лишние существующие доказательства допустимы.

Сопоставление одно к одному: максимальное паросочетание, при равной мощности
детерминированный выбор: ошибки по issue_id, для каждой наименьший индекс вывода.
Повторный вывод остаётся отдельным FP; explanation на выбор рёбер не влияет.
"""
from .normalize import finding_view, truth_view


def failed_conditions(f: dict, t: dict, tolerance_minor=None, known_ids=None) -> list[str]:
    """Имена нарушенных условий. f и t уже приведены (finding_view, truth_view)."""
    tau = t["tolerance"] if tolerance_minor is None else tolerance_minor
    failed = []
    if f["type"] != t["type"]:
        failed.append("type")
    if f["affected"] != t["affected"]:
        failed.append("ids")
    if abs(f["amount"] - t["amount"]) > tau:
        failed.append("amount")
    if f["currency"] != t["currency"]:
        failed.append("currency")
    unknown = known_ids is not None and not f["evidence"] <= known_ids
    if unknown or not any(s <= f["evidence"] for s in t["evidence_sets"]):
        failed.append("evidence")
    return failed


def is_match(f, t, tolerance_minor=None, known_ids=None) -> bool:
    return not failed_conditions(f, t, tolerance_minor, known_ids)


def match(findings: list[dict], truth_errors: list[dict], tolerance_minor=None, known_ids=None):
    """Сырые выводы и ошибки truth -> отсортированный список пар (индекс вывода, индекс ошибки)."""
    fs = [finding_view(f) for f in findings]
    ts = [truth_view(t) for t in truth_errors]
    known = None if known_ids is None else frozenset(known_ids)
    edges = {j: [i for i, f in enumerate(fs) if is_match(f, t, tolerance_minor, known)] for j, t in enumerate(ts)}

    order = sorted(range(len(ts)), key=lambda j: (ts[j]["issue_id"], j))

    def max_size(issues, blocked) -> int:
        owner: dict[int, int] = {}

        def aug(j, seen):
            for i in edges[j]:
                if i in blocked or i in seen:
                    continue
                seen.add(i)
                if i not in owner or aug(owner[i], seen):
                    owner[i] = j
                    return True
            return False

        return sum(aug(j, set()) for j in issues)

    target = max_size(order, frozenset())
    chosen: dict[int, int] = {}  # вывод -> ошибка
    for pos, j in enumerate(order):
        need = target - len(chosen)
        if need == 0:
            break
        rest = order[pos + 1:]
        for i in edges[j]:  # индексы выводов по возрастанию
            if i in chosen:
                continue
            if max_size(rest, frozenset(chosen) | {i}) == need - 1:
                chosen[i] = j
                break
    return sorted(chosen.items())
