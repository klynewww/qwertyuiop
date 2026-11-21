"""Проверка структуры findings (§05 и определения проверок жюри).

Вывод: type, affected_ids, evidence_ids, claimed_amount_minor, currency, explanation.
Лимиты: 30 выводов, 10 ID доказательств на вывод (считаются ДО удаления повторов),
10 затронутых ID, explanation до 1000 символов Unicode.

Структурная ошибка = неверный тип значения, отсутствующее обязательное поле или
превышение лимита. Она делает неверным весь submit (награда −1). Структурно верный
вывод с неправильным содержанием структурной ошибкой не является: он просто ложный (FP).
Неизвестное значение type структурной ошибкой тоже не является.
"""
MAX_FINDINGS = 30
MAX_IDS = 10
MAX_EVIDENCE = 10
MAX_EXPLANATION = 1000
REQUIRED_FIELDS = ("type", "affected_ids", "evidence_ids", "claimed_amount_minor", "currency", "explanation")


def unwrap(findings):
    """Принимает список выводов или {'findings': [...]}; иначе возвращает None."""
    if isinstance(findings, dict):
        findings = findings.get("findings") if set(findings) == {"findings"} else None
    return findings if isinstance(findings, list) else None


def validate_findings(findings, allowed_types=None) -> list[str]:
    """Список текстовых ошибок структуры. Пустой = структура верна. allowed_types не используется
    (неизвестный тип не структурная ошибка), параметр оставлен для совместимости."""
    items = unwrap(findings)
    if items is None:
        return ["ответ должен быть списком выводов или объектом {'findings': [...]}"]
    errors: list[str] = []
    if len(items) > MAX_FINDINGS:
        errors.append(f"выводов больше {MAX_FINDINGS}: {len(items)}")
    for i, f in enumerate(items):
        errors += [f"[{i}] {e}" for e in _validate_one(f)]
    return errors


def _ids(value, name, limit, allow_empty):
    if not isinstance(value, list):
        return [f"{name} должен быть списком"]
    if len(value) > limit:
        return [f"в {name} больше {limit} ID"]
    if not allow_empty and not value:
        return [f"{name} пуст"]
    if not all(isinstance(x, str) for x in value):
        return [f"каждый ID в {name} должен быть строкой"]
    return []


def _validate_one(f) -> list[str]:
    if not isinstance(f, dict):
        return ["вывод должен быть объектом"]
    missing = [k for k in REQUIRED_FIELDS if k not in f]
    if missing:
        return [f"нет полей: {missing}"]
    errs: list[str] = []
    if not isinstance(f["type"], str):
        errs.append("type должен быть строкой")
    errs += _ids(f["affected_ids"], "affected_ids", MAX_IDS, allow_empty=False)
    errs += _ids(f["evidence_ids"], "evidence_ids", MAX_EVIDENCE, allow_empty=True)
    a = f["claimed_amount_minor"]
    if isinstance(a, bool) or not isinstance(a, int):
        errs.append("claimed_amount_minor должен быть целым числом (не float, не строка, не bool)")
    if not isinstance(f["currency"], str):
        errs.append("currency должна быть строкой")
    e = f["explanation"]
    if not isinstance(e, str):
        errs.append("explanation должен быть строкой")
    elif len(e) > MAX_EXPLANATION:
        errs.append(f"explanation длиннее {MAX_EXPLANATION} символов")
    return errs
