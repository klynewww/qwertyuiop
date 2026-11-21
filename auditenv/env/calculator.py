"""Безопасный калькулятор: только арифметика над десятичными числами.

Выражение разбирается через ast и проверяется по белому списку узлов;
вычисляет это НЕ eval, а наш обход дерева в Decimal. Разрешено:
числа (123, 12.5, .5), + - * /, унарные + -, скобки. Всё остальное
(имена, вызовы, атрибуты, строки, **, %, экспоненты, файлы) даёт
CalcError с короткой фиксированной фразой без технических деталей.
"""
import ast
import re
import warnings
from decimal import Decimal, DivisionByZero, InvalidOperation, Overflow, localcontext

MAX_EXPR_LEN = 200
MAX_NODES = 60
MAX_LITERAL_LEN = 30
PRECISION = 28

_NUMBER = re.compile(r"^(\d+\.?\d*|\.\d+)$")
_BINARY = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}


class CalcError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _bad(message="Выражение не разрешено"):
    return CalcError("CALC_ERROR", message)


def calculate(expression) -> str:
    """Возвращает результат строкой (обычная запись, без экспоненты)."""
    if not isinstance(expression, str):
        raise _bad("Выражение должно быть строкой")
    if not expression.strip():
        raise _bad("Пустое выражение")
    if len(expression) > MAX_EXPR_LEN:
        raise _bad("Выражение слишком длинное")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # мусорные строки вроде "\\L" дают SyntaxWarning
            tree = ast.parse(expression.strip(), mode="eval")
    except Exception:  # SyntaxError, ValueError (нулевой байт), RecursionError ...
        raise _bad("Не удалось разобрать выражение") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise _bad("Выражение слишком сложное")

    source = expression.strip()
    try:
        with localcontext() as ctx:
            ctx.prec = PRECISION
            ctx.traps[DivisionByZero] = True
            ctx.traps[InvalidOperation] = True
            ctx.traps[Overflow] = True
            value = _eval(tree.body, source)
            return _format(value)
    except CalcError:
        raise
    except DivisionByZero:
        raise _bad("Деление на ноль") from None
    except Exception:  # InvalidOperation, Overflow, RecursionError ...
        raise _bad("Не удалось вычислить") from None


def _eval(node, source: str) -> Decimal:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _bad()
        text = ast.get_source_segment(source, node)
        if text is None or not _NUMBER.match(text) or len(text) > MAX_LITERAL_LEN:
            raise _bad()
        return Decimal(text)  # из текста, а не из float: без потери точности
    if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
        v = _eval(node.operand, source)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_eval(node.left, source), _eval(node.right, source))
    raise _bad()


def _format(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    return text
