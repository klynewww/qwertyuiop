import random
import string

import pytest

from env.calculator import CalcError, calculate
from env.environment import AuditEnv
from tests.helpers import MINI


@pytest.mark.parametrize("expr,expected", [
    ("1+2", "3"),
    ("10/4", "2.5"),
    ("0.1+0.2", "0.3"),  # Decimal, не float
    ("1000.50*3", "3001.5"),
    ("-(2+3)*4", "-20"),
    ("+5", "5"),
    (".5+.5", "1"),
    ("100", "100"),
    ("0-0", "0"),
    ("1/3", "0.3333333333333333333333333333"),
    ("  7 * 6  ", "42"),
])
def test_valid_expressions(expr, expected):
    assert calculate(expr) == expected


@pytest.mark.parametrize("expr", [
    "__import__('os').system('ls')", "open('hidden/truth.json').read()", "eval('1')",
    "().__class__", "x", "abs(1)", "1 if 1 else 2", "[1,2]", "{1:2}", "'a'", "'a'*3",
    "True", "None", "2**8", "2**9999999", "9**9**9", "7%3", "7//2", "1e5", "0x10", "1_000",
    "1j", "lambda: 1", "(1,2)", "1<2", "not 1", "1;2", "a=1", "import os", "\x00", "",
    "   ", "1+", "(", ")", "1/0", "0/0", "1.2.3", "$", "-", "1 2",
    "1" * 31, "(" * 300 + "1" + ")" * 300, "1+" * 100 + "1",
])
def test_forbidden_or_broken_expressions_raise_bounded_error(expr):
    with pytest.raises(CalcError) as e:
        calculate(expr)
    assert e.value.code == "CALC_ERROR"
    assert len(e.value.message) < 60  # короткая фраза без технических деталей
    assert "Traceback" not in e.value.message and "Error" not in e.value.message


@pytest.mark.parametrize("expr", [None, 5, 1.5, [], {}, b"1+1", True])
def test_non_string_expression(expr):
    with pytest.raises(CalcError):
        calculate(expr)


def test_calculator_does_not_run_code(tmp_path):
    marker = tmp_path / "pwned"
    with pytest.raises(CalcError):
        calculate(f"__import__('pathlib').Path({str(marker)!r}).touch()")
    assert not marker.exists()


def test_env_calculator_spends_a_step_even_on_error():
    env = AuditEnv()
    env.reset(seed=1, config={"open_dir": str(MINI / "open")})
    obs, _, term, trunc, info = env.step({"action": "calculator", "expression": "__import__('os')"})
    assert obs["error"]["code"] == "CALC_ERROR" and info["step"] == 1 and not term and not trunc
    obs, *_ = env.step({"action": "calculator"})  # без expression
    assert obs["error"]["code"] == "CALC_ERROR"
    assert env.inspection_steps == 2  # каждый шаг кроме submit тратит бюджет


def test_random_garbage_never_crashes():
    rng = random.Random(7)
    alphabet = string.printable + "√π€№é‮"
    for _ in range(3000):
        expr = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        try:
            out = calculate(expr)
            assert isinstance(out, str)
        except CalcError:
            pass
