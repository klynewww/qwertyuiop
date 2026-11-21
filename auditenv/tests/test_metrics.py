import pytest

from evaluator.metrics import compute_metrics, reward


def test_perfect():
    m = compute_metrics(1, 1, 1)
    assert (m.precision, m.recall, m.f1) == (1, 1, 1)


def test_empty_truth_empty_findings_is_perfect():
    m = compute_metrics(0, 0, 0)
    assert (m.precision, m.recall, m.f1) == (0, 0, 1)  # P=R=0, F1=1 по специальному правилу


def test_empty_truth_with_findings_is_false_alarm():
    m = compute_metrics(2, 0, 0)
    assert (m.precision, m.recall, m.f1, m.fp) == (0, 0, 0, 2)


def test_truth_but_no_findings():
    m = compute_metrics(0, 3, 0)
    assert (m.precision, m.recall, m.f1, m.fn) == (0, 0, 0, 3)


def test_partial():
    m = compute_metrics(2, 1, 1)  # 1 TP, 1 FP
    assert m.precision == 0.5 and m.recall == 1 and m.f1 == pytest.approx(2 / 3)
    m = compute_metrics(1, 4, 1)  # 1 TP, 3 FN
    assert m.recall == 0.25 and m.f1 == pytest.approx(0.4)


def test_invalid_tp():
    with pytest.raises(ValueError):
        compute_metrics(1, 1, 2)


# награда: ручной расчёт clip(2*F1 - 1 - 0.002*steps, -1, 1)
@pytest.mark.parametrize("f1,steps,expected", [
    (1.0, 0, 1.0),
    (1.0, 10, 0.98),
    (1.0, 100, 0.8),
    (2 / 3, 0, 1 / 3),
    (2 / 3, 25, 1 / 3 - 0.05),
    (0.5, 0, 0.0),
    (0.0, 0, -1.0),
    (0.0, 50, -1.0),   # клип снизу
    (0.1, 500, -1.0),  # клип снизу, большой штраф
])
def test_reward_manual_examples(f1, steps, expected):
    assert reward(f1, steps) == pytest.approx(expected)


def test_reward_negative_steps_rejected():
    with pytest.raises(ValueError):
        reward(1.0, -1)
