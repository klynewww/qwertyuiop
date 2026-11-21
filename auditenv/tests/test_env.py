import pytest

from env.environment import AuditEnv, MAX_STEPS
from tests.helpers import MINI

OPEN = MINI / "open"


@pytest.fixture
def env():
    e = AuditEnv()
    e.reset(seed=1, config={"open_dir": str(OPEN)})
    return e


def test_reset_observation():
    e = AuditEnv()
    obs, info = e.reset(seed=1, config={"open_dir": str(OPEN)})
    assert obs["tables"] == {"ledger": 5, "invoices": 3, "payments": 2, "fx": 1}
    assert obs["steps_left"] == MAX_STEPS
    assert obs["manifest"]["company_id"] == "mini-001"
    assert info == {"step": 0, "steps_left": MAX_STEPS}


def test_reset_requires_open_dir():
    with pytest.raises(ValueError):
        AuditEnv().reset(seed=1, config={})


def test_step_before_reset():
    with pytest.raises(RuntimeError):
        AuditEnv().step({"action": "list_records", "table": "ledger"})


def test_list_records_and_cursor(env):
    obs, r, term, trunc, info = env.step({"action": "list_records", "table": "invoices", "limit": 2})
    assert [x["invoice_id"] for x in obs["records"]] == ["INV-0001", "INV-0002"]
    assert obs["next_cursor"] == "c2" and obs["total"] == 3
    assert (term, trunc, info["step"]) == (False, False, 1)
    obs, *_ = env.step({"action": "list_records", "table": "invoices", "limit": 2, "cursor": "c2"})
    assert [x["invoice_id"] for x in obs["records"]] == ["INV-0003"]
    assert obs["next_cursor"] is None


def test_cursor_is_stable_across_calls(env):
    a, *_ = env.step({"action": "list_records", "table": "ledger", "limit": 2, "cursor": "c1"})
    b, *_ = env.step({"action": "list_records", "table": "ledger", "limit": 2, "cursor": "c1"})
    assert a == b


@pytest.mark.parametrize("cursor", ["x", "c-1", "c99", "", 5])
def test_invalid_cursor(env, cursor):
    obs, *_ = env.step({"action": "list_records", "table": "ledger", "cursor": cursor})
    assert obs["error"]["code"] == "INVALID_CURSOR"


@pytest.mark.parametrize("limit", [0, -1, "5", True])
def test_invalid_limit(env, limit):
    obs, *_ = env.step({"action": "list_records", "table": "ledger", "limit": limit})
    assert obs["error"]["code"] == "INVALID_ARGUMENT"


def test_limit_above_50_is_clamped(env):
    obs, *_ = env.step({"action": "list_records", "table": "ledger", "limit": 500})
    assert "records" in obs and len(obs["records"]) <= 50


def test_get_record(env):
    obs, *_ = env.step({"action": "get_record", "table": "payments", "id": "PAY-0002"})
    assert obs["record"]["invoice_id"] == "INV-0002"


def test_unknown_table_and_id_spend_a_step(env):
    obs, _, _, _, info = env.step({"action": "get_record", "table": "nope", "id": "x"})
    assert obs["error"]["code"] == "UNKNOWN_TABLE" and info["step"] == 1
    obs, _, _, _, info = env.step({"action": "get_record", "table": "invoices", "id": "INV-9999"})
    assert obs["error"]["code"] == "UNKNOWN_ID" and info["step"] == 2


def test_get_document(env):
    obs, *_ = env.step({"action": "get_document", "doc_id": "DOC-0003"})
    assert "INV-0003" in obs["document"]["text"]
    obs, *_ = env.step({"action": "get_document", "doc_id": "DOC-9999"})
    assert obs["error"]["code"] == "UNKNOWN_DOCUMENT"


def test_returned_records_are_copies(env):
    obs, *_ = env.step({"action": "get_record", "table": "invoices", "id": "INV-0001"})
    obs["record"]["amount_minor"] = 1
    obs2, *_ = env.step({"action": "get_record", "table": "invoices", "id": "INV-0001"})
    assert obs2["record"]["amount_minor"] == 100000


@pytest.mark.parametrize("bad", [None, "list", [], {}, {"action": None}, {"action": "drop_tables"}])
def test_bad_actions_return_error(env, bad):
    obs, _, term, trunc, info = env.step(bad)
    assert obs["error"]["code"] in {"INVALID_ACTION", "UNKNOWN_ACTION"}
    assert not term and not trunc and info["step"] == 1


def test_calculator_basic(env):
    obs, _, term, trunc, info = env.step({"action": "calculator", "expression": "(1000.50 + 499.50) / 3"})
    assert obs == {"result": "500"} and not term and not trunc and info["step"] == 1


def test_submit_terminates(env):
    obs, _, term, trunc, _ = env.step({"action": "submit", "findings": []})
    assert obs == {"submitted": True} and term and not trunc
    with pytest.raises(RuntimeError):
        env.step({"action": "list_records", "table": "ledger"})


def test_submit_without_findings_is_error_not_end(env):
    obs, _, term, *_ = env.step({"action": "submit"})
    assert obs["error"]["code"] == "INVALID_ARGUMENT" and not term


def test_step_limit_truncates():
    e = AuditEnv()
    e.reset(seed=1, config={"open_dir": str(OPEN), "max_steps": 3})
    for i in range(3):
        _, _, term, trunc, info = e.step({"action": "list_records", "table": "fx"})
    assert trunc and not term and info["steps_left"] == 0
    with pytest.raises(RuntimeError):
        e.step({"action": "list_records", "table": "fx"})


def test_default_limit_is_100():
    e = AuditEnv()
    e.reset(seed=1, config={"open_dir": str(OPEN)})
    for i in range(99):
        _, _, _, trunc, _ = e.step({"action": "list_records", "table": "fx"})
        assert not trunc
    _, _, _, trunc, info = e.step({"action": "list_records", "table": "fx"})
    assert trunc and info["step"] == 100


def test_submit_on_last_step_is_not_truncated():
    e = AuditEnv()
    e.reset(seed=1, config={"open_dir": str(OPEN), "max_steps": 2})
    e.step({"action": "list_records", "table": "fx"})
    _, _, term, trunc, _ = e.step({"action": "submit", "findings": []})
    assert term and not trunc


def test_inspection_steps_count_every_step_but_submit(env):
    env.step({"action": "list_records", "table": "fx"})
    env.step({"action": "get_record", "table": "nope", "id": "x"})  # ошибка тоже шаг
    env.step({"action": "bogus"})  # и неизвестное действие
    env.step({"action": "submit", "findings": []})  # submit не считается
    assert env.inspection_steps == 3


def test_reset_forgets_previous_episode(env):
    env.step({"action": "list_records", "table": "fx"})
    env.step({"action": "submit", "findings": [1]})
    obs, info = env.reset(seed=2, config={"open_dir": str(OPEN)})
    assert info == {"step": 0, "steps_left": MAX_STEPS}
    assert env.submission is None and env.inspection_steps == 0
    _, _, term, trunc, info = env.step({"action": "list_records", "table": "fx"})
    assert info["step"] == 1 and not term and not trunc


def test_info_contains_only_safe_fields(env):
    *_, info = env.step({"action": "list_records", "table": "fx"})
    assert set(info) == {"step", "steps_left"}
