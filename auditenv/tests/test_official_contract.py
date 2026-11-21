"""Блок 1 определений жюри: C01-C20 через judge_adapter (плоский episode_dir, закрытый режим)."""
import json

import pytest

import judge_adapter as ja
from qa.leaks import scan_observations, secret_tokens

CLOSED = lambda d: {"mode": "closed", "episode_dir": str(d), "team_config": {}}


@pytest.fixture(scope="module")
def eps(tmp_path_factory):
    base = tmp_path_factory.mktemp("official")
    out = {}
    for name, seed, cfg in (("C", 7001, {"profile": "clean"}),
                            ("E", 7002, {"n_invoices": 60, "errors": {"duplicate_invoice": 1}}),
                            ("L", 7003, {"profile": "easy"})):
        d = base / name
        ja.generate_episode(seed, cfg, str(d))
        out[name] = d
    return out


def fresh(d):
    env = ja.make_env()
    first, info = env.reset(1, CLOSED(d))
    return env, first, info


def S(env, name, **args):
    return env.step({"name": name, "args": args})


def jsonable(x):
    json.dumps(x, allow_nan=False)
    return True


def table(d, name):
    env, *_ = fresh(d)
    rows, cur = [], None
    while True:
        o = S(env, "list_records", table=name, cursor=cur, limit=50)[0]
        rows += o["result"]["records"]
        cur = o["result"]["next_cursor"]
        if cur is None:
            return rows


def test_c01_reset_pair_json(eps):
    env = ja.make_env()
    out = env.reset(1, CLOSED(eps["C"]))
    assert isinstance(out, tuple) and len(out) == 2 and jsonable(out)


def test_c02_first_observation(eps):
    _, first, info = fresh(eps["C"])
    assert set(first) == {"task", "manifest", "policy", "remaining_steps"} and first["remaining_steps"] == 100
    text = json.dumps([first, info])
    tokens = secret_tokens(eps["C"])
    assert not [t for t in tokens if t in text]
    assert "seed" not in text.lower() and "truth" not in text.lower()


def test_c03_manifest_matches_data(eps):
    _, first, _ = fresh(eps["C"])
    m = first["manifest"]
    for k in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone", "rounding"):
        assert k in m
    c = m["counts"]
    assert c["invoices"] == len(table(eps["C"], "invoices"))
    assert c["payments"] == len(table(eps["C"], "payments"))
    assert c["ledger_rows"] == len(table(eps["C"], "ledger"))
    assert c["fx_rows"] == len(table(eps["C"], "fx"))
    env, *_ = fresh(eps["C"])
    docs = {r["doc_id"] for t in ("invoices", "payments") for r in table(eps["C"], t)}
    assert c["documents"] == len(docs)


def test_c04_step_shape(eps):
    env, *_ = fresh(eps["C"])
    out = S(env, "list_records", table="ledger", cursor=None, limit=5)
    assert len(out) == 5 and jsonable(out)
    obs, reward, term, trunc, info = out
    assert isinstance(term, bool) and isinstance(trunc, bool) and isinstance(reward, (int, float))
    assert obs["remaining_steps"] == 99 and obs["error"] is None


def test_c05_pagination_complete_no_dupes(eps):
    d = eps["L"]
    rows = table(d, "ledger")
    assert len(rows) >= 101
    ids = [r["row_id"] for r in rows]
    assert len(ids) == len(set(ids))
    env, *_ = fresh(d)
    cur, pages = None, 0
    while True:
        o = S(env, "list_records", table="ledger", cursor=cur, limit=50)[0]
        assert len(o["result"]["records"]) <= 50
        pages += 1
        cur = o["result"]["next_cursor"]
        if cur is None:
            break
    assert pages >= 3


def test_c06_same_cursor_same_page(eps):
    env, *_ = fresh(eps["L"])
    c = S(env, "list_records", table="ledger", cursor=None, limit=50)[0]["result"]["next_cursor"]
    a = S(env, "list_records", table="ledger", cursor=c, limit=50)[0]
    b = S(env, "list_records", table="ledger", cursor=c, limit=50)[0]
    assert a["result"] == b["result"] and b["remaining_steps"] == a["remaining_steps"] - 1


def test_c07_get_record(eps):
    r1 = table(eps["C"], "invoices")[0]
    env, *_ = fresh(eps["C"])
    o = S(env, "get_record", table="invoices", id=r1["invoice_id"])[0]
    assert o["result"] == r1 and o["error"] is None
    assert "truth" not in json.dumps(o)


def test_c08_get_document(eps):
    r1 = table(eps["C"], "invoices")[0]
    env, *_ = fresh(eps["C"])
    o = S(env, "get_document", doc_id=r1["doc_id"])[0]
    assert o["error"] is None and r1["invoice_id"] in json.dumps(o["result"])


def test_c09_calculator_decimal(eps):
    env, *_ = fresh(eps["C"])
    from decimal import Decimal
    o = S(env, "calculator", expression="0.1+0.2")[0]
    assert Decimal(o["result"]) == Decimal("0.3") and o["error"] is None


@pytest.mark.parametrize("expr", ["open('/etc/passwd').read()", "__import__('os').system('id')", "a.b", "2**99999999"])
def test_c10_calculator_rejects_non_arithmetic(eps, expr):
    env, *_ = fresh(eps["C"])
    o = S(env, "calculator", expression=expr)[0]
    assert o["error"]["code"] and o["result"] is None
    assert "root:" not in json.dumps(o)


def unknown_id(d):
    known = {r.get("row_id") or r.get("invoice_id") or r.get("payment_id") or r.get("doc_id") for t in ("ledger", "invoices", "payments") for r in table(d, t)}
    u = "NOPE-000000"
    assert u not in known
    return u


def test_c11_get_record_unknown(eps):
    env, *_ = fresh(eps["C"])
    o, _, t, tr, _ = S(env, "get_record", table="invoices", id=unknown_id(eps["C"]))
    assert o["error"] and o["remaining_steps"] == 99 and not t and not tr
    assert S(env, "get_record", table="invoices", id=table(eps["C"], "invoices")[0]["invoice_id"])[0]["error"] is None


def test_c12_get_document_unknown(eps):
    env, *_ = fresh(eps["C"])
    o = S(env, "get_document", doc_id=unknown_id(eps["C"]))[0]
    assert o["error"] and o["remaining_steps"] == 99 and o["result"] is None


def test_c13_unknown_action(eps):
    env, *_ = fresh(eps["C"])
    o = S(env, "drop_tables")[0]
    assert o["error"] and o["remaining_steps"] == 99
    assert env.step("garbage")[0]["error"]
    assert env.inspection_steps == 2


def test_c14_budget_one_per_request(eps):
    env, *_ = fresh(eps["C"])
    rid = table(eps["C"], "invoices")[0]["invoice_id"]
    a = S(env, "get_record", table="invoices", id=rid)[0]["remaining_steps"]
    b = S(env, "get_record", table="invoices", id="NOPE")[0]["remaining_steps"]
    c = S(env, "get_record", table="invoices", id=rid)[0]
    assert (a, b, c["remaining_steps"]) == (99, 98, 97) and c["error"] is None


def test_c15_submit_terminates_and_freezes(eps):
    d = eps["E"]
    env, *_ = fresh(d)
    S(env, "list_records", table="invoices", cursor=None, limit=5)
    o, r, t, tr, _ = env.step(ja.oracle_submit_action(d))
    assert t and not tr and o["result"] == {"accepted": True}
    frozen = (env.submission, env.inspection_steps, ja.score_episode(str(d), env.submission, env.inspection_steps))
    S(env, "submit", findings={"findings": []})
    S(env, "list_records", table="invoices")
    assert (env.submission, env.inspection_steps, ja.score_episode(str(d), env.submission, env.inspection_steps)) == frozen
    assert frozen[2]["tp"] == 1 and frozen[2]["fp"] == 0


def test_c16_submit_as_action_100(eps):
    d = eps["E"]
    env, *_ = fresh(d)
    for _ in range(99):
        S(env, "calculator", expression="1+1")
    o, r, t, tr, _ = env.step(ja.oracle_submit_action(d))
    assert t and not tr and o["error"] is None
    assert env.inspection_steps == 99
    assert ja.score_episode(str(d), env.submission, env.inspection_steps)["tp"] == 1


def test_c17_hundred_reads_truncate(eps):
    d = eps["E"]
    env, *_ = fresh(d)
    for _ in range(99):
        assert not S(env, "calculator", expression="1+1")[3]
    o, r, t, tr, _ = S(env, "calculator", expression="1+1")
    assert tr and not t
    late = env.step(ja.oracle_submit_action(d))
    assert late[2:4] == (False, True) and env.submission is None
    s = ja.score_episode(str(d), {"findings": []} if env.submission is None else env.submission, env.inspection_steps)
    assert s["tp"] == 0 and s["fn"] == 1


def test_c18_time_limit_includes_policy_time(eps):
    now = [0.0]
    env = ja.make_env(clock=lambda: now[0])
    env.reset(1, CLOSED(eps["E"]))
    S(env, "calculator", expression="1+1")
    now[0] = 61.0   # политика «думала» 61 секунду
    o, r, t, tr, _ = S(env, "calculator", expression="1+1")
    assert tr and not t and env.submission is None
    assert env.step(ja.oracle_submit_action(eps["E"]))[3] is True and env.submission is None


def test_c19_closed_mode_hides_reward(eps):
    d = eps["E"]
    env, *_ = fresh(d)
    seen = []
    for _ in range(3):
        seen.append(S(env, "list_records", table="invoices", cursor=None, limit=5))
    seen.append(env.step(ja.oracle_submit_action(d)))
    for obs, reward, t, tr, info in seen:
        assert reward == 0 and isinstance(reward, int | float) and not isinstance(reward, bool)
        text = json.dumps([obs, info])
        assert not [x for x in secret_tokens(d) if x in text]
        assert "reward" not in obs and "f1" not in text.lower()
    assert ja.score_episode(str(d), env.submission, env.inspection_steps)["return"] > 0.9


def test_c20_no_leak_over_all_records_and_docs(eps):
    for name in ("C", "E"):
        d = eps[name]
        env, first, info = fresh(d)
        seen = [first, info]
        for t in ja.TABLES:
            cur = None
            while True:
                o, _, _, _, i = S(env, "list_records", table=t, cursor=cur, limit=50)
                seen += [o, i]
                cur = o["result"]["next_cursor"]
                if cur is None:
                    break
        for t in ("invoices", "payments"):
            for r in table(d, t)[:20]:
                seen.append(S(env, "get_document", doc_id=r["doc_id"])[0])
        for bad in (S(env, "get_record", table="ledger", id="x"), S(env, "calculator", expression="open('f')"),
                    S(env, "get_document", doc_id="../truth.json"), S(env, "list_records", table="../hidden")):
            seen += [bad[0], bad[4]]
        assert scan_observations(seen, secret_tokens(d)) == []


def test_closed_mode_never_opens_truth(eps, monkeypatch):
    import builtins
    opened = []
    real = builtins.open

    def spy(file, *a, **k):
        opened.append(str(file))
        return real(file, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    env = ja.make_env()
    env.reset(1, CLOSED(eps["E"]))
    S(env, "list_records", table="invoices")
    assert not [p for p in opened if "truth" in p or "provenance" in p]


def test_reset_restarts_same_episode(eps):
    env = ja.make_env()
    env.reset(1, CLOSED(eps["E"]))
    S(env, "calculator", expression="1+1")
    first, _ = env.reset(1, CLOSED(eps["E"]))
    assert first["remaining_steps"] == 100 and env.inspection_steps == 0


def test_train_mode_gives_terminal_reward(eps):
    env = ja.make_env()
    env.reset(5, {"mode": "train", "team_config": {"profile": "easy"}})
    S(env, "list_records", table="invoices")
    o, r, t, tr, _ = S(env, "submit", findings={"findings": []})
    assert t and r < 0
    env.close()


def test_invalid_submit_terminates_with_bounded_error(eps):
    env, *_ = fresh(eps["E"])
    o, r, t, tr, _ = S(env, "submit", findings={"findings": [{}] * 31})
    assert t and o["error"]["code"] == "INVALID_STRUCTURE" and len(o["error"]["message"]) <= 200
    s = ja.score_episode(str(eps["E"]), env.submission, env.inspection_steps)
    assert s["return"] == -1 and s["tp"] is None and s["f1"] is None
