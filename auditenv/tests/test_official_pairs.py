"""Блок 4 определений жюри: 20 парных проверок P01-P20 на копиях эпизодов (episode_dir)."""
import copy
import json

import pytest

import judge_adapter as ja
from qa.leaks import scan_observations, secret_tokens
from tests.official import MATRIX, build_all, transform, unmap_findings

PAIRS = [(f"P{i + 1:02d}", MATRIX[i][0], "shuffle" if i < 5 else "remap" if i < 10 else "both") for i in range(20)]
CLOSED = lambda d: {"mode": "closed", "episode_dir": str(d), "team_config": {}}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("P")
    eps = build_all(base / "orig")
    out = {}
    for n, (pid, did, kind) in enumerate(PAIRS):
        dst = base / "copy" / pid
        dst.parent.mkdir(exist_ok=True)
        m = transform(eps[did], dst, shuffle=kind in ("shuffle", "both"), remap=kind in ("remap", "both"), seed=500 + n)
        out[pid] = (eps[did], dst, m, kind)
    return out


def play(d, extra_reads=3, oracle=None):
    """Одни и те же действия: чтение трёх страниц, затем submit. Возвращает стенограмму и итог."""
    env = ja.make_env()
    log = []
    first, _ = env.reset(1, CLOSED(d))
    log.append(first)
    for t in ("invoices", "payments", "ledger")[:extra_reads]:
        log.append(env.step({"name": "list_records", "args": {"table": t, "cursor": None, "limit": 50}})[:4])
    action = oracle or ja.oracle_submit_action(d)
    obs, reward, term, trunc, _ = env.step(action)
    log.append((obs, reward, term, trunc))
    score = ja.score_episode(str(d), env.submission, env.inspection_steps)
    return log, score, env.inspection_steps, action["args"]["findings"]


@pytest.mark.parametrize("pid,did,kind", PAIRS)
def test_pair(world, pid, did, kind):
    orig, copy_dir, mapping, _ = world[pid]
    before = {p.name: p.read_bytes() for p in orig.iterdir()}
    # 1. повтор на исходном точно воспроизводит записи, переходы, награды
    log1, score1, steps1, f1 = play(orig)
    log2, score2, steps2, _ = play(orig)
    assert log1 == log2 and score1 == score2 and steps1 == steps2
    # исходный комплект не перезаписан
    assert {p.name: p.read_bytes() for p in orig.iterdir()} == before
    # 2. oracle на копии: те же выводы с новыми ID, после обратного отображения совпадают
    _, score3, steps3, f3 = play(copy_dir)
    ev = lambda fs: sorted((f["type"], tuple(f["affected_ids"]), f["claimed_amount_minor"], f["currency"],
                            tuple(f["evidence_ids"])) for f in fs["findings"])
    assert ev(unmap_findings(f3, mapping)) == ev(f1)
    # 3. метрики и награда совпадают при одинаковом inspection_steps
    assert steps1 == steps3 and score1 == score3
    # 4. нет новых каналов утечки; сломанные доказательства награду меняют
    env = ja.make_env()
    first, _ = env.reset(1, CLOSED(copy_dir))
    seen = [first]
    for t in ja.TABLES:
        seen.append(env.step({"name": "list_records", "args": {"table": t, "cursor": None, "limit": 50}})[0])
    assert scan_observations(seen, secret_tokens(copy_dir)) == []
    if f3["findings"]:
        broken = copy.deepcopy(f3)
        broken["findings"][0]["evidence_ids"] = []
        assert ja.score_episode(str(copy_dir), broken, steps3)["return"] < score3["return"]


def test_ids_really_changed_where_declared(world):
    for pid, (orig, dst, mapping, kind) in world.items():
        a = (orig / "invoices.jsonl").read_text(encoding="utf-8")
        b = (dst / "invoices.jsonl").read_text(encoding="utf-8")
        assert (a != b), pid
        if kind in ("remap", "both"):
            assert mapping and any(k != v for k, v in mapping.items())
        else:
            assert not mapping and sorted(a.splitlines()) == sorted(b.splitlines())
    assert len(world) == 20
