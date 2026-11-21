from env.harness import HostEpisode, run_policy
from policies.oracle import OraclePolicy, load_correct_findings
from evaluator import TruthGrader
from tests.helpers import MINI


def make_host():
    return HostEpisode(MINI, TruthGrader(MINI))


def test_oracle_full_cycle_gets_max_reward():
    policy = OraclePolicy(load_correct_findings(MINI))
    reward, steps = run_policy(make_host(), policy, seed=0)
    assert steps == 1
    assert reward == 1.0  # 0 шагов осмотра, штрафа нет


def test_empty_answer_on_faulty_company_is_bad():
    class Empty:
        def act(self, obs):
            return {"action": "submit", "findings": []}

    reward, _ = run_policy(make_host(), Empty(), seed=0)
    assert reward == -1.0


def test_inspection_penalty_applies():
    class ReadThenAnswer:
        def __init__(self):
            self.n = 0
            self.answer = load_correct_findings(MINI)["findings"]

        def act(self, obs):
            self.n += 1
            if self.n <= 5:
                return {"action": "list_records", "table": "invoices"}
            return {"action": "submit", "findings": self.answer}

    reward, steps = run_policy(make_host(), ReadThenAnswer(), seed=0)
    assert steps == 6
    assert abs(reward - (1.0 - 0.002 * 5)) < 1e-9


def test_truncation_grades_empty_answer():
    class Loop:
        def act(self, obs):
            return {"action": "list_records", "table": "fx"}

    reward, steps = run_policy(make_host(), Loop(), seed=0, config={"max_steps": 5})
    assert steps == 5 and reward == -1.0


def test_two_episodes_in_one_process_are_independent():
    host = make_host()
    host.reset(seed=1)
    host.step({"action": "list_records", "table": "fx"})
    host.reset(seed=2)
    _, _, _, _, info = host.step({"action": "list_records", "table": "fx"})
    assert info["step"] == 1
