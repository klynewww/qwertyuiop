"""Oracle: сдаёт заранее известный верный ответ.

Ответ ему передаёт ХОСТ (из закрытой папки), а не среда: агентская сторона
ответа не видит. Oracle ничего не читает и сдаёт ответ сразу, то есть с
inspection_steps = 0 (максимальная награда).
"""
import json
from pathlib import Path

from env.layout import hidden_dir_of


def findings_from_truth(truth: dict) -> list:
    """Верный ответ по truth генератора A: канонические affected_ids и один полный набор доказательств."""
    findings = []
    for issue in truth["issues"]:
        findings.append({
            "type": issue["type"],
            "affected_ids": sorted(issue["affected_ids"]),
            "evidence_ids": sorted(issue["acceptable_evidence_sets"][0]),
            "claimed_amount_minor": issue["expected_amount_minor"],
            "currency": issue["currency"],
            "explanation": f"{issue['type']}: {', '.join(issue['affected_ids'])}",
        })
    return findings


def load_correct_findings(episode_dir) -> dict:
    hidden = hidden_dir_of(episode_dir)
    ready = hidden / "findings.json"
    if ready.is_file():
        return json.loads(ready.read_text(encoding="utf-8"))
    truth = json.loads((hidden / "truth.json").read_text(encoding="utf-8"))
    return {"findings": findings_from_truth(truth)}


class OraclePolicy:
    def __init__(self, correct_findings: dict):
        self._findings = correct_findings

    def act(self, observation):
        return {"action": "submit", "findings": self._findings["findings"]}
