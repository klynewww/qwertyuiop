"""Временный оценщик-заглушка (настоящий судья делает участник C)."""
import json
from pathlib import Path

MINI = Path(__file__).parent / "fixtures" / "mini"


class StubGrader:
    """1 - 0.002*steps за точное совпадение с эталоном, иначе -1."""

    def __init__(self, episode_dir):
        with open(Path(episode_dir) / "hidden" / "findings.json", encoding="utf-8") as f:
            self._expected = json.load(f)["findings"]

    def grade(self, findings, inspection_steps):
        got = findings["findings"] if isinstance(findings, dict) else findings
        if got == self._expected:
            return max(-1.0, min(1.0, 1.0 - 0.002 * inspection_steps))
        return -1.0
