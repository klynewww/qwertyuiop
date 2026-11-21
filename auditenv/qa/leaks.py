"""Сканер утечек: ищет содержимое закрытых файлов в том, что видит агент.

«Секрет» = ключ или строковое значение из private/ (hidden/), которого НЕТ в
открытых файлах (ID счетов и дат, которые есть в данных, секретом не считаются),
плюс канарейки CANARY-* (метки, подложенные в закрытые файлы специально).
Сканируются наблюдения, ошибки и info; в траекториях действия агента не смотрим
(там агент сам пишет типы ошибок), смотрим observation и info.

  python -m qa.leaks --episode DIR [--trajectories FILE]
"""
import argparse
import json
import secrets
import shutil
import sys
from pathlib import Path

from env.layout import hidden_dir_of, open_dir_of

MIN_TOKEN = 5


def _walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)
    elif isinstance(obj, str):
        yield obj


# Названия четырёх типов ошибок публичны (они есть в задании и в README), секретом не считаются.
# Названия сценариев взаимодействия остаются секретом.
PUBLIC_NAMES = {"duplicate_invoice", "missing_or_partial_allocation", "wrong_date_or_fx_rate", "period_cutoff"}


def _split_files(root: Path):
    """(открытые файлы, закрытые файлы). В плоской раскладке делит белый список имён."""
    open_dir, hidden_dir = open_dir_of(root), hidden_dir_of(root)
    if open_dir != hidden_dir:
        return ([p for p in sorted(open_dir.rglob("*")) if p.is_file()],
                [p for p in sorted(hidden_dir.rglob("*")) if p.is_file()])
    from env.loader import OPEN_FILES
    pub = [p for p in sorted(root.rglob("*")) if p.is_file()
           and (p.name in OPEN_FILES and p.parent == root or "docs" in p.relative_to(root).parts[:1])]
    return pub, [p for p in sorted(root.rglob("*")) if p.is_file() and p not in pub]


def secret_tokens(episode_dir) -> dict[str, str]:
    """token -> файл, где он лежит."""
    root = Path(episode_dir)
    pub_files, hidden_files = _split_files(root)
    public_text = "\n".join(p.read_text(encoding="utf-8") for p in pub_files)
    tokens: dict[str, str] = {}
    for p in hidden_files:
        raw = p.read_text(encoding="utf-8")
        try:
            values = list(_walk(json.loads(raw)))
        except ValueError:
            values = [ln.strip() for ln in raw.splitlines()]
        for tok in values:
            if any(tok in name for name in PUBLIC_NAMES):  # публичные имена и их части («partial», «missing»)
                continue
            if len(tok) >= MIN_TOKEN and (tok not in public_text or tok.startswith("CANARY-")):
                tokens.setdefault(tok, p.name)
    return tokens


def scan_text(text: str, tokens: dict[str, str]) -> list[tuple[str, str]]:
    return sorted((tok, src) for tok, src in tokens.items() if tok in text)


def scan_observations(observations, tokens) -> list[tuple[str, str]]:
    return scan_text(json.dumps(list(observations), ensure_ascii=False), tokens)


def scan_trajectories(path, tokens) -> list[tuple[str, str]]:
    """Смотрит observation и info каждой строки журнала."""
    found = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        part = json.dumps({"observation": row.get("observation"), "info": row.get("info")},
                          ensure_ascii=False)
        found.update(scan_text(part, tokens))
    return sorted(found)


def plant_canary(src_episode, dst_episode) -> str:
    """Копия эпизода с канарейкой в truth.json и отдельным файлом. Возвращает метку."""
    src, dst = Path(src_episode), Path(dst_episode)
    shutil.copytree(src, dst)
    canary = "CANARY-" + secrets.token_hex(6)
    hidden = hidden_dir_of(dst)
    truth_path = hidden / "truth.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth["canary"] = canary
    truth_path.write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    (hidden / "canary.txt").write_text(canary + "\n", encoding="utf-8")
    return canary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--trajectories", required=True)
    args = ap.parse_args(argv)
    tokens = secret_tokens(args.episode)
    found = scan_trajectories(args.trajectories, tokens)
    for tok, src in found:
        print(f"УТЕЧКА: {tok!r} (из {src})")
    print(f"проверено секретов: {len(tokens)}, утечек: {len(found)}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
