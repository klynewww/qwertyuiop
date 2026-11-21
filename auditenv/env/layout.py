"""Где лежат открытые и закрытые файлы эпизода.

Поддерживаются три раскладки:
  public/ + private/   папки нашего генератора (data/);
  open/ + hidden/      мини-эпизод тестов;
  плоская              все файлы прямо в episode_dir (так жюри подаёт эпизод через
                       generate_episode и копии для парных проверок). Закрытые
                       truth.json и provenance.json лежат рядом, но среда читает
                       только белый список открытых имён (env/loader.py).
"""
from pathlib import Path


def open_dir_of(episode_dir) -> Path:
    root = Path(episode_dir)
    if (root / "public").is_dir():
        return root / "public"
    if (root / "open").is_dir():
        return root / "open"
    return root


def hidden_dir_of(episode_dir) -> Path:
    root = Path(episode_dir)
    if (root / "private").is_dir():
        return root / "private"
    if (root / "hidden").is_dir():
        return root / "hidden"
    return root


def is_episode_dir(path) -> bool:
    p = Path(path)
    if (p / "public").is_dir() and (p / "private").is_dir():
        return True
    if (p / "open").is_dir() and (p / "hidden").is_dir():
        return True
    return (p / "manifest.json").is_file() and (p / "truth.json").is_file()
