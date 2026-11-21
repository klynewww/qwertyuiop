import pytest

from generator.generate_v1 import build_episode, write_episode


@pytest.fixture(scope="session")
def gen(tmp_path_factory):
    """Эпизоды настоящего генератора A по профилям: {'clean': Path, ...}."""
    root = tmp_path_factory.mktemp("gen_session")
    return {p: write_episode(build_episode(4000 + i, {"profile": p}), root / f"ep_{4000 + i}_{p}")
            for i, p in enumerate(["clean", "easy", "medium", "hard"])}
