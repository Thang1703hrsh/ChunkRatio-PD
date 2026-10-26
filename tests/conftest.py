"""Keep pytest temporary artifacts inside the repository's writable output root."""
from pathlib import Path


def pytest_configure() -> None:
    """Create basetemp's parent before pytest initializes tmp_path (§12)."""
    (Path(__file__).resolve().parents[1]/'output').mkdir(exist_ok=True)
