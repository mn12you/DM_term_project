from pathlib import Path
from typing import Final

IN_DIR: Final[Path] = Path(__file__).parent/ "database"
OUT_DIR: Final[Path] = Path(__file__).parent / "output"
MODEL_DIR: Final[Path] = Path(__file__).parent / "models"

assert IN_DIR.exists(), f"inputs directory does not exist: {IN_DIR}"
OUT_DIR.mkdir(exist_ok=True)