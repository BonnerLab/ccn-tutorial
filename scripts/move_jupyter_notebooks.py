#!/.venv/bin/python

from pathlib import Path
import shutil

for filepath in Path("site").rglob("*.ipynb"):
    shutil.copy(filepath, Path("notebooks") / filepath.name)
