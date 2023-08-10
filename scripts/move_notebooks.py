#!/.venv/bin/python

from pathlib import Path
import shutil

for file in Path("site").rglob("*.ipynb"):
    shutil.copy(file, Path("notebooks") / file.name)
